# -*- coding: utf-8 -*-
"""덕남정수장 모델링 공통 모듈: 데이터 로딩, gap-safe 시프트, 지표, 저장."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

COL_DT = "datetime"
RAW_FEATURES = [
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
]
COL_DOSE = "PAC.AI.PAC_주입율제어_SV"
COL_DOSE_S2 = "PAC.AI.PACS2_주입율제어_SV"
COL_FLOW = "RCS_1.AI.FT101"
TARGET_TB = {1: "RCS_6.AI.침전지1_탁도", 2: "RCS_6.AI.침전지2_탁도"}

# Task 1 후보 체류시간(분). 203분은 10분 격자에 없어 200분 사용.
LEAD_CANDIDATES = {
    0: [(180, 180), (200, 200), (210, 210), (230, 230)],
    1: [(150, 150), (180, 180), (203, 200), (210, 210)],
    2: [(210, 210), (240, 240), (260, 260), (270, 270)],
}


def setup_output():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_cluster(cid: int, clean: bool = False) -> pd.DataFrame:
    suffix = "_clean" if clean else ""
    df = pd.read_parquet(DATA_DIR / f"덕남_cluster{cid}{suffix}.parquet")
    return df.sort_values(COL_DT).reset_index(drop=True)


def load_pooled() -> pd.DataFrame:
    frames = []
    for cid in range(3):
        df = load_cluster(cid)
        df["cluster"] = cid
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(COL_DT).reset_index(drop=True)


def shift_target_by_time(df: pd.DataFrame, target_col: str, lead_min: int) -> pd.DataFrame:
    """t+lead 시점의 target_col을 y_future로 결합 (datetime 병합, gap-safe)."""
    fut = df[[COL_DT, target_col]].copy()
    fut[COL_DT] = fut[COL_DT] - pd.Timedelta(minutes=lead_min)
    fut = fut.rename(columns={target_col: "y_future"})
    return df.merge(fut, on=COL_DT, how="left")


def time_ordered_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(COL_DT).reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    smape = 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    mape = 100.0 * np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    return {
        "r2": r2_score(y_true, y_pred),
        "smape": smape,
        "mape": mape,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def passes(m: dict) -> bool:
    return m["r2"] >= 0.9 and m["smape"] <= 10.0


def save_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
