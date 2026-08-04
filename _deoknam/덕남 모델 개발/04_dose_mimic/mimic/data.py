# -*- coding: utf-8 -*-
"""일 단위 결정 테이블 생성: 오늘 조건 + 어제 주입률 → 오늘 주입률 클래스."""
import numpy as np
import pandas as pd

from . import config as C


def build_daily_table() -> pd.DataFrame:
    df = pd.read_parquet(C.IN_PARQUET, columns=C.SRC_COLS + list(C.SV_COLS.values()))
    day = pd.DataFrame({
        "TB": df["RCS_6.AI.착수정_TB"].resample("1D").mean(),
        "TB_max": df["RCS_6.AI.착수정_TB"].resample("1D").max(),
        "PH": df["RCS_6.AI.착수정_PH"].resample("1D").mean(),
        "AL": df["RCS_6.AI.착수정_AL"].resample("1D").mean(),
        "수온": df["RCS_6.AI.착수정_온도"].resample("1D").mean(),
        "전도도": df["RCS_6.AI.착수정_전기전도도"].resample("1D").mean(),
        "정수량": df["주입변경.정수량_천m3d"].resample("1D").mean(),
        "침전지1_전일": df["RCS_6.AI.침전지1_탁도"].resample("1D").mean().shift(1),
        "침전지2_전일": df["RCS_6.AI.침전지2_탁도"].resample("1D").mean().shift(1),
    })
    for k, col in C.SV_COLS.items():
        sv = df[col].resample("1D").last()
        day[f"SV{k}"] = sv.round().clip(C.CLASSES[0], C.CLASSES[-1])
        day[f"SV{k}_어제"] = day[f"SV{k}"].shift(1)
    day["logTB"] = np.log(day["TB"].clip(lower=1e-3))
    day["TB_7일최대"] = day["TB_max"].rolling(7, min_periods=1).max()
    day["TB_변화"] = day["TB"].diff()
    day["수온_변화7일"] = day["수온"].diff(7)
    doy = day.index.dayofyear
    day["계절_sin"] = np.sin(2 * np.pi * doy / 365.25)
    day["계절_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return day.dropna()


FEATURES = ["TB", "logTB", "TB_max", "TB_7일최대", "TB_변화", "PH", "AL",
            "수온", "수온_변화7일", "전도도", "정수량",
            "침전지1_전일", "침전지2_전일", "계절_sin", "계절_cos"]


def split(day: pd.DataFrame):
    cut = int(len(day) * (1 - C.TEST_FRAC))
    return day.iloc[:cut], day.iloc[cut:]
