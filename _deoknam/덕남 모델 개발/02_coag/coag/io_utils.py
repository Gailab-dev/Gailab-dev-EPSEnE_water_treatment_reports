# -*- coding: utf-8 -*-
"""입출력·환경: setup(시드 고정 포함), 군집 로드, 번들 저장/재로딩."""
import json
import random
import sys

import joblib
import numpy as np
import pandas as pd

from . import config as C


def setup_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    random.seed(C.RANDOM_STATE)
    np.random.seed(C.RANDOM_STATE)
    try:
        import torch

        torch.manual_seed(C.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(C.RANDOM_STATE)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"torch {torch.__version__} / CUDA {'사용' if torch.cuda.is_available() else '불가'}"
              f"{' (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")
    except ImportError:
        print("torch 미설치 — DL 모델 사용 불가")


def load_cluster(cid: int) -> pd.DataFrame:
    df = pd.read_parquet(C.DATA_DIR / f"덕남_cluster{cid}.parquet").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("cluster parquet 인덱스가 DatetimeIndex가 아님")
    return df


def save_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_bundle(cid: int, model_name: str, model_obj, feature_cols: list,
                target_cols: list, scaler_x, scaler_y, meta: dict) -> None:
    """best 모델 번들 저장 (요구 파일명 규약)."""
    d = C.MODELS_DIR / f"cluster{cid}"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, d / "feature_cols.pkl")
    joblib.dump(target_cols, d / "target_cols.pkl")
    joblib.dump(model_obj, d / f"{model_name}_model.pkl")
    joblib.dump(scaler_x, d / "scaler_x.pkl")
    joblib.dump(scaler_y, d / "scaler_y.pkl")
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"모델 번들 저장: {d} ({model_name}, 피처 {len(feature_cols)}개)")


def load_bundle(cid: int) -> dict:
    d = C.MODELS_DIR / f"cluster{cid}"
    with open(d / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    name = meta["model_name"]
    return {
        "feature_cols": joblib.load(d / "feature_cols.pkl"),
        "target_cols": joblib.load(d / "target_cols.pkl"),
        "model": joblib.load(d / f"{name}_model.pkl"),
        "scaler_x": joblib.load(d / "scaler_x.pkl"),
        "scaler_y": joblib.load(d / "scaler_y.pkl"),
        "meta": meta,
    }
