# -*- coding: utf-8 -*-
"""환경 설정(시드·폰트·경로)·저장 유틸 — coag/io_utils.py 이식(경로만 교체)."""
import json
import random
import sys
from pathlib import Path

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
        print(f"torch {torch.__version__} / CUDA "
              f"{'사용 (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else '불가 — CPU'}")
    except ImportError:
        print("torch 미설치 — DL(GRU/LSTM) 모델 사용 불가")


def add_coag_path() -> None:
    """02_coag(dose_compare·coag) import 경로 추가 — 원 코드는 무수정."""
    p = str(C.COAG_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_master() -> pd.DataFrame:
    """전처리완료 parquet 로드 → 완전 1분 정규격자 reindex (positional=temporal 보장)."""
    df = pd.read_parquet(C.DATA_PARQUET).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("전처리완료 parquet 인덱스가 DatetimeIndex가 아님")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    return df.reindex(full)


def load_flags() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(C.FLAGS_PARQUET).sort_index()
    except FileNotFoundError:
        return None


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = C.OUTPUT_DIR / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_json(obj, name: str) -> Path:
    path = C.OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(name: str):
    with open(C.OUTPUT_DIR / name, encoding="utf-8") as f:
        return json.load(f)
