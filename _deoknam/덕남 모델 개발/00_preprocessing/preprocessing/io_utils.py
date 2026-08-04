# -*- coding: utf-8 -*-
"""입출력 유틸: 출력 환경 설정, 원본 복사, 로딩/저장."""
import shutil
import sys

import pandas as pd

from . import config as C


def setup_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    C.OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    C.RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def copy_raw_if_missing() -> None:
    if not C.RAW_PARQUET.exists():
        if not C.SOURCE_PARQUET.exists():
            raise FileNotFoundError(f"원본 데이터 없음: {C.SOURCE_PARQUET}")
        shutil.copy2(C.SOURCE_PARQUET, C.RAW_PARQUET)
        print(f"원본 복사: {C.SOURCE_PARQUET} → {C.RAW_PARQUET}")


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(C.RAW_PARQUET).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("raw parquet 인덱스가 DatetimeIndex가 아님")
    return df


def save_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
