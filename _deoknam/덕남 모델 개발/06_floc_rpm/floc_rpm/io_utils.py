# -*- coding: utf-8 -*-
"""입출력·환경: parquet 로드 + 1분 전체 그리드 정규화, 출력 환경 설정, CSV 저장."""
import sys

import pandas as pd

from . import config as C


def setup_output() -> None:
    """stdout utf-8, 출력폴더 생성, matplotlib Agg + 한글 폰트."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(데이터, 채움플래그)를 전체 1분 정규 그리드로 reindex하여 반환.

    가변시차 정렬(alignment)이 정수 스텝 시프트를 쓰므로 그리드 정규성이 전제.
    결락 행은 NaN으로 삽입되고, 플래그는 False로 채운다.
    """
    df = pd.read_parquet(C.IN_PARQUET, columns=C.LOAD_COLS).sort_index()
    flags = pd.read_parquet(C.FLAG_PARQUET).sort_index()
    flags = flags.reindex(columns=C.LOAD_COLS, fill_value=False)

    grid = pd.date_range(df.index[0], df.index[-1], freq="1min")
    df = df.reindex(grid)
    flags = flags.reindex(grid).fillna(False).astype(bool)
    df.index.name = flags.index.name = "datetime"
    return df, flags


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(C.OUTPUT_DIR / name, index=index, encoding="utf-8-sig")
    print(f"저장: {name} ({len(df)}행)")
