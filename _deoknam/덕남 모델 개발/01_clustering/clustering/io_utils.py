# -*- coding: utf-8 -*-
"""입출력 유틸: 출력 환경 설정, 전처리 데이터 로딩(검증 포함), 결과 저장."""
import sys

import pandas as pd

from . import config as C


def setup_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.CLUSTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_preprocessed() -> pd.DataFrame:
    """전처리 완료 parquet 로드. 1분 연속성이 깨지면 명시적으로 실패."""
    df = pd.read_parquet(C.IN_PARQUET).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("전처리 parquet 인덱스가 DatetimeIndex가 아님")
    gaps = df.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta("1min")).all():
        n_bad = int((gaps != pd.Timedelta("1min")).sum())
        raise ValueError(f"1분 연속성 위반 구간 {n_bad}건 — 위치 롤링 == 시간 롤링 가정이 깨짐")
    missing = [c for c in C.RAW_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"원수성상 컬럼 누락: {missing}")
    return df


def save_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_labeled(df_labeled: pd.DataFrame) -> None:
    """라벨 통합 parquet + 라벨별 분할 parquet 저장."""
    df_labeled.to_parquet(C.OUT_LABELED)
    print(f"저장: {C.OUT_LABELED} — {len(df_labeled):,}행 × {df_labeled.shape[1]}컬럼")
    for cid in range(C.K):
        part = df_labeled[df_labeled[C.LABEL_COL] == cid]
        path = C.CLUSTER_DATA_DIR / f"덕남_cluster{cid}.parquet"
        part.to_parquet(path)
        print(f"저장: {path.name} — {len(part):,}행 ({len(part) / len(df_labeled) * 100:.1f}%)")
