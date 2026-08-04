# -*- coding: utf-8 -*-
"""10분 리샘플, 세그먼트 분리, 운휴 마스킹, DTW-lag 타깃 정렬, 학습 테이블."""
import numpy as np
import pandas as pd

from . import config as C


def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """1분 → 10분 평균. 군집은 시간상 불연속이므로 전량-NaN 행은 제거."""
    out = df.drop(columns=["cluster_label"], errors="ignore")
    out = out.resample(C.RESAMPLE_RULE).mean()
    return out.dropna(how="all")


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """리샘플 간격(10분) 초과 공백마다 세그먼트 증가 — DTW·시퀀스가 경계를 넘지 않게."""
    gap = df.index.to_series().diff() > pd.Timedelta(minutes=C.GAP_MIN)
    df = df.copy()
    df["seg_id"] = gap.cumsum()
    return df


def mask_idle_basin2(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """침전지2 장기 무변동(운휴 추정) 구간의 타깃만 NaN 처리."""
    col = C.TARGET_COLS[1]
    std = df[col].rolling(C.IDLE["roll"]).std()
    idle = (std < C.IDLE["std_eps"]).fillna(False)
    df = df.copy()
    df.loc[idle, col + "_y_masked"] = True
    return df, int(idle.sum())


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """AR 추세 피처: 세그먼트 내에서 현재값 − 30/60분 전 값."""
    df = df.copy()
    g = df.groupby("seg_id")
    for col in C.TREND_SRC:
        for w in (3, 6):
            df[f"{col}__d{w}"] = df[col] - g[col].shift(w)
    return df


def add_smoothed_targets(df: pd.DataFrame) -> pd.DataFrame:
    """타깃 평활: 세그먼트 내 1h 트레일링 이동평균 (미래 정보 없음)."""
    df = df.copy()
    g = df.groupby("seg_id")
    for tgt in C.TARGET_COLS:
        df[f"{tgt}__sm"] = g[tgt].transform(
            lambda s: s.rolling(C.TARGET_SMOOTH_STEPS, min_periods=3).mean())
    return df


def align_targets(df: pd.DataFrame, lags: dict[str, int],
                  dynamic: bool = False) -> pd.DataFrame:
    """타깃별 t+lag 시점의 평활 탁도를 datetime 병합으로 결합 (gap-safe).

    lags: {타깃컬럼: lag(분, 10분 배수)}. y 컬럼명 = "y1_future", "y2_future".
    운휴 마스킹(_y_masked)이 표시된 미래 시점 값은 NaN 처리.

    dynamic=True면 행별 lag(t) = lag_c × Q_ref/Q(t) (10분 배수 반올림,
    [lag_min, lag_max] 클립, Q_ref = 유량 중앙값)로 동적 정렬.
    """
    out = df.copy()
    q = df[C.DYN_LAG_Q_COL] if dynamic else None
    q_ref = float(q.median()) if dynamic else None
    for i, tgt in enumerate(C.TARGET_COLS, start=1):
        sm = f"{tgt}__sm"
        fut = df[sm].copy()
        if f"{tgt}_y_masked" in df.columns:
            fut[df[f"{tgt}_y_masked"].fillna(False).to_numpy()] = np.nan
        ycol = f"y{i}_future"
        if not dynamic:
            shifted = fut.copy()
            shifted.index = shifted.index - pd.Timedelta(minutes=lags[tgt])
            out = out.join(shifted.rename(ycol), how="left")
        else:
            lag_row = ((lags[tgt] * q_ref / q)
                       .clip(C.DTW["lag_min"], C.DTW["lag_max"])
                       .div(10).round().mul(10).astype(int))
            y = pd.Series(np.nan, index=df.index)
            for lag_v, idx in df.index.to_series().groupby(lag_row).groups.items():
                y.loc[idx] = fut.reindex(idx + pd.Timedelta(minutes=int(lag_v))).to_numpy()
            out[ycol] = y
    return out


def build_table(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """후보 피처 + 타깃 + seg_id 학습 테이블.

    타깃은 델타 방식: yN_diff = 평활침전지N(t+lag) − 평활침전지N(t).
    계절 간 수준(level) 이동에 강건하며, 절대값 복원은 현재 평활값 + 예측 델타.
    yN_future(평활 절대값)와 기준값(BASE_COLS)은 평가용으로 함께 유지.
    """
    out = df_aligned.copy()
    out["y1_diff"] = out["y1_future"] - out[C.BASE_COLS[0]]
    out["y2_diff"] = out["y2_future"] - out[C.BASE_COLS[1]]
    cols = (C.CAND_FEATURES + C.BASE_COLS
            + ["y1_future", "y2_future", "y1_diff", "y2_diff", "seg_id"])
    table = out[cols].dropna(subset=C.CAND_FEATURES + C.BASE_COLS
                             + ["y1_future", "y2_future"])
    return table
