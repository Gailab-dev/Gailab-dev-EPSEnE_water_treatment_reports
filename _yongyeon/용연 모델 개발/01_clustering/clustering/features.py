# -*- coding: utf-8 -*-
"""슬라이딩 10분 윈도우 특징 계산 (벡터화).

시점 t의 특징 = 직전 10분(t-9 ~ t, 10샘플)의 컬럼별 {평균, 기울기}.
기울기는 x = 0..9(분)에 대한 OLS 회귀계수로, 단위는 '분당 변화율'.
x를 중심화하면 slope = Σ(x-4.5)·y / Σ(x-4.5)² 이므로 ȳ 항이 소거되어
einsum 한 번으로 전 시점 계산이 가능하다.

앞 window-1행(불완전 윈도우)과 윈도우에 NaN이 포함된 시점은 특징 NaN
→ 군집 라벨 NA가 된다.
"""
import numpy as np
import pandas as pd

from . import config as C


def feature_columns(cols=None) -> list:
    """[col__mean, col__slope] × 5 — 학습/예측/온라인 공통 고정 순서."""
    cols = C.RAW_FEATURES if cols is None else cols
    out = []
    for c in cols:
        out += [f"{c}__mean", f"{c}__slope"]
    return out


def make_window_features(df: pd.DataFrame, cols=None, window: int = C.WINDOW) -> pd.DataFrame:
    """전 시점 트레일링 윈도우 특징 (N × 2·len(cols)). 무복사 stride 뷰 사용."""
    cols = C.RAW_FEATURES if cols is None else cols
    vals = df[cols].to_numpy(dtype=np.float64)                              # (N, F)
    win = np.lib.stride_tricks.sliding_window_view(vals, window, axis=0)    # (N-w+1, F, w)
    x = np.arange(window, dtype=np.float64)
    xc = x - x.mean()
    means = win.mean(axis=2)                                                # NaN 포함 창 → NaN
    slopes = np.einsum("nfw,w->nf", win, xc) / (xc @ xc)
    out = np.full((len(df), 2 * len(cols)), np.nan)
    out[window - 1:, 0::2] = means
    out[window - 1:, 1::2] = slopes
    return pd.DataFrame(out, index=df.index, columns=feature_columns(cols))


def window_features_single(window_df: pd.DataFrame, cols=None,
                           window: int = C.WINDOW) -> np.ndarray:
    """온라인 경로: 최근 10행(오름차순) → (1, 2F) 특징. 배치와 동일 수식."""
    cols = C.RAW_FEATURES if cols is None else cols
    if len(window_df) != window:
        raise ValueError(f"윈도우 길이 {len(window_df)} != {window}")
    vals = window_df[cols].to_numpy(dtype=np.float64)                       # (w, F)
    x = np.arange(window, dtype=np.float64)
    xc = x - x.mean()
    means = vals.mean(axis=0)
    slopes = xc @ vals / (xc @ xc)
    out = np.empty(2 * len(cols))
    out[0::2] = means
    out[1::2] = slopes
    return out.reshape(1, -1)


def naive_window_features(df: pd.DataFrame, cols=None, window: int = C.WINDOW) -> pd.DataFrame:
    """검증 전용: 시점별 루프 + np.polyfit. make_window_features 대조용."""
    cols = C.RAW_FEATURES if cols is None else cols
    x = np.arange(window, dtype=np.float64)
    out = np.full((len(df), 2 * len(cols)), np.nan)
    vals = df[cols].to_numpy(dtype=np.float64)
    for t in range(window - 1, len(df)):
        w = vals[t - window + 1:t + 1]
        for j in range(len(cols)):
            y = w[:, j]
            out[t, 2 * j] = y.mean()
            if np.isfinite(y).all():
                out[t, 2 * j + 1] = np.polyfit(x, y, 1)[0]
    return pd.DataFrame(out, index=df.index, columns=feature_columns(cols))
