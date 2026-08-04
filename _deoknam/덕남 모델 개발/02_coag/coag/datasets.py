# -*- coding: utf-8 -*-
"""시간순 분할, 스케일링, 태뷸러/시퀀스 데이터셋 생성."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C

# 모델 입출력 타깃 = 델타(변화량). 절대값 평가는 현재값 + 델타로 복원.
Y_COLS = ["y1_diff", "y2_diff"]
Y_ABS_COLS = ["y1_future", "y2_future"]


def chrono_split(table: pd.DataFrame):
    """시간순 64/16/20 분할."""
    n = len(table)
    i1 = int(n * C.SPLIT["train"])
    i2 = int(n * (C.SPLIT["train"] + C.SPLIT["val"]))
    return table.iloc[:i1], table.iloc[i1:i2], table.iloc[i2:]


def fit_scalers(df_tr: pd.DataFrame, feat_cols: list):
    scaler_x = StandardScaler().fit(df_tr[feat_cols].to_numpy())
    scaler_y = StandardScaler().fit(df_tr[Y_COLS].to_numpy())
    return scaler_x, scaler_y


def to_tabular(df: pd.DataFrame, feat_cols: list, scaler_x, scaler_y):
    """(X_scaled, Y_scaled, index)."""
    X = scaler_x.transform(df[feat_cols].to_numpy())
    Y = scaler_y.transform(df[Y_COLS].to_numpy())
    return X, Y, df.index


def abs_true(df: pd.DataFrame, idx) -> np.ndarray:
    """평가용 절대값 실측 (N,2)."""
    return df.loc[idx, Y_ABS_COLS].to_numpy(float)


def reconstruct_abs(df: pd.DataFrame, idx, diff_pred: np.ndarray) -> np.ndarray:
    """델타 예측(원단위) → 절대값 복원: 현재 평활 탁도(BASE_COLS) + 예측 델타."""
    base = df.loc[idx, C.BASE_COLS].to_numpy(float)
    return base + diff_pred


def to_sequences(df: pd.DataFrame, feat_cols: list, scaler_x, scaler_y,
                 window: int = None):
    """세그먼트 내부에서만 슬라이딩한 시퀀스 (X3d, Y, index).

    index = 예측 기준 시점(윈도우 마지막 행). 세그먼트 경계를 넘는 윈도우는 배제.
    """
    window = C.SEQ["window"] if window is None else window
    Xs, Ys, idxs = [], [], []
    for _, seg in df.groupby("seg_id"):
        if len(seg) < window:
            continue
        X = scaler_x.transform(seg[feat_cols].to_numpy())
        Y = scaler_y.transform(seg[Y_COLS].to_numpy())
        win = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)  # (n-w+1, F, w)
        Xs.append(win.transpose(0, 2, 1).copy())                           # (n-w+1, w, F)
        Ys.append(Y[window - 1:])
        idxs.append(seg.index[window - 1:])
    if not Xs:
        return (np.empty((0, window, len(feat_cols))), np.empty((0, 2)),
                pd.DatetimeIndex([]))
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(Ys).astype(np.float32),
            idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0])
