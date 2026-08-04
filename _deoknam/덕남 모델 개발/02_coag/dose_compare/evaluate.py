# -*- coding: utf-8 -*-
"""평가지표: MAE/RMSE/R²/MAPE + persistence 베이스라인 + 구간별 오차."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 2:
        return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE=np.nan, n=len(y_true))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    # 타깃 분산이 ~0이면 R²는 무의미 → N/A (상수 타깃 test 구간 오해 방지)
    r2 = float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-9 else np.nan
    return dict(
        MAE=float(mean_absolute_error(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2=r2,
        MAPE=float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100),
        n=int(len(y_true)),
    )


def persistence_pred(sv_prev):
    """직전 SV 유지 = persistence 예측."""
    return np.asarray(sv_prev, float)


def segment_errors(aux_te: pd.DataFrame, y_pred, target_col="__target__"):
    """구간별(계절·착수탁도·유량) MAE. aux_te = test 원본행(피처 스케일 전)."""
    df = aux_te.copy()
    df["_pred"] = y_pred
    df["_ae"] = (df[target_col] - df["_pred"]).abs()
    rows = []
    # 계절(월)
    season = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12],
                    labels=["겨울(1-3)", "봄(4-6)", "여름(7-9)", "가을(10-12)"])
    for k, g in df.groupby(season, observed=True):
        rows.append(("계절", str(k), len(g), float(g["_ae"].mean())))
    # 착수탁도 구간
    tb_tag = C.TURBIDITY_COL.split(".")[-1]
    if tb_tag in df.columns:
        tb = pd.cut(df[tb_tag], bins=[-1, 1, 2, 5, 1e9],
                    labels=["≤1", "1-2", "2-5", ">5"])
        for k, g in df.groupby(tb, observed=True):
            rows.append(("착수탁도", str(k), len(g), float(g["_ae"].mean())))
    # 유량 구간
    fl_tag = C.FLOW_COL.split(".")[-1]
    if fl_tag in df.columns:
        q = df[fl_tag]
        fb = pd.qcut(q, q=[0, .33, .66, 1], labels=["저유량", "중유량", "고유량"], duplicates="drop")
        for k, g in df.groupby(fb, observed=True):
            rows.append(("유량", str(k), len(g), float(g["_ae"].mean())))
    return pd.DataFrame(rows, columns=["구간종류", "구간", "n", "MAE"])


def change_point_error(aux_te: pd.DataFrame, y_pred, target_col="__target__", win=5):
    """실제 SV 변경 직후 win분 이내 MAE (시계열 안정성)."""
    df = aux_te.copy()
    df["_pred"] = y_pred
    changed = df[target_col].diff().abs() > 1e-9
    near = changed.rolling(win, min_periods=1).max().fillna(0).astype(bool)
    ae = (df[target_col] - df["_pred"]).abs()
    return float(ae[near].mean()) if near.any() else np.nan
