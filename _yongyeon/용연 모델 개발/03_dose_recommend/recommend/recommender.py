# -*- coding: utf-8 -*-
"""주입률 what-if 스윕 → 목표 탁도 만족 최소 주입률 추천 (N-타깃 일반화)."""
import numpy as np
import pandas as pd

from . import config as R
from . import dose_model
from coag import config as CC


def dose_grid() -> np.ndarray:
    return np.arange(R.DOSE_GRID_MIN, R.DOSE_GRID_MAX + 1e-9, R.DOSE_GRID_STEP)


def sweep_predict(models: dict, df_rows: pd.DataFrame) -> np.ndarray:
    """행×후보 주입률 what-if 예측 (N, G, n_targets). 후보축 누적최소 처리."""
    grid = dose_grid()
    n, g = len(df_rows), len(grid)
    rep = df_rows.loc[df_rows.index.repeat(g)].copy()
    rep[CC.COL_DOSE] = np.tile(grid, n)
    pred = dose_model.predict_abs(models, rep).reshape(n, g, dose_model.N_TARGETS)
    return np.minimum.accumulate(pred, axis=1)  # 주입률↑ → 탁도 비증가 강제


def recommend(models: dict, df_rows: pd.DataFrame,
              dose_lo: float = None, dose_hi: float = None) -> pd.DataFrame:
    """행별 추천: 목표(마진 포함) 만족 최소 주입률 + 지지구간·램프 제한 적용."""
    grid = dose_grid()
    pred = sweep_predict(models, df_rows)                      # (N, G, T)
    ok = np.ones(pred.shape[:2], dtype=bool)
    for j, tgt_ntu in enumerate(R.TARGET_NTU):
        ok &= pred[:, :, j] <= tgt_ntu - R.MARGIN_NTU
    if dose_lo is not None:
        ok &= grid[None, :] >= dose_lo
    if dose_hi is not None:
        ok &= grid[None, :] <= dose_hi
    first = ok.argmax(axis=1)
    feasible = ok.any(axis=1)
    fallback = dose_hi if dose_hi is not None else grid[-1]
    ideal = np.where(feasible, grid[first], fallback)

    cur = df_rows[CC.COL_DOSE].to_numpy(float)
    rec = np.clip(ideal, cur - R.RAMP_PPM, cur + R.RAMP_PPM)
    rec = np.clip(rec, R.DOSE_GRID_MIN, R.DOSE_GRID_MAX)

    gi = np.abs(rec[:, None] - grid[None, :]).argmin(axis=1)
    rows = np.arange(len(df_rows))
    out = pd.DataFrame({
        "현재주입률": cur,
        "이상적주입률": ideal,
        "추천주입률": rec,
        "목표달성가능": feasible,
    }, index=df_rows.index)
    for j, name in enumerate(R.TARGET_NAMES):
        out[f"예측_{name}"] = pred[rows, gi, j]
    return out
