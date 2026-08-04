# -*- coding: utf-8 -*-
"""주입률 what-if 스윕 → 목표 탁도 만족 최소 주입률 추천."""
import numpy as np
import pandas as pd

from . import config as R
from . import dose_model
from coag import config as CC


def dose_grid() -> np.ndarray:
    return np.arange(R.DOSE_GRID_MIN, R.DOSE_GRID_MAX + 1e-9, R.DOSE_GRID_STEP)


def sweep_predict(models: dict, df_rows: pd.DataFrame) -> np.ndarray:
    """행×후보 주입률 what-if 예측.

    반환 (N, G, 2): 후보 g의 주입률로 치환했을 때 침전지1/2 예측 절대 탁도.
    단조 제약이 있어도 안전을 위해 후보축으로 누적최소(단조 비증가) 처리.
    """
    grid = dose_grid()
    n, g = len(df_rows), len(grid)
    rep = df_rows.loc[df_rows.index.repeat(g)].copy()
    rep[CC.COL_DOSE] = np.tile(grid, n)
    pred = dose_model.predict_abs(models, rep).reshape(n, g, 2)
    return np.minimum.accumulate(pred, axis=1)  # 주입률↑ → 탁도 비증가 강제


def recommend(models: dict, df_rows: pd.DataFrame,
              dose_lo: float = None, dose_hi: float = None) -> pd.DataFrame:
    """행별 추천: 목표(마진 포함) 만족 최소 주입률 + 지지구간·램프 제한 적용.

    dose_lo/hi: 학습 데이터 지지 구간 — 이 밖의 후보는 외삽이라 탐색에서 제외.
    반환 컬럼: 이상적주입률, 추천주입률(램프 적용), 목표달성가능, 예측탁도1/2,
    현재주입률.
    """
    grid = dose_grid()
    pred = sweep_predict(models, df_rows)                      # (N, G, 2)
    ok = ((pred[:, :, 0] <= R.TARGET_NTU[1] - R.MARGIN_NTU)
          & (pred[:, :, 1] <= R.TARGET_NTU[2] - R.MARGIN_NTU))  # (N, G)
    if dose_lo is not None:
        ok &= grid[None, :] >= dose_lo
    if dose_hi is not None:
        ok &= grid[None, :] <= dose_hi
    first = ok.argmax(axis=1)                                   # 최초 만족 후보
    feasible = ok.any(axis=1)
    fallback = dose_hi if dose_hi is not None else grid[-1]     # 미달 시 지지구간 상한
    ideal = np.where(feasible, grid[first], fallback)

    cur = df_rows[CC.COL_DOSE].to_numpy(float)
    rec = np.clip(ideal, cur - R.RAMP_PPM, cur + R.RAMP_PPM)
    rec = np.clip(rec, R.DOSE_GRID_MIN, R.DOSE_GRID_MAX)

    # 추천 주입률에서의 예측 탁도 (그리드 최근접 후보)
    gi = np.abs(rec[:, None] - grid[None, :]).argmin(axis=1)
    rows = np.arange(len(df_rows))
    return pd.DataFrame({
        "현재주입률": cur,
        "이상적주입률": ideal,
        "추천주입률": rec,
        "목표달성가능": feasible,
        "예측_침전지1": pred[rows, gi, 0],
        "예측_침전지2": pred[rows, gi, 1],
    }, index=df_rows.index)
