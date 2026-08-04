# -*- coding: utf-8 -*-
"""가변시차 시계열 정렬 — 전체 1분 정규 그리드 전제(io_utils.load_grid 보장).

상류(§6.4): 현재 플록형성지에 도달한 물의 원수수질 = 과거 t-lag(t) 값 조회.
하류(§12): RPM 시점 조건 ↔ 체류시간 후 침전수 탁도(중앙값 창) 연결.
그리드가 정규이므로 merge_asof 대신 정수 스텝 팬시 인덱싱으로 O(n) 처리.
"""
import numpy as np
import pandas as pd


def shift_by_lag(cols: pd.DataFrame, lag_min: pd.Series) -> pd.DataFrame:
    """각 행 t에 대해 cols[t - lag(t)] 를 반환 (lag는 분, 1분 단위 반올림).

    시계열 앞머리(과거 데이터 없음)는 NaN.
    """
    n = len(cols)
    steps = lag_min.round().astype(int).to_numpy()
    src = np.arange(n) - steps
    valid = src >= 0
    out = cols.to_numpy(dtype=float)[np.clip(src, 0, n - 1)]
    out[~valid] = np.nan
    return pd.DataFrame(out, index=cols.index, columns=cols.columns)


def rolling_median_shift(s: pd.Series, lag_min: int, half_win_min: int) -> pd.Series:
    """TB_result(t) = median(s[t+lag-Δ : t+lag+Δ]) — 미래 창 중앙값을 t로 당김 (§12)."""
    med = s.rolling(window=2 * half_win_min + 1, center=True,
                    min_periods=half_win_min).median()
    return med.shift(-lag_min)
