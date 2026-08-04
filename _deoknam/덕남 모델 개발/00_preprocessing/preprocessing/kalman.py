# -*- coding: utf-8 -*-
"""1D 랜덤워크 칼만 필터: 평활(kalman_smooth) 및 결측 채움(kalman_fill).

kalman_smooth는 참고 프로젝트(덕남_응집제_투입량/src/smoothing_compare.py)와 동일.
NaN 구간에서는 관측 갱신 없이 예측만 수행해 상태를 전파하므로,
이상치 제거로 생긴 공백이 직전 상태 기반 추정값으로 채워진다.
"""
import numpy as np
import pandas as pd

from . import config as C


def nan_run_len(mask: np.ndarray) -> np.ndarray:
    """불리언 NaN 마스크 → 각 위치가 속한 연속 NaN 런의 길이(비NaN=0)."""
    n = len(mask)
    out = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def kalman_smooth(s: pd.Series, r_scale: float = 5.0,
                  q_scale: float = C.KALMAN_Q_SCALE) -> pd.Series:
    """1D 랜덤워크 칼만 보정. r_scale↑ → 더 매끄럽게(관측 불신)."""
    x = s.to_numpy(dtype=float)
    n = len(x)
    finite = x[np.isfinite(x)]
    r = np.var(np.diff(finite)) * r_scale if len(finite) > 1 else 1.0
    if not np.isfinite(r) or r <= 0:
        r = 1.0  # 상수 신호(var=0)에서 0/0 방지
    q = r * q_scale
    xhat = np.empty(n)
    xh = finite[0] if len(finite) else 0.0
    P = 1.0
    for i in range(n):
        Pm = P + q
        xi = x[i]
        if np.isfinite(xi):
            K = Pm / (Pm + r)
            xh = xh + K * (xi - xh)
            P = (1 - K) * Pm
        else:
            P = Pm
        xhat[i] = xh
    return pd.Series(xhat, index=s.index)


def kalman_fill(s: pd.Series, r_scale: float = C.KALMAN_R_DEFAULT,
                q_scale: float = C.KALMAN_Q_SCALE,
                max_gap: int | None = None) -> pd.Series:
    """유효구간 내부의 NaN만 칼만 추정값으로 채움. 관측값은 그대로 유지.

    유효구간 밖(선두/후미 결측)은 채우지 않는다 — 칼만 초기상태가
    첫 관측값이라 선두 결측이 상수로 채워지는 것을 방지.
    max_gap(분/샘플)이 주어지면 그보다 긴 내부 결측 런은 채우지 않고 NaN 유지
    — 장기 결측을 상수로 지어내는 것 방지.
    """
    if s.notna().sum() == 0:
        return s.copy()
    est = kalman_smooth(s, r_scale=r_scale, q_scale=q_scale)
    out = s.where(s.notna(), est)
    first, last = s.first_valid_index(), s.last_valid_index()
    out[(out.index < first) | (out.index > last)] = np.nan
    if max_gap is not None:
        too_long = nan_run_len(s.isna().to_numpy()) > max_gap
        out[too_long] = np.nan
    return out
