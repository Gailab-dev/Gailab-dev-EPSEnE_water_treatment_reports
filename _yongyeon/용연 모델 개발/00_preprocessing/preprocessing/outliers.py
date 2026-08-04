# -*- coding: utf-8 -*-
"""이상치 탐지: Hampel, 롤링 IQR, IsolationForest(구간 제한) 및 결합 로직.

최종 마스크 = (Hampel ∪ IQR) ∪ (ISO ∩ dilate(Hampel ∪ IQR)).
ISO는 고정 contamination으로 정상 구간까지 일정 비율을 지목하므로,
Hampel/IQR이 탐지한 구간의 확장 윈도우 안에서 잡은 지점만 채택한다.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C


def hampel_mask(s: pd.Series, window: int, n_sigma: float, floor: float) -> pd.Series:
    """롤링 중앙값 대비 |x-med| > max(n_sigma*1.4826*MAD, floor) 인 지점을 이상치로."""
    minp = max(3, window // 2)
    med = s.rolling(window, center=True, min_periods=minp).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=minp).median()
    thresh = (n_sigma * 1.4826 * mad).clip(lower=floor)
    return ((s - med).abs() > thresh).fillna(False)


def iqr_mask(s: pd.Series, k: float = C.IQR_K, floor: float = 0.0,
             window: str = C.IQR_WINDOW) -> pd.Series:
    """롤링 IQR 경계 밖 이상치. 경계폭 k*IQR은 floor 하한으로 상수 구간 과탐 억제."""
    q1 = s.rolling(window, center=True, min_periods=C.IQR_MIN_PERIODS).quantile(0.25)
    q3 = s.rolling(window, center=True, min_periods=C.IQR_MIN_PERIODS).quantile(0.75)
    band = (k * (q3 - q1)).clip(lower=floor)
    return ((s < q1 - band) | (s > q3 + band)).fillna(False)


def isoforest_mask(s: pd.Series) -> pd.Series:
    """IsolationForest: 원값, 11pt 롤링중앙값 잔차, 1차 차분을 특징으로 사용."""
    feat = pd.DataFrame({
        "x": s,
        "resid": s - s.rolling(11, center=True, min_periods=1).median(),
        "diff": s.diff(),
    }).fillna(0.0)
    iso = IsolationForest(**C.ISO_PARAMS)
    return pd.Series(iso.fit_predict(feat.values) == -1, index=s.index)


def dilate_mask(mask: pd.Series, window: int = C.ISO_DILATE) -> pd.Series:
    """불리언 마스크를 ±window//2 샘플 확장 (rolling max)."""
    return mask.astype(float).rolling(window, center=True, min_periods=1).max().astype(bool)


def combined_mask(s: pd.Series, hampel_params: tuple, iqr_floor: float):
    """Hampel∪IQR 전체 + 그 구간 확장 윈도우 내 ISO 탐지 지점 추가.

    반환: (최종 마스크, 방법별 마스크 dict — 리포트/시각화용)
    """
    w, ns, floor = hampel_params
    h = hampel_mask(s, w, ns, floor)
    q = iqr_mask(s, floor=iqr_floor)
    base = h | q
    iso_raw = isoforest_mask(s)
    iso_used = iso_raw & dilate_mask(base)
    final = base | iso_used
    return final, {"hampel": h, "iqr": q, "iso_raw": iso_raw, "iso_used": iso_used, "final": final}
