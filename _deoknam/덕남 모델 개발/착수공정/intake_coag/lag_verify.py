# -*- coding: utf-8 -*-
"""시차상관·DTW 이벤트 검증 (§4.5-3) — 90~540분 범위.

coag/dtw_lag.py의 banded_dtw_path를 직접 재사용(config 무의존)하고,
segment/xcorr lag는 90~540분·이벤트창 기준으로 파라미터화해 이식.
이벤트: 착수정_TB 60분 변화량 z-score > event_z 시점 → [t-6h, t+18h] 창.
주입률 proxy(주입_유량1) 급변 이벤트는 부호 반전(-dose) xcorr 참고치.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import io_utils as IO


def _znorm(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def _dtw_lag(src: np.ndarray, tgt: np.ndarray) -> float | None:
    """이벤트창 DTW lag(분) — banded_dtw_path 재사용, [lag_lo, lag_hi] 범위."""
    IO.add_coag_path()
    from coag.dtw_lag import banded_dtw_path

    L = C.LAGV
    path = banded_dtw_path(_znorm(src), _znorm(tgt), L["band_steps"])
    if len(path) < 10:
        return None
    trim = max(1, int(len(path) * L["trim_frac"]))
    offs = (path[trim:-trim, 1] - path[trim:-trim, 0]) * L["step_min"]
    lag = float(np.median(offs))
    return lag if L["lag_lo"] <= lag <= L["lag_hi"] else None


def _xcorr_lag(src: np.ndarray, tgt: np.ndarray) -> float | None:
    """교차상관 lag(분): corr(src[t], tgt[t+k]) 최대 k, [lag_lo, lag_hi] 범위."""
    L = C.LAGV
    s, t = _znorm(src), _znorm(tgt)
    lo, hi = L["lag_lo"] // L["step_min"], L["lag_hi"] // L["step_min"]
    best_k, best_c = None, -np.inf
    for k in range(lo, hi + 1):
        if len(s) - k < 20:
            break
        c = float(np.corrcoef(s[: len(s) - k], t[k:])[0, 1])
        if np.isfinite(c) and c > best_c:
            best_c, best_k = c, k
    if best_k is None:
        return None
    return best_k * L["step_min"]


def find_events(df10: pd.DataFrame) -> pd.DatetimeIndex:
    """착수정_TB 60분 변화량 |z| > event_z 이벤트 시점(창 중복 제거)."""
    L = C.LAGV
    tb = df10[C.TURBIDITY_COL]
    d = (tb - tb.shift(60 // L["step_min"])).abs()
    z = (d - d.mean()) / (d.std() or 1.0)
    cand = df10.index[z > L["event_z"]]
    events, last = [], None
    min_gap = pd.Timedelta(hours=L["post_h"])
    for t in cand:
        if last is None or t - last >= min_gap:
            events.append(t)
            last = t
    return pd.DatetimeIndex(events)


def event_lags(df10: pd.DataFrame, events: pd.DatetimeIndex) -> pd.DataFrame:
    """이벤트창별 착수정_TB → 침전지N_탁도 DTW·xcorr lag (+ -dose xcorr 참고)."""
    L = C.LAGV
    rows = []
    for ev in events:
        win = df10.loc[ev - pd.Timedelta(hours=L["pre_h"]):
                       ev + pd.Timedelta(hours=L["post_h"])]
        need = (L["pre_h"] + L["post_h"]) * 60 // L["step_min"]
        if len(win) < int(need * 0.9):
            continue
        src = win[C.TURBIDITY_COL].to_numpy(float)
        if not np.isfinite(src).all() or src.std() < L["min_std_src"]:
            continue
        for b, tcol in enumerate(C.SED_TB, start=1):
            tgt = win[tcol].to_numpy(float)
            if not np.isfinite(tgt).all() or tgt.std() < 1e-3:
                continue
            rows.append({"이벤트": ev, "침전지": b,
                         "DTW_lag(분)": _dtw_lag(src, tgt),
                         "xcorr_lag(분)": _xcorr_lag(src, tgt)})
        # 주입률 proxy 급변 → 침전지1 (부호 반전, 참고)
        dose = win[C.DOSE_FLOW[0].split(".")[-1]] if C.DOSE_FLOW[0].split(".")[-1] in win \
            else win.get(C.DOSE_FLOW[0])
        if dose is not None:
            dv = dose.to_numpy(float)
            if np.isfinite(dv).all() and dv.std() > 1e-3:
                tgt = win[C.SED_TB[0]].to_numpy(float)
                if np.isfinite(tgt).all() and tgt.std() > 1e-3:
                    rows.append({"이벤트": ev, "침전지": 0,   # 0 = dose→sed1 참고
                                 "DTW_lag(분)": None,
                                 "xcorr_lag(분)": _xcorr_lag(-dv, tgt)})
    return pd.DataFrame(rows)
