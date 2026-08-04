# -*- coding: utf-8 -*-
"""시간정렬 (§5) — X(t) → y(t+T_outlet) 타깃 84종 사전생성 + 급변행 플래그.

마스터는 완전 1분 정규격자(io_utils.load_master)이므로 positional == temporal.
y 명명: y{1|2}__{c121|c334|c359|c481|dynmean|dynpeak}__{pt|med30|med45|med60|max30|max45|max60}
  - pt   : 단일점  TB[t+T]                        (§5.2 단일점 목표)
  - medΔ : 시간창  median(TB[t+T-Δ : t+T+Δ])      (시간창 목표)
  - maxΔ : 위험    max(TB[t+T-Δ : t+T+Δ])         (위험 목표)
운휴(침전지2 장기 무변동) 시점의 미래값은 NaN 처리(coag.mask_idle_basin2 패턴).
급변행: 체류창 [t, t+T] 내 유량 상대변동 > RAPID_Q_REL → 학습 제외 플래그(§5.3).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import residence as R


def positional_shift(values: np.ndarray, lag) -> np.ndarray:
    """y[i] = values[i + lag_i] (lag: int 또는 행별 int/NaN 배열). 범위 밖·NaN lag → NaN."""
    n = len(values)
    out = np.full(n, np.nan)
    if np.isscalar(lag):
        lag = int(lag)
        if lag < n:
            out[: n - lag] = values[lag:]
        return out
    lag = np.asarray(lag, float)
    ok = np.isfinite(lag)
    idx = np.arange(n) + np.where(ok, lag, 0).astype(int)
    ok &= idx < n
    out[ok] = values[idx[ok]]
    return out


def idle_mask_basin2(df: pd.DataFrame) -> pd.Series:
    """침전지2 장기 무변동(운휴 추정) bool 마스크."""
    std = df[C.SED_TB[1]].rolling(C.IDLE["roll"]).std()
    return (std < C.IDLE["std_eps"]).fillna(False)


def _fwd_window_stat(s: pd.Series, win: int, stat: str, min_frac: float) -> np.ndarray:
    """앞방향 창 [t, t+win-1] 통계 — 역방향 rolling 트릭 (정규격자 전제)."""
    mp = max(2, int(win * min_frac))
    rev = s.iloc[::-1].rolling(win, min_periods=mp)
    r = getattr(rev, stat)()
    return r.iloc[::-1].to_numpy()


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """마스터(1분 정규격자) → (y 84종 + 급변플래그 6종 DataFrame, 통계 dict)."""
    idx = df.index
    n = len(df)
    q = df[C.FLOW_COL]

    # 운휴 마스킹: 침전지2 미래값이 운휴면 y로 쓰지 않음
    idle2 = idle_mask_basin2(df)
    tb = {1: df[C.SED_TB[0]].to_numpy(float),
          2: df[C.SED_TB[1]].where(~idle2).to_numpy(float)}

    # 중심별 lag: 정적 상수 + 동적 행별
    lag_by_center: dict[str, object] = dict(C.CENTERS_STATIC)
    for kind in C.CENTERS_DYNAMIC:
        lag_by_center[kind] = R.dynamic_lag_minutes(q, kind).to_numpy()

    # 시간창 통계(중심 t 기준 ±Δ) — 중앙정렬 rolling 1회/(침전지,Δ,stat)
    win_stats = {}
    for b in (1, 2):
        s = pd.Series(tb[b], index=idx)
        for d in C.DELTAS:
            w = 2 * d + 1
            mp = max(2, int(w * C.WIN_MIN_FRAC))
            win_stats[(b, d, "med")] = s.rolling(w, center=True, min_periods=mp).median().to_numpy()
            win_stats[(b, d, "max")] = s.rolling(w, center=True, min_periods=mp).max().to_numpy()

    out = pd.DataFrame(index=idx)
    for cname, lag in lag_by_center.items():
        for b in (1, 2):
            out[f"y{b}__{cname}__pt"] = positional_shift(tb[b], lag)
            for d in C.DELTAS:
                out[f"y{b}__{cname}__med{d}"] = positional_shift(win_stats[(b, d, "med")], lag)
                out[f"y{b}__{cname}__max{d}"] = positional_shift(win_stats[(b, d, "max")], lag)

    # 급변행 플래그: 체류창 [t, t+T] 내 유량 상대변동 (§5.3)
    qs = pd.to_numeric(q, errors="coerce")
    static_items = list(C.CENTERS_STATIC.items())
    for cname, T in static_items:
        w = int(T) + 1
        qmax = _fwd_window_stat(qs, w, "max", 0.25)
        qmin = _fwd_window_stat(qs, w, "min", 0.25)
        qmean = _fwd_window_stat(qs, w, "mean", 0.25)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = (qmax - qmin) / qmean
        out[f"rapid__{cname}"] = rel > C.RAPID_Q_REL
    # 동적 중심: 행별 lag에 가장 가까운 정적 중심의 플래그 재사용
    svals = np.array([v for _, v in static_items])
    for kind in C.CENTERS_DYNAMIC:
        lag = np.asarray(lag_by_center[kind], float)
        nearest = np.abs(lag[:, None] - svals[None, :]).argmin(axis=1)
        flags = np.column_stack([out[f"rapid__{nm}"].to_numpy() for nm, _ in static_items])
        out[f"rapid__{kind}"] = flags[np.arange(n), nearest]
        out.loc[~np.isfinite(lag), f"rapid__{kind}"] = True   # lag 미산출 행은 제외

    stats = {
        "idle2_rows": int(idle2.sum()),
        "coverage": {c: round(float(out[c].notna().mean()), 4)
                     for c in out.columns if c.startswith("y")},
        "rapid_frac": {c: round(float(out[c].mean()), 4)
                       for c in out.columns if c.startswith("rapid")},
    }
    return out, stats


def verify_alignment(df: pd.DataFrame, targets: pd.DataFrame,
                     n_sample: int = 20, seed: int = 42) -> None:
    """누설 방지 검증: y{b}__c{T}__pt[t] == 침전지 탁도[t+T] 표본 대조 assert."""
    rng = np.random.default_rng(seed)
    idx = df.index
    idle2 = idle_mask_basin2(df)
    for cname, T in C.CENTERS_STATIC.items():
        col = f"y1__{cname}__pt"
        pos = rng.choice(len(df) - T - 1, size=n_sample, replace=False)
        for p in pos:
            expect = df[C.SED_TB[0]].iloc[p + T]
            got = targets[col].iloc[p]
            if np.isnan(expect) and np.isnan(got):
                continue
            assert np.isclose(expect, got, equal_nan=True), \
                f"{col}@{idx[p]}: 기대 {expect} vs 실제 {got}"
    # med 창 수동 검산 1건 (침전지1, c334, Δ45)
    T, d = C.CENTERS_STATIC["c334"], 45
    p = int(len(df) * 0.5)
    seg = df[C.SED_TB[0]].iloc[p + T - d: p + T + d + 1]
    if seg.notna().sum() >= max(2, int((2 * d + 1) * C.WIN_MIN_FRAC)):
        expect = float(seg.median())
        got = float(targets[f"y1__c334__med{d}"].iloc[p])
        assert np.isclose(expect, got), f"med 검산 실패: {expect} vs {got}"
    print(f"정렬 누설검증 통과: 정적 4중심 × {n_sample}표본 + med 검산")
