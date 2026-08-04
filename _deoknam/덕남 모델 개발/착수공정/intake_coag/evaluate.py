# -*- coding: utf-8 -*-
"""평가 (§7.3·7.4) — ε 고정 sMAPE·4지표·게이트 판정·구간별·persistence·예측구간.

sMAPE = 100/n × Σ[ 2|y-ŷ| / (|y|+|ŷ|+ε) ],  ε = SMAPE_EPS(0.01 NTU) 전 모델 동일 고정.
§7.3 금지사항을 코드로 강제: ε은 인자로 받지 않으며(고정 상수), 저탁도 행 제외 없음.
시험구간 평가는 TestGuard로 (과제, 대상, 모델)당 1회 강제(§7.4 재튜닝 금지).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C


def smape(y_true, y_pred) -> float:
    """ε 고정 sMAPE(%). ε 조정·행 선별 금지(§7.3) — 인자 없음."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(y - p)
                                 / (np.abs(y) + np.abs(p) + C.SMAPE_EPS)))


def metrics(y_true, y_pred) -> dict:
    """RMSE/MAE/sMAPE/R² (§7.3 필수 4지표). 상수 타깃이면 R²=NaN."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return dict(RMSE=np.nan, MAE=np.nan, sMAPE=np.nan, R2=np.nan, n=len(y))
    return dict(
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        MAE=float(mean_absolute_error(y, p)),
        sMAPE=smape(y, p),
        R2=float(r2_score(y, p)) if np.var(y) > 1e-9 else np.nan,
        n=int(len(y)),
    )


def gate_pass(smape_val: float) -> bool:
    """§7.4 승인 게이트: 시험 sMAPE ≤ 5.0%."""
    return bool(np.isfinite(smape_val) and smape_val <= C.GATE_SMAPE)


def persistence_smoothed(aux: pd.DataFrame, basin: int) -> np.ndarray:
    """persistence 기준선: 현재 1h 트레일링 평활 침전지 탁도 유지(sed{b}_sm)."""
    return aux[f"sed{basin}_sm"].to_numpy(float)


def interval_stats(y_val, p_val, y_te, p_te) -> dict:
    """예측구간(§7.3): 검증 잔차 σ 기반 ±1.96σ — 시험 포함률·평균 폭."""
    r = np.asarray(y_val, float) - np.asarray(p_val, float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return dict(포함률95=np.nan, 구간폭=np.nan)
    half = 1.96 * float(np.std(r))
    y, p = np.asarray(y_te, float), np.asarray(p_te, float)
    m = np.isfinite(y) & np.isfinite(p)
    cover = float(np.mean(np.abs(y[m] - p[m]) <= half)) if m.any() else np.nan
    return dict(포함률95=round(cover, 3), 구간폭=round(2 * half, 4))


def segment_metrics(aux_te: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """구간별(§7.3): 계절·착수탁도·급상승·유량 구간 RMSE/MAE/sMAPE/R²."""
    df = aux_te.copy()
    df["_y"], df["_p"] = np.asarray(y_true, float), np.asarray(y_pred, float)
    rows = []

    def add(kind, key, g):
        if len(g) >= 10:
            rows.append({"구간종류": kind, "구간": str(key), **metrics(g["_y"], g["_p"])})

    season = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12],
                    labels=["겨울(1-3)", "봄(4-6)", "여름(7-9)", "가을(10-12)"])
    for k, g in df.groupby(season, observed=True):
        add("계절", k, g)
    tb_tag = C.TURBIDITY_COL.split(".")[-1]
    if tb_tag in df.columns:
        tb = pd.cut(df[tb_tag], bins=C.TB_BINS, labels=C.TB_BIN_LABELS)
        for k, g in df.groupby(tb, observed=True):
            add("착수탁도", k, g)
    if "탁도급상승" in df.columns:
        for k, g in df.groupby(df["탁도급상승"] > 0.5):
            add("급상승", "급상승" if k else "평상", g)
    fl_tag = C.FLOW_COL.split(".")[-1]
    if fl_tag in df.columns:
        fb = pd.qcut(df[fl_tag], q=[0, .33, .66, 1],
                     labels=["저유량", "중유량", "고유량"], duplicates="drop")
        for k, g in df.groupby(fb, observed=True):
            add("유량", k, g)
    if "SV_since_change" in df.columns:
        near = df["SV_since_change"] <= 60
        for k, g in df.groupby(near):
            add("주입변경", "변경후 1h 이내" if k else "그 외", g)
    return pd.DataFrame(rows)


class TestGuard:
    """시험구간 평가 1회 강제(§7.4) — 동일 키 재평가 시 RuntimeError."""

    def __init__(self):
        self._used: set = set()

    def check(self, key: tuple) -> None:
        if key in self._used:
            raise RuntimeError(f"시험구간 재평가 금지(§7.4): {key}")
        self._used.add(key)


def _selftest():
    """ε-sMAPE 수기 검산: y=[0.2,0.4], ŷ=[0.22,0.38] →
    2×0.02/(0.62+ε)=0.063898.., 2×0.02/(0.78+ε)=0.050633.. 평균×100=5.7266%"""
    v = smape([0.2, 0.4], [0.22, 0.38])
    e1 = 2 * 0.02 / (0.2 + 0.22 + C.SMAPE_EPS)
    e2 = 2 * 0.02 / (0.4 + 0.38 + C.SMAPE_EPS)
    expect = 100 * (e1 + e2) / 2
    assert abs(v - expect) < 1e-9, f"sMAPE 단위테스트 실패: {v} vs {expect}"


_selftest()
