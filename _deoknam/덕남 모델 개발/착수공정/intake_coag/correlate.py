# -*- coding: utf-8 -*-
"""관계분석 (§6) — Pearson/Spearman/Kendall/log1p/sqrt/제곱 + VIF + 수질구간 표본.

상관은 학습구간만 사용(누설 방지). VIF는 상관행렬 역행렬 대각(diag(inv(R)))로 산출.
0·음수값에는 로그 직접 적용 금지 → log1p는 비음수 컬럼에만(§6).
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def corr_table(df: pd.DataFrame, feat_cols: list[str], ycol: str) -> pd.DataFrame:
    """피처×타깃 상관 — 방법별(§6 표). 표본 결측쌍 제거."""
    y = df[ycol].astype(float)
    rows = []
    for f in feat_cols:
        x = df[f].astype(float)
        m = x.notna() & y.notna()
        if m.sum() < 100:
            continue
        xv, yv = x[m], y[m]
        if xv.std() < 1e-12:
            continue
        row = {"피처": f, "n": int(m.sum()),
               "Pearson": float(xv.corr(yv)),
               "Spearman": float(xv.corr(yv, method="spearman")),
               "Kendall": float(stats.kendalltau(xv, yv, nan_policy="omit")[0])}
        if (xv >= 0).all():
            row["log1p"] = float(np.log1p(xv).corr(yv))
            row["sqrt"] = float(np.sqrt(xv).corr(yv))
        else:
            row["log1p"] = np.nan
            row["sqrt"] = np.nan
        row["제곱"] = float((xv ** 2).corr(yv))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("Spearman", key=lambda s: s.abs(), ascending=False)


def vif_reduce(df: pd.DataFrame, feat_cols: list[str],
               vif_max: float = None, max_iter: int = 400) -> tuple[list[str], pd.DataFrame]:
    """VIF > vif_max 피처를 반복 제거 → (축약 피처, VIF 표). diag(inv(corr)) 방식."""
    vif_max = C.VIF_MAX if vif_max is None else vif_max
    X = df[feat_cols].astype(float)
    X = X.loc[:, X.std() > 1e-12]
    X = X.dropna()
    cols = list(X.columns)
    history = []
    for _ in range(max_iter):
        R = np.corrcoef(X[cols].to_numpy(), rowvar=False)
        vif = np.diag(np.linalg.pinv(R))
        worst = int(np.argmax(vif))
        if vif[worst] <= vif_max or len(cols) <= 2:
            final = pd.DataFrame({"피처": cols, "VIF": np.round(vif, 2)})
            return cols, pd.concat([final, pd.DataFrame(history)], ignore_index=True)
        history.append({"피처": cols[worst], "VIF": round(float(vif[worst]), 2),
                        "탈락": True})
        cols.pop(worst)
    final = pd.DataFrame({"피처": cols, "VIF": np.nan})
    return cols, pd.concat([final, pd.DataFrame(history)], ignore_index=True)


def bin_counts(df: pd.DataFrame) -> pd.DataFrame:
    """수질·유량 구간 표본수(§6 수질구간 분석)."""
    rows = []
    tb_tag = C.TURBIDITY_COL.split(".")[-1]
    tb = pd.cut(df[tb_tag], bins=C.TB_BINS, labels=C.TB_BIN_LABELS)
    for k, g in df.groupby(tb, observed=True):
        rows.append({"구분": "착수탁도", "구간": str(k), "n": len(g)})
    temp_tag = C.TEMP_COL.split(".")[-1]
    tp = pd.cut(df[temp_tag], bins=[-1, 10, 20, 40], labels=["≤10℃", "10-20℃", ">20℃"])
    for k, g in df.groupby(tp, observed=True):
        rows.append({"구분": "수온", "구간": str(k), "n": len(g)})
    fl_tag = C.FLOW_COL.split(".")[-1]
    fb = pd.qcut(df[fl_tag], q=[0, .33, .66, 1],
                 labels=["저유량", "중유량", "고유량"], duplicates="drop")
    for k, g in df.groupby(fb, observed=True):
        rows.append({"구분": "유량", "구간": str(k), "n": len(g)})
    season = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12],
                    labels=["겨울(1-3)", "봄(4-6)", "여름(7-9)", "가을(10-12)"])
    for k, g in df.groupby(season, observed=True):
        rows.append({"구분": "계절", "구간": str(k), "n": len(g)})
    return pd.DataFrame(rows)
