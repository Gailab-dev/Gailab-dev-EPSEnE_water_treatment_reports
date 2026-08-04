# -*- coding: utf-8 -*-
"""피처 생성 (설계문서 §6.1). 입력은 t 이전 값만 — shift/diff/rolling 전부 과거 참조."""
import numpy as np
import pandas as pd

from . import config as C


def load_labeled() -> pd.DataFrame:
    """군집라벨 정제데이터 + 채움플래그(결측 표식) 로드."""
    df = pd.read_parquet(C.LABELED_PARQUET).sort_index()
    try:
        flags = pd.read_parquet(C.FILL_FLAGS_PARQUET).sort_index()
        flags = flags.reindex(df.index)
    except FileNotFoundError:
        flags = None
    return df, flags


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def build_features(df: pd.DataFrame, flags: pd.DataFrame | None,
                   target_col: str = C.TARGET):
    """피처 테이블 반환. (table, feat_ar, feat_noar)
    target_col: 예측 타깃 컬럼. AR 피처(직전값·경과시간)는 이 타깃 기준으로 생성.
    feat_ar/feat_noar = AR 포함/제외 피처 컬럼 리스트.
    """
    out = pd.DataFrame(index=df.index)
    cont = C.RAW5 + [C.FLOW_COL] + C.LEVEL_COLS
    base_feats = []
    for col in cont:
        s = _num(df, col)
        tag = col.split(".")[-1]
        out[f"{tag}"] = s
        base_feats.append(tag)
        for k in C.LAGS:
            out[f"{tag}__lag{k}"] = s.shift(k)
            out[f"{tag}__d{k}"] = s - s.shift(k)
            base_feats += [f"{tag}__lag{k}", f"{tag}__d{k}"]
        for w in C.ROLL:
            out[f"{tag}__ma{w}"] = s.rolling(w, min_periods=max(2, w // 3)).mean()
            base_feats.append(f"{tag}__ma{w}")

    # 시간 피처
    idx = df.index
    for name, period, val in [("hour", 24, idx.hour), ("month", 12, idx.month),
                              ("dow", 7, idx.dayofweek)]:
        out[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
        base_feats += [f"{name}_sin", f"{name}_cos"]

    # 소독 설정 변경 여부(직전 대비 변화) — 과거 정보
    chg = pd.Series(0.0, index=idx)
    for col in C.DISINFECT_COLS:
        if col in df.columns:
            chg = chg + (_num(df, col).diff().abs() > 1e-9).astype(float)
    out["소독설정변경"] = (chg > 0).astype(float)
    base_feats.append("소독설정변경")

    # 결측·채움 플래그 (원수성상이 채워진 값인지) — 과거/현재 상태
    if flags is not None:
        for col in C.RAW5 + [C.FLOW_COL]:
            if col in flags.columns:
                tag = col.split(".")[-1] + "__filled"
                out[tag] = flags[col].astype(float).values
                base_feats.append(tag)

    # 타깃 + AR(직전값·마지막 변경 후 경과시간). 직전값=과거 정보(t-1)만.
    tgt = _num(df, target_col)
    out["__target__"] = tgt
    prev = tgt.shift(1)
    out["SV_prev"] = prev
    changed = (prev != prev.shift(1))
    grp = changed.cumsum()
    out["SV_since_change"] = out.groupby(grp).cumcount().astype(float)
    ar_feats = ["SV_prev", "SV_since_change"]

    out[C.LABEL_COL] = df[C.LABEL_COL].values
    feat_ar = base_feats + ar_feats
    feat_noar = base_feats
    return out, feat_ar, feat_noar


def quality_filter(table: pd.DataFrame, tmin: float = None,
                   tmax: float = None) -> tuple[pd.DataFrame, dict]:
    """설계문서 §7: pH 0 무효, 핵심 결측·타깃 유효범위 밖 제외. (표, 제외집계)."""
    tmin = C.TARGET_MIN if tmin is None else tmin
    tmax = C.TARGET_MAX if tmax is None else tmax
    n0 = len(table)
    reasons = {}
    m = pd.Series(True, index=table.index)

    ph_tag = C.PH_COL.split(".")[-1]
    if ph_tag in table.columns:
        bad_ph = (table[ph_tag] == 0)
        reasons["pH_0"] = int(bad_ph.sum())
        m &= ~bad_ph.fillna(False)

    core_tags = [c.split(".")[-1] for c in C.CORE_REQUIRED]
    core_na = table[core_tags].isna().any(axis=1)
    reasons["핵심결측"] = int(core_na.sum())
    m &= ~core_na

    t = table["__target__"]
    bad_t = t.isna() | (t < tmin) | (t > tmax)
    reasons["타깃무효"] = int(bad_t.sum())
    m &= ~bad_t

    out = table[m].copy()
    reasons["최종유효"] = len(out)
    reasons["전체"] = n0
    return out, reasons
