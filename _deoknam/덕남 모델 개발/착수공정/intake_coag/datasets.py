# -*- coding: utf-8 -*-
"""학습 데이터셋 (§7.3) — 공통 시간순 분할·태뷸러 pack·시퀀스 생성.

dose_compare/datasets.py 패턴 이식(군집 키 → 정렬조합 키 일반화).
결측 처리는 전 모델 동일(§7.3): 핵심(원수5+유량) 결측행 제외 → 잔여 NaN은
train 중앙값 채움 → 표준화(스케일은 단조변환이라 트리 모델 결과 불변).
"""
import numpy as np
import pandas as pd

from . import config as C

CORE_TAGS = [c.split(".")[-1] for c in C.RAW5 + [C.FLOW_COL]]


def add_seg_id(table: pd.DataFrame) -> pd.DataFrame:
    """스트라이드 격자 공백(> GAP_MIN) 마다 세그먼트 증가 — 시퀀스 경계용."""
    gap = table.index.to_series().diff() > pd.Timedelta(minutes=C.STRIDE_MIN)
    out = table.copy()
    out["seg_id"] = gap.cumsum()
    return out


def core_valid(table: pd.DataFrame) -> pd.Series:
    """핵심(원수5+유량) 유효 + pH 0 무효 행 제외 마스크."""
    m = table[CORE_TAGS].notna().all(axis=1)
    ph_tag = C.PH_COL.split(".")[-1]
    m &= table[ph_tag] != 0
    return m


def split_boundaries(valid_index: pd.DatetimeIndex):
    """시간순 64/16/20 경계 (ts1, ts2) — 전 과제·전 정렬조합 공용(§7.3)."""
    n = len(valid_index)
    i1 = int(n * C.SPLIT["train"])
    i2 = int(n * (C.SPLIT["train"] + C.SPLIT["val"]))
    return valid_index[i1], valid_index[i2]


def make_pack(table: pd.DataFrame, feat_cols: list[str], ycol: str,
              rapid_col: str, ts1, ts2) -> dict | None:
    """태뷸러 (Xtr,ytr,Xva,yva,Xte,yte,...) — 핵심결측·급변행·y결측 제외 후 채움·표준화."""
    m = core_valid(table) & table[ycol].notna() & ~table[rapid_col].fillna(True)
    sub = table[m]
    n_excl = int(len(table) - len(sub))
    tr = sub[sub.index < ts1]
    va = sub[(sub.index >= ts1) & (sub.index < ts2)]
    te = sub[sub.index >= ts2]
    if len(tr) < C.MIN_ROWS or len(te) < C.MIN_ROWS:
        return None
    med = tr[feat_cols].median()
    mu = tr[feat_cols].fillna(med).mean()
    sd = tr[feat_cols].fillna(med).std().replace(0, 1.0)

    def X(part):
        return ((part[feat_cols].fillna(med) - mu) / sd).to_numpy(np.float32)

    return dict(
        Xtr=X(tr), ytr=tr[ycol].to_numpy(float),
        Xva=X(va), yva=va[ycol].to_numpy(float),
        Xte=X(te), yte=te[ycol].to_numpy(float),
        idx_va=va.index, idx_te=te.index,
        aux_va=va, aux_te=te,
        feat_cols=feat_cols, fill=med, mu=mu, sd=sd,
        n_excluded=n_excl, n_total=len(table),
    )


def make_sequences(table: pd.DataFrame, feat_cols: list[str], ycol: str,
                   rapid_col: str, window_min: int, ts1, ts2) -> dict | None:
    """시퀀스 (X3d,y,idx) — 세그먼트 내 슬라이딩 창(스트라이드 10분 격자).

    창 내 피처 NaN → train 중앙값 채움(전 모델 동일 규칙). 창 끝 시점 y·급변행
    조건은 태뷸러와 동일하게 적용.
    """
    step = C.STRIDE_MIN
    w = max(2, window_min // step)
    t = add_seg_id(table)
    end_ok = (core_valid(t) & t[ycol].notna() & ~t[rapid_col].fillna(True)).to_numpy()

    med = t.loc[(t.index < ts1) & core_valid(t), feat_cols].median()
    Xmat = t[feat_cols].fillna(med).to_numpy(np.float32)
    yv = t[ycol].to_numpy(np.float32)
    seg = t["seg_id"].to_numpy()

    Xs, ys, idxs = [], [], []
    for s in np.unique(seg):
        loc = np.flatnonzero(seg == s)
        if len(loc) < w:
            continue
        win = np.lib.stride_tricks.sliding_window_view(Xmat[loc], w, axis=0)
        win = win.transpose(0, 2, 1)                       # (m, w, F)
        end = loc[w - 1:]
        ok = end_ok[end]
        if not ok.any():
            continue
        Xs.append(win[ok])
        ys.append(yv[end[ok]])
        idxs.append(t.index[end[ok]])
    if not Xs:
        return None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    idx = idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0]

    tr = idx < ts1
    va = (idx >= ts1) & (idx < ts2)
    te = idx >= ts2
    if tr.sum() < C.MIN_ROWS or te.sum() < C.MIN_ROWS:
        return None
    flat = X[tr].reshape(-1, len(feat_cols))
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd[sd == 0] = 1.0
    Xsc = (X - mu) / sd
    return dict(
        Xtr=Xsc[tr], ytr=y[tr], Xva=Xsc[va], yva=y[va],
        Xte=Xsc[te], yte=y[te], idx_va=idx[va], idx_te=idx[te],
    )
