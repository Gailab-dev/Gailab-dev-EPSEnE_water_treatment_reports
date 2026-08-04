# -*- coding: utf-8 -*-
"""시간순 분할·군집별 필터·스케일링, 태뷸러/시퀀스 데이터셋 생성."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C


def split_times(table: pd.DataFrame):
    """시간순 64/16/20 경계 타임스탬프 (ts1, ts2). table은 시간정렬 가정."""
    n = len(table)
    i1 = int(n * C.SPLIT["train"])
    i2 = int(n * (C.SPLIT["train"] + C.SPLIT["val"]))
    return table.index[i1], table.index[i2]


def tabular_by_cluster(table: pd.DataFrame, feat_cols: list, ts1, ts2):
    """군집별 (Xtr,ytr,Xval,yval,Xte,yte, idx_te, scaler) — 피처 결측행 제거 후 스케일."""
    cols = feat_cols + ["__target__", C.LABEL_COL]
    t = table[cols].dropna(subset=feat_cols + ["__target__"])
    out = {}
    for cid in C.CLUSTERS:
        sub = t[t[C.LABEL_COL] == cid]
        tr = sub[sub.index < ts1]
        va = sub[(sub.index >= ts1) & (sub.index < ts2)]
        te = sub[sub.index >= ts2]
        if len(tr) < 50 or len(te) < 50:
            out[cid] = None
            continue
        sx = StandardScaler().fit(tr[feat_cols].to_numpy())
        pack = dict(
            Xtr=sx.transform(tr[feat_cols].to_numpy()), ytr=tr["__target__"].to_numpy(float),
            Xva=sx.transform(va[feat_cols].to_numpy()) if len(va) else np.empty((0, len(feat_cols))),
            yva=va["__target__"].to_numpy(float),
            Xte=sx.transform(te[feat_cols].to_numpy()), yte=te["__target__"].to_numpy(float),
            idx_te=te.index, sv_prev_te=(te["SV_prev"].to_numpy(float) if "SV_prev" in te else None),
            scaler=sx, feat_cols=feat_cols,
            aux_te=te,   # 구간별 분석용 원본(피처 스케일 전)
        )
        out[cid] = pack
    return out


def resample_for_seq(table: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """시퀀스용 10분 리샘플(피처 mean, 타깃·라벨 last)."""
    agg = {c: "mean" for c in feat_cols}
    agg["__target__"] = "last"
    agg[C.LABEL_COL] = "last"
    r = table[list(agg)].resample(C.SEQ_RESAMPLE).agg(agg)
    return r


def _segments(idx: pd.DatetimeIndex, step_min: int):
    """리샘플 간격 초과 공백마다 세그먼트 id."""
    gap = idx.to_series().diff() > pd.Timedelta(minutes=C.SEQ_GAP_MIN)
    return gap.cumsum().to_numpy()


def sequences_by_cluster(rs: pd.DataFrame, feat_cols: list, window_min: int, ts1, ts2):
    """군집별 시퀀스 (X3d, y, idx) train/val/test. window_min=분 → 10분스텝 환산."""
    step = int(pd.Timedelta(C.SEQ_RESAMPLE).total_seconds() // 60)
    w = max(2, window_min // step)
    seg = _segments(rs.index, step)
    F = feat_cols
    Xs, ys, idxs = [], [], []
    Xmat = rs[F].to_numpy(np.float32)
    yv = rs["__target__"].to_numpy(np.float32)
    lab = rs[C.LABEL_COL].to_numpy()
    for s in np.unique(seg):
        loc = np.flatnonzero(seg == s)
        if len(loc) < w:
            continue
        block = Xmat[loc]
        if not np.isfinite(block).all():
            # 결측 포함 창은 스킵 위해 유효행만? 여기선 세그먼트 통째 결측이면 제외
            block = np.nan_to_num(block, nan=0.0)
        win = np.lib.stride_tricks.sliding_window_view(block, w, axis=0)  # (m, F, w)
        win = win.transpose(0, 2, 1)                                       # (m, w, F)
        end = loc[w - 1:]
        Xs.append(win)
        ys.append(yv[end])
        idxs.append(rs.index[end])
    if not Xs:
        empty = np.empty((0, w, len(F)), np.float32)
        return {cid: None for cid in C.CLUSTERS}
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    idx = idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0]
    endlab = pd.Series(lab, index=rs.index).reindex(idx).to_numpy()

    # 스케일러: 전체 train 창으로 적합 (군집 무관 공통 스케일)
    tr_mask = idx < ts1
    valid_y = np.isfinite(y)
    flat = X[tr_mask & valid_y].reshape(-1, len(F))
    mu = np.nanmean(flat, axis=0) if len(flat) else np.zeros(len(F))
    sd = np.nanstd(flat, axis=0) if len(flat) else np.ones(len(F))
    sd[sd == 0] = 1.0
    Xsc = (X - mu) / sd

    out = {}
    for cid in C.CLUSTERS:
        m = (endlab == cid) & valid_y
        tr = m & (idx < ts1)
        va = m & (idx >= ts1) & (idx < ts2)
        te = m & (idx >= ts2)
        if tr.sum() < 50 or te.sum() < 50:
            out[cid] = None
            continue
        out[cid] = dict(
            Xtr=Xsc[tr], ytr=y[tr], Xva=Xsc[va], yva=y[va],
            Xte=Xsc[te], yte=y[te], idx_te=idx[te],
        )
    return out
