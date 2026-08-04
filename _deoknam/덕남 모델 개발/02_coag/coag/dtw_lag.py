# -*- coding: utf-8 -*-
"""DTW 기반 체류시간(lag) 추정.

군집은 시간상 교차 세그먼트로 이뤄지므로 세그먼트 단위로
원수탁도(착수정_TB) ↔ 침전지N_탁도의 밴드 DTW 정렬 경로를 구하고,
경로의 수직 오프셋(j-i)×10분 중앙값을 해당 세그먼트의 lag로 본다.
군집×침전지 lag = 세그먼트 길이 가중 중앙값 (10분 배수 반올림).
외부 라이브러리 없이 순수 numpy 구현 (Sakoe-Chiba 밴드).
"""
import numpy as np
import pandas as pd

from . import config as C

STEP_MIN = 10  # 10분 리샘플 스텝


def banded_dtw_path(a: np.ndarray, b: np.ndarray, band: int) -> np.ndarray:
    """Sakoe-Chiba 밴드 DTW 최적 경로. 반환 (L, 2) [i, j].

    lag(타깃이 소스보다 늦음)만 관심이므로 j-i ∈ [0, 2*band] 오프셋 창을 사용.
    """
    n, m = len(a), len(b)
    width = 2 * band + 1
    INF = np.inf
    # cost[i, k] = D(i, j=i+k), k ∈ [0, width)
    D = np.full((n, width), INF)
    step = np.zeros((n, width), dtype=np.int8)  # 0=diag, 1=left(j-1), 2=up(i-1)
    for i in range(n):
        j_lo, j_hi = i, min(i + width - 1, m - 1)
        if j_lo > j_hi:
            break
        ks = np.arange(j_lo - i, j_hi - i + 1)
        cost = np.abs(a[i] - b[i + ks])
        if i == 0:
            # 첫 행: 왼쪽 이동만 허용 (j 누적)
            acc = np.cumsum(cost)
            D[0, ks] = acc
            step[0, ks[1:]] = 1
            continue
        # diag: D[i-1, k] (j-1 → k 동일-1+1 = k), up: D[i-1, k+1], left: D[i, k-1]
        diag = D[i - 1, ks]                                   # (i-1, j-1) → offset k
        up = np.full(len(ks), INF)
        up_idx = ks + 1
        valid = up_idx < width
        up[valid] = D[i - 1, up_idx[valid]]
        for t, k in enumerate(ks):
            left = D[i, k - 1] if k - 1 >= 0 else INF
            best = diag[t]
            s = 0
            if left < best:
                best, s = left, 1
            if up[t] < best:
                best, s = up[t], 2
            D[i, k] = cost[t] + best
            step[i, k] = s
    # 종점: i=n-1에서 유효한 최소 비용 k
    end_k = int(np.argmin(D[n - 1]))
    if not np.isfinite(D[n - 1, end_k]):
        return np.empty((0, 2), dtype=int)
    path = []
    i, k = n - 1, end_k
    while True:
        path.append((i, i + k))
        s = step[i, k]
        if i == 0 and k == 0:
            break
        if s == 0:
            if i == 0:
                k -= 1
            else:
                i -= 1
        elif s == 1:
            k -= 1
        else:
            i -= 1
            k += 1
        if k < 0:
            break
    return np.array(path[::-1], dtype=int)


def _znorm(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def segment_lag(src: np.ndarray, tgt: np.ndarray):
    """한 세그먼트의 lag(분). 경로 양끝 trim 후 오프셋 중앙값. 범위 밖이면 None."""
    path = banded_dtw_path(_znorm(src), _znorm(tgt), C.DTW["band_steps"])
    if len(path) < 10:
        return None
    trim = max(1, int(len(path) * C.DTW["trim_frac"]))
    offs = (path[trim:-trim, 1] - path[trim:-trim, 0]) * STEP_MIN
    lag = float(np.median(offs))
    if not (C.DTW["lag_min"] <= lag <= C.DTW["lag_max"]):
        return None
    return lag


def xcorr_lag(src: np.ndarray, tgt: np.ndarray):
    """참고용 교차상관 lag(분): corr(src[t], tgt[t+k]) 최대 k."""
    s, t = _znorm(src), _znorm(tgt)
    max_k = C.DTW["band_steps"] * 2
    best_k, best_c = None, -np.inf
    for k in range(0, max_k + 1):
        if len(s) - k < 20:
            break
        c = float(np.corrcoef(s[: len(s) - k], t[k:])[0, 1])
        if c > best_c:
            best_c, best_k = c, k
    if best_k is None:
        return None
    lag = best_k * STEP_MIN
    return lag if C.DTW["lag_min"] <= lag <= C.DTW["lag_max"] else None


def estimate_cluster_lags(df10: pd.DataFrame, cid: int):
    """세그먼트별 DTW lag → 군집×침전지 대표 lag.

    반환: (lags dict {타깃컬럼: lag분(10분 배수)}, 세그먼트별 상세 DataFrame)
    """
    src_col = "RCS_6.AI.착수정_TB"
    rows = []
    lags = {}
    for tgt_col in C.TARGET_COLS:
        seg_lags = []
        for sid, seg in df10.groupby("seg_id"):
            if len(seg) < C.DTW["min_seg_steps"]:
                continue
            src = seg[src_col].to_numpy(float)
            tgt = seg[tgt_col].to_numpy(float)
            if np.isnan(src).any() or np.isnan(tgt).any():
                continue
            # 평탄 세그먼트(변동 없는 겨울 저탁도 등)는 정렬이 노이즈에 지배됨 → 제외
            if src.std() < C.DTW["min_std_src"] or tgt.std() < C.DTW["min_std_tgt"]:
                continue
            lag = segment_lag(src, tgt)
            xl = xcorr_lag(src, tgt)
            rows.append({"cluster": cid, "타깃": tgt_col.split(".")[-1], "seg_id": sid,
                         "세그먼트길이(스텝)": len(seg), "DTW_lag(분)": lag, "xcorr_lag(분)": xl})
            if lag is not None:
                seg_lags.append(lag)
        if len(seg_lags) >= C.DTW["min_valid_segs"]:
            # 단순 중앙값 — 길이 가중은 평탄한 장기 세그먼트에 끌려 편향됨
            lag_final = int(round(float(np.median(seg_lags)) / STEP_MIN) * STEP_MIN)
        else:
            # 유효 세그먼트 부족(평탄 군집) → 기존 연구 체류시간으로 폴백
            lag_final = C.PRIOR_LAGS[cid]
        lags[tgt_col] = lag_final
    detail = pd.DataFrame(rows)
    return lags, detail
