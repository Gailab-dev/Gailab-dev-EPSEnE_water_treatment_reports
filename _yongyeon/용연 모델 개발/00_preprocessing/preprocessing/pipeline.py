# -*- coding: utf-8 -*-
"""컬럼 그룹별 전처리 파이프라인 오케스트레이션.

A/B(연속 신호): 물리범위 클립 → Hampel∪IQR(∪구간제한 ISO) → NaN → 칼만 채움
C(설정치/계단형): 물리범위 밖만 NaN → 유효구간 내 ffill
용연에는 적산계·전중염소 분할이 없어 덕남의 D그룹/split_chlorine 단계는 제거됨.
"""
import numpy as np
import pandas as pd

from . import config as C
from .kalman import kalman_fill, kalman_smooth, nan_run_len
from .outliers import combined_mask


def _phys_clip(s: pd.Series, col: str):
    lo, hi = C.PHYS_RANGE.get(col, C.DEFAULT_RANGE)
    bad = pd.Series(False, index=s.index)
    if lo is not None:
        bad |= s < lo
    if hi is not None:
        bad |= s > hi
    bad = bad.fillna(False)
    out = s.copy()
    out[bad] = np.nan
    return out, bad


def _flow_floor(s: pd.Series) -> float:
    nonzero = s[s > 0]
    med = float(nonzero.median()) if len(nonzero) else 0.0
    return max(C.FLOW_ABS_MIN, C.FLOW_FLOOR_RATIO * med)


def clean_sensor_column(s: pd.Series, col: str, hampel_params: tuple,
                        iqr_floor: float | None = None):
    """A/B 그룹: 물리범위 → 결합 이상치 마스크 → NaN → 칼만.

    내부 결측 런이 C.MAX_FILL_GAP_MIN 초과면 채우지 않고 NaN 유지.
    반환에 채움 플래그(fill_flag: True=결측을 채운 위치) 포함.
    """
    clean, phys_bad = _phys_clip(s, col)
    if iqr_floor is None:
        iqr_floor = hampel_params[2]
    final, detail = combined_mask(clean, hampel_params, iqr_floor=iqr_floor)
    clean[final] = np.nan
    pre_fill_nan = clean.isna().to_numpy().copy()          # 채움 대상(원결측+이상치제거)
    max_gap = C.MAX_FILL_GAP_MIN
    if col in C.KALMAN_SMOOTH_FULL:
        # 탁도: 전구간 칼만 평활 대체 (참고 프로젝트 KALMAN_SMOOTH 승계)
        smoothed = kalman_smooth(clean, r_scale=C.KALMAN_SMOOTH_FULL[col])
        first, last = clean.first_valid_index(), clean.last_valid_index()
        sm = smoothed.to_numpy().copy()
        outside = (smoothed.index < first) | (smoothed.index > last)
        sm[outside] = np.nan
        if max_gap is not None:
            sm[nan_run_len(pre_fill_nan) > max_gap] = np.nan
        clean = pd.Series(sm, index=clean.index)
    else:
        clean = kalman_fill(clean, r_scale=C.KALMAN_R_DEFAULT, max_gap=max_gap)
    after_nan = clean.isna().to_numpy()
    fill_flag = pd.Series(pre_fill_nan & ~after_nan, index=s.index)     # 실제 채운 위치
    # 장기결측유지 = 유효구간 '내부'의 >max_gap 결측(캡으로 새로 NaN 유지된 칸).
    # 선두/후미 결측은 원래도 미채움이라 제외.
    valid = np.flatnonzero(~pre_fill_nan)
    internal = np.zeros_like(pre_fill_nan)
    if len(valid):
        internal[valid[0]:valid[-1] + 1] = True
    long_kept = int(((nan_run_len(pre_fill_nan) > (max_gap or 10**9)) & internal).sum())
    counts = {
        "물리범위밖": int(phys_bad.sum()),
        "Hampel": int(detail["hampel"].sum()),
        "IQR": int(detail["iqr"].sum()),
        "ISO원탐지": int(detail["iso_raw"].sum()),
        "ISO채택": int(detail["iso_used"].sum()),
        "최종제거": int(final.sum()),
        "장기결측유지": long_kept,
    }
    removed = phys_bad | final
    return clean, removed, counts, fill_flag


def clean_setpoint_column(s: pd.Series, col: str):
    """C 그룹: 물리범위 밖만 NaN → 유효구간 내 ffill (계단 엣지 보존).

    설정치는 값 유지가 의미상 타당하므로 결측 길이 캡 미적용(ffill 유지).
    채움 플래그만 생성.
    """
    clean, phys_bad = _phys_clip(s, col)
    pre_fill_nan = clean.isna().to_numpy().copy()
    clean = clean.ffill(limit_area="inside")
    fill_flag = pd.Series(pre_fill_nan & ~clean.isna().to_numpy(), index=s.index)
    counts = {"물리범위밖": int(phys_bad.sum()), "Hampel": 0, "IQR": 0,
              "ISO원탐지": 0, "ISO채택": 0, "최종제거": int(phys_bad.sum()),
              "장기결측유지": 0}
    return clean, phys_bad, counts, fill_flag


def run_pipeline(df: pd.DataFrame, verbose: bool = True):
    """전 컬럼 그룹 디스패치 → 리포트.

    반환: (df_clean, report_df, removed_masks dict, fill_flags_df)
    fill_flags_df: 컬럼별 bool (True=결측을 채운 위치). 모델이 진짜값/채운값 구분용.
    """
    groups = {
        **{c: "A.수질센서" for c in C.SENSOR_COLS},
        **{c: "B.유량수위" for c in C.FLOW_COLS},
        **{c: "C.설정치" for c in C.SETPOINT_COLS},
    }
    missing = set(df.columns) - set(groups)
    extra = set(groups) - set(df.columns)
    assert not missing and not extra, f"그룹 미할당 컬럼: {missing}, 데이터에 없는 컬럼: {extra}"

    n0 = len(df)
    df_clean = df.copy()
    report, masks, flags = [], {}, {}
    for col in df.columns:
        grp = groups[col]
        s = df[col]
        if (getattr(C, "WATER_QUALITY_ONLY", False) and grp != "A.수질센서"
                and col not in getattr(C, "PREPROCESS_FLOW_COLS", [])):
            # 공정컬럼(B/C/D): 전처리 미적용, 원시값 그대로 통과
            df_clean[col] = s
            masks[col] = pd.Series(False, index=df.index)
            flags[col] = pd.Series(False, index=df.index)
            report.append({
                "변수": col, "그룹": grp + "(원시통과)",
                "물리범위밖": 0, "Hampel": 0, "IQR": 0, "ISO원탐지": 0, "ISO채택": 0,
                "최종제거": 0, "장기결측유지": 0, "제거율(%)": 0.0, "결측채움행": 0,
            })
            if verbose:
                print(f"통과: [{grp}] {col} — 전처리 미적용(원시값)")
            continue
        if grp == "A.수질센서":
            clean, removed, counts, fill_flag = clean_sensor_column(s, col, C.HAMPEL[col])
        elif grp == "B.유량수위":
            params = (*C.FLOW_HAMPEL, _flow_floor(s))
            nonzero = s[s > 0]
            med_nz = float(nonzero.median()) if len(nonzero) else 0.0
            iqr_floor = max(C.FLOW_ABS_MIN, C.FLOW_IQR_FLOOR_RATIO * med_nz)
            clean, removed, counts, fill_flag = clean_sensor_column(
                s, col, params, iqr_floor=iqr_floor)
        else:
            clean, removed, counts, fill_flag = clean_setpoint_column(s, col)
        df_clean[col] = clean
        masks[col] = removed
        flags[col] = fill_flag
        report.append({
            "변수": col, "그룹": grp, **counts,
            "제거율(%)": round(counts["최종제거"] / n0 * 100, 3),
            "결측채움행": int(fill_flag.sum()),
        })
        if verbose:
            print(f"완료: [{grp}] {col} — 최종제거 {counts['최종제거']:,}행 "
                  f"({counts['최종제거'] / n0 * 100:.3f}%), "
                  f"채움 {int(fill_flag.sum()):,} / 장기결측유지 {counts['장기결측유지']:,}")

    fill_flags = pd.DataFrame(flags, index=df.index)
    return df_clean, pd.DataFrame(report), masks, fill_flags
