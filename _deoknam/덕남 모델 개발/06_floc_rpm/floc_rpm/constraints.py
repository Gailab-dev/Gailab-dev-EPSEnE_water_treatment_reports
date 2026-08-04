# -*- coding: utf-8 -*-
"""RPM 제약조건·안전검사 (§10.4): 상하한 클립·점감·Gt 범위."""
import numpy as np
import pandas as pd

from . import config as C


def apply_limits(rpm_rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """단별 상하한 클립 → (rpm_safe, 클립 전 기준 위반마스크).

    3단 표준 RPM(0.57~0.70)은 하한 0.5 위이므로 클립되지 않는다 —
    §10.4의 "임의로 3단을 1rpm 이상으로 올리지 않음" 충족.
    """
    safe, viol = {}, []
    for i, k in enumerate((1, 2, 3)):
        lo, hi = C.RPM_LIMITS[k]
        col = rpm_rec.iloc[:, i]
        safe[f"rpm_safe_{k}"] = col.clip(lo, hi)
        viol.append((col.lt(lo) | col.gt(hi)) & col.notna())
    violation = np.logical_or.reduce(viol)
    return (pd.DataFrame(safe, index=rpm_rec.index),
            pd.Series(violation, index=rpm_rec.index, name="limit_violation"))


def taper_violation(rpm_safe: pd.DataFrame) -> pd.Series:
    """점감조건 RPM1 ≥ RPM2 ≥ RPM3 위반 (클립 후 재검사, §10.4)."""
    r1, r2, r3 = (rpm_safe.iloc[:, i] for i in range(3))
    ok = r1.ge(r2) & r2.ge(r3)
    return (~ok & rpm_safe.notna().all(axis=1)).rename("taper_violation")


def gt_by_basin(g_target: tuple, t_stage_s: pd.DataFrame) -> pd.DataFrame:
    """지별 Gt = Σ G_target,k × t_stage,k (§9.2 — 단별 체류시간 동일 3등분)."""
    g_sum = float(np.sum(g_target))
    return t_stage_s * g_sum


def gt_violation(gt_basins: pd.DataFrame) -> pd.Series:
    """지별 Gt 중 하나라도 승인범위(3×10⁴~2×10⁵) 밖이면 True (§9.2).

    자동보정 없이 점검경보만 — 가동지수·유량단위·시설용량 오류 우선 점검 대상.
    """
    out = ((gt_basins.lt(C.GT_RANGE[0]) | gt_basins.gt(C.GT_RANGE[1]))
           & gt_basins.notna()).any(axis=1)
    return out.rename("gt_violation")
