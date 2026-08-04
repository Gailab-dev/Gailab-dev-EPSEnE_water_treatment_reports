# -*- coding: utf-8 -*-
"""조견표·진단표 생성: 수온별 권장 RPM, 지별 운영 진단, 일별 권장 RPM 시계열."""
import numpy as np
import pandas as pd

from . import config as C
from . import gvalue


def rpm_table(temps=None) -> pd.DataFrame:
    """수온별 × 단별 목표 G 달성 필요 rpm 조견표 (+주변속도)."""
    temps = np.arange(0, 31) if temps is None else np.asarray(temps)
    rows = []
    for t in temps:
        row = {"수온(℃)": float(t)}
        for s, g in C.G_TARGETS.items():
            rpm = float(gvalue.rpm_for_g(g, t))
            row[f"{s}단 rpm (G={g:.0f})"] = round(rpm, 2)
            row[f"{s}단 주변속도(cm/s)"] = round(float(gvalue.tip_speed_cm_s(rpm)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_basins(temp_c: float, flow_m3d: float,
                    n_basins_op: int = None) -> pd.DataFrame:
    """지별 실측 rpm(진단서) → 단별 G, Gt, 기준 판정.

    n_basins_op: Gt 체류시간 계산에 쓸 가동 지수 (기본: 전체 12지 — 진단서 방식).
    """
    n = C.N_BASINS_TOTAL if n_basins_op is None else n_basins_op
    rows = []
    for basin, rpms in C.BASIN_RPM.items():
        g, gt_v = gvalue.gt(rpms, temp_c, flow_m3d, n)
        issues = []
        if g[2] < C.G_STAGE3_MIN:
            issues.append(f"3단 G {g[2]:.1f} < 기준 {C.G_STAGE3_MIN:.0f}")
        if not (g[0] >= g[1] >= g[2]):
            issues.append("점감 위배")
        if not (C.GT_RANGE[0] <= gt_v <= C.GT_RANGE[1]):
            issues.append("Gt 범위 밖")
        rows.append({
            "지": basin,
            "rpm(1/2/3단)": "/".join(f"{r:.2f}" for r in rpms),
            "G 1단": round(float(g[0]), 1), "G 2단": round(float(g[1]), 1),
            "G 3단": round(float(g[2]), 1), "Gt": f"{gt_v:.2e}",
            "권장 rpm(1/2/3단)": "/".join(
                f"{float(gvalue.rpm_for_g(C.G_TARGETS[s], temp_c)):.2f}" for s in (1, 2, 3)),
            "판정": "적정" if not issues else "; ".join(issues),
        })
    return pd.DataFrame(rows)


def daily_recommended_rpm() -> pd.DataFrame:
    """전처리 데이터의 착수정 온도 일평균 → 일별 단별 권장 rpm 시계열."""
    temp = pd.read_parquet(C.IN_PARQUET, columns=[C.COL_TEMP])[C.COL_TEMP]
    day = temp.resample("1D").mean().dropna().rename("수온")
    out = day.to_frame()
    for s, g in C.G_TARGETS.items():
        out[f"{s}단 권장rpm"] = np.round(gvalue.rpm_for_g(g, day.to_numpy()), 2)
    return out
