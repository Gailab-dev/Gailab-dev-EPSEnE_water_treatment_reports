# -*- coding: utf-8 -*-
"""교반강도 G값 ↔ RPM 변환, Gt, 수류혼화 G.

G = K0 · √(ρ(T)/μ(T)) · rpm^1.5   (K0: 진단서 표준운영조건 캘리브레이션)
역산: rpm = (G / (K0·√(ρ/μ)))^(2/3) — 해석적.
"""
import numpy as np

from . import config as C
from . import water


def _k0() -> float:
    """캘리브레이션 상수 K0 (기준점: 25℃, 1.72rpm ↔ G50)."""
    rho, mu = water.props(C.CAL_TEMP_C)
    return C.CAL_G / (np.sqrt(rho / mu) * C.CAL_RPM ** 1.5)


K0 = None  # 모듈 로드 시 계산 (아래)


def k_temp(temp_c):
    """수온별 계수 K(T) = K0·√(ρ/μ) — G = K(T)·rpm^1.5."""
    rho, mu = water.props(temp_c)
    return K0 * np.sqrt(rho / mu)


def g_from_rpm(rpm, temp_c):
    """회전수(rpm)·수온(℃) → G값 (/sec)."""
    return k_temp(temp_c) * np.asarray(rpm, dtype=float) ** 1.5


def rpm_for_g(g_target, temp_c):
    """목표 G값·수온 → 필요 회전수 (rpm)."""
    return (np.asarray(g_target, dtype=float) / k_temp(temp_c)) ** (2.0 / 3.0)


def tip_speed_cm_s(rpm):
    """패들 주변속도 (cm/s) = 2πR·rpm/60 × 100."""
    return 2.0 * np.pi * C.R_EFF_M * np.asarray(rpm, dtype=float) / 60.0 * 100.0


def stage_hrt_s(flow_m3d, n_basins: int = C.N_BASINS_TOTAL):
    """단별 체류시간(초) = (지체적/3 × 가동지수) / 총유량."""
    q_m3s = flow_m3d / 86400.0
    v_stage_total = C.BASIN_VOL_M3 / C.N_STAGES * n_basins
    return v_stage_total / q_m3s


def gt(rpm_stages, temp_c, flow_m3d, n_basins: int = C.N_BASINS_TOTAL):
    """단별 rpm 3개 → (단별 G, Gt 합계). Gt = Σ G_i × t_i."""
    g = g_from_rpm(np.asarray(rpm_stages, dtype=float), temp_c)
    t = stage_hrt_s(flow_m3d, n_basins)
    return g, float(np.sum(g) * t)


def hydraulic_g(temp_c, head_m: float = C.HYD_HEAD_M, hrt_s: float = None):
    """수류혼화(낙차) G = √(γ·h_l / (μ·t)) (/sec)."""
    rho, mu = water.props(temp_c)
    gamma = rho * water.G_ACC
    return np.sqrt(gamma * head_m / (mu * hrt_s))


K0 = _k0()
