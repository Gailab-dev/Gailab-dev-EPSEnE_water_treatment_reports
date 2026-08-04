# -*- coding: utf-8 -*-
"""교반강도 G값 ↔ RPM 변환, Gt, 수류혼화 G.

G = K0 · √(ρ(T)/μ(T)) · rpm^1.5

용연 진단서는 계수가 1.21배 다른 두 표를 제시하므로 이원 계수를 쓴다:
  - g_from_rpm / gt : 실측 rpm→G (지별 진단·Gt) → K0_DIAG (표 2.3-10/11)
  - rpm_for_g       : 목표 G→권장 rpm (조견표)   → K0_STD  (표 2.3-12)
따라서 rpm_for_g와 g_from_rpm은 서로 정확한 역함수가 아니다(의도적 —
진단서 두 표를 각각 재현하기 위함).
"""
import numpy as np

from . import config as C
from . import water


def _k0(cal) -> float:
    """캘리브레이션 앵커 (temp℃, rpm, G) → K0 = G / (√(ρ/μ)·rpm^1.5)."""
    temp_c, rpm, g = cal
    rho, mu = water.props(temp_c)
    return g / (np.sqrt(rho / mu) * rpm ** 1.5)


K0_DIAG = None  # 지별 진단 계수 (표 2.3-10/11) — 모듈 로드 시 계산
K0_STD = None   # 표준운영 조견표 계수 (표 2.3-12)


def k_temp(temp_c, k0):
    """수온별 계수 K(T) = k0·√(ρ/μ) — G = K(T)·rpm^1.5."""
    rho, mu = water.props(temp_c)
    return k0 * np.sqrt(rho / mu)


def g_from_rpm(rpm, temp_c):
    """실측 회전수(rpm)·수온(℃) → G값 (/sec). 지별 진단 계수(표 2.3-10/11)."""
    return k_temp(temp_c, K0_DIAG) * np.asarray(rpm, dtype=float) ** 1.5


def rpm_for_g(g_target, temp_c):
    """목표 G값·수온 → 권장 회전수 (rpm). 표준운영 조견표 계수(표 2.3-12)."""
    return (np.asarray(g_target, dtype=float) / k_temp(temp_c, K0_STD)) ** (2.0 / 3.0)


def tip_speed_cm_s(rpm):
    """패들 주변속도 (cm/s) = 2πR·rpm/60 × 100 (K0 무관)."""
    return 2.0 * np.pi * C.R_EFF_M * np.asarray(rpm, dtype=float) / 60.0 * 100.0


def stage_hrt_s(flow_m3d, n_basins: int = C.N_BASINS_TOTAL):
    """단별 체류시간(초) = (지체적/3 × 가동지수) / 총유량."""
    q_m3s = flow_m3d / 86400.0
    v_stage_total = C.BASIN_VOL_M3 / C.N_STAGES * n_basins
    return v_stage_total / q_m3s


def gt(rpm_stages, temp_c, flow_m3d, n_basins: int = C.N_BASINS_TOTAL):
    """단별 실측 rpm 3개 → (단별 G, Gt 합계). Gt = Σ G_i × t_i. 진단 계수."""
    g = g_from_rpm(np.asarray(rpm_stages, dtype=float), temp_c)
    t = stage_hrt_s(flow_m3d, n_basins)
    return g, float(np.sum(g) * t)


def hydraulic_g(temp_c, head_m: float = C.HYD_HEAD_M, hrt_s: float = None):
    """수류혼화(낙차) G = √(γ·h_l / (μ·t)) (/sec). K0 무관 — 직접 공식."""
    rho, mu = water.props(temp_c)
    gamma = rho * water.G_ACC
    return np.sqrt(gamma * head_m / (mu * hrt_s))


K0_DIAG = _k0(C.CAL_DIAG)
K0_STD = _k0(C.CAL_STD)
