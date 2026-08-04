# -*- coding: utf-8 -*-
"""용연 플록큐레이터 교반강도(G값)·RPM 산정 패키지.

사용 예:
    from mixing_g import gvalue
    gvalue.g_from_rpm(1.82, temp_c=15)       # 실측 rpm·수온 → G값 (진단 계수)
    gvalue.rpm_for_g(50, temp_c=15)          # 목표 G → 권장 rpm (조견표 계수)
    gvalue.gt([1.82, 1.59, 1.06], 15, 276540)  # 단별 G와 Gt
"""
from . import config
from .gvalue import (K0_DIAG, K0_STD, g_from_rpm, gt, hydraulic_g, rpm_for_g,
                     tip_speed_cm_s)
from .water import props, specific_weight

__all__ = ["config", "K0_DIAG", "K0_STD", "g_from_rpm", "rpm_for_g", "gt",
           "hydraulic_g", "tip_speed_cm_s", "props", "specific_weight"]
