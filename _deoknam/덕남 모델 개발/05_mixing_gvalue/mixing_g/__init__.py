# -*- coding: utf-8 -*-
"""덕남 플록큐레이터 교반강도(G값)·RPM 산정 패키지.

사용 예:
    from mixing_g import gvalue
    gvalue.g_from_rpm(1.2, temp_c=10)        # rpm·수온 → G값
    gvalue.rpm_for_g(50, temp_c=10)          # 목표 G → 필요 rpm
    gvalue.gt([2.0, 1.4, 0.7], 10, 227728)   # 단별 G와 Gt
"""
from . import config
from .gvalue import K0, g_from_rpm, gt, hydraulic_g, rpm_for_g, tip_speed_cm_s
from .water import props, specific_weight

__all__ = ["config", "K0", "g_from_rpm", "rpm_for_g", "gt", "hydraulic_g",
           "tip_speed_cm_s", "props", "specific_weight"]
