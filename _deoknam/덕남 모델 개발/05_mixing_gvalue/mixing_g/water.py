# -*- coding: utf-8 -*-
"""물 물성: 수온(℃) → 밀도 ρ(kg/㎥), 점성계수 μ(Pa·s). 표준표 선형보간."""
import numpy as np

# 0~35℃ 표준 물성 (밀도 kg/㎥, 점성계수 mPa·s)
_T = np.array([0, 5, 10, 15, 20, 25, 30, 35], dtype=float)
_RHO = np.array([999.84, 999.97, 999.70, 999.10, 998.21, 997.05, 995.65, 994.03])
_MU = np.array([1.792, 1.519, 1.307, 1.139, 1.002, 0.890, 0.797, 0.719]) * 1e-3

G_ACC = 9.80665  # 중력가속도 m/s²


def props(temp_c):
    """수온(℃, 스칼라/배열) → (ρ kg/㎥, μ Pa·s). 범위 밖은 경계값 클립."""
    t = np.clip(np.asarray(temp_c, dtype=float), _T[0], _T[-1])
    rho = np.interp(t, _T, _RHO)
    mu = np.interp(t, _T, _MU)
    return rho, mu


def specific_weight(temp_c):
    """비중량 γ = ρ·g (N/㎥)."""
    rho, _ = props(temp_c)
    return rho * G_ACC
