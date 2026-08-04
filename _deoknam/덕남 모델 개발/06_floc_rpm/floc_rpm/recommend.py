# -*- coding: utf-8 -*-
"""RPM 추천: 표준 RPM 룩업보간(§10.1) + 목표 G 보정(§10.2) + K0 검산(§10.3 역할)
+ 낙차부 G 감시(§8.2).

추천 기준은 표 2.3-13 선형보간값이며, 05_mixing_gvalue의 K0 물리모델은
독립 검산·경보(WARN_MODEL_DIVERGENCE)로만 쓴다.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import mixing_link


def interp_std_rpm(temp_c: pd.Series) -> pd.DataFrame:
    """수온 → 단별 표준 RPM 선형보간. 5~30℃ 밖은 경계값 클램프 (§7, §10.1)."""
    t = temp_c.clip(*C.TEMP_STD_C).to_numpy(dtype=float)
    out = pd.DataFrame(
        {f"rpm_std_{k}": np.interp(t, C.STD_RPM_TEMPS, C.STD_RPM_TABLE[k])
         for k in (1, 2, 3)},
        index=temp_c.index)
    out.loc[temp_c.isna()] = np.nan
    return out


def correct_rpm(rpm_std: pd.DataFrame, g_target: tuple,
                g_base: tuple = C.G_BASE) -> pd.DataFrame:
    """RPM_rec,k = RPM_std,k × (G_target,k/G_base,k)^(2/3) — G²∝RPM³ (§10.2)."""
    factor = (np.asarray(g_target, dtype=float) / np.asarray(g_base)) ** (2.0 / 3.0)
    out = rpm_std.to_numpy() * factor
    return pd.DataFrame(out, index=rpm_std.index,
                        columns=[f"rpm_rec_{k}" for k in (1, 2, 3)])


def crosscheck_k0(rpm_rec: pd.DataFrame, temp_c: pd.Series,
                  g_target: tuple = C.G_BASE) -> pd.DataFrame:
    """K0 물리모델 독립 검산 (§10.3 역할).

    rpm_phys_k = 05 모델의 목표 G 필요 회전수. 최대 상대편차가 문턱(3%)을
    넘으면 warn — 구현버그·클램프 구간·물성 이상 신호.
    """
    t = temp_c.clip(*C.TEMP_STD_C)
    phys = pd.DataFrame(
        {f"rpm_phys_{k}": mixing_link.rpm_for_g(g, t.to_numpy(dtype=float))
         for k, g in zip((1, 2, 3), g_target)},
        index=temp_c.index)
    phys.loc[temp_c.isna()] = np.nan
    dev = np.abs(rpm_rec.to_numpy() - phys.to_numpy()) / phys.to_numpy()
    phys["k0_dev_max"] = pd.DataFrame(dev, index=phys.index).max(axis=1)
    phys["warn_divergence"] = phys["k0_dev_max"].gt(C.MODEL_DIVERGENCE_WARN) \
                                                .fillna(False)
    return phys


def g_drop(temp_now: pd.Series, t_drop: pd.Series) -> pd.Series:
    """낙차부 수류혼화 G = √(ρ·g·h_loss/(μ·T_drop)) (§8.2).

    현재 착수정 수온 사용 (낙차부는 상류정렬 대상 아님).
    """
    rho, mu = mixing_link.props(temp_now.to_numpy(dtype=float))
    out = np.sqrt(rho * mixing_link.G_ACC * C.HYD_HEAD_M
                  / (mu * t_drop.to_numpy(dtype=float)))
    return pd.Series(out, index=temp_now.index, name="g_drop") \
        .where(temp_now.notna())
