# -*- coding: utf-8 -*-
"""유량 정규화(§6.1)·지별 배분(§6.2)·체류시간(§6.3, §8.1)·상류지연(§6.4)."""
import pandas as pd

from . import config as C


def normalize_flow(ft101: pd.Series) -> pd.Series:
    """FT101 → ㎥/h. 단위는 config.FLOW_UNIT 가정 (확정 전 제어명령 금지 §6.1)."""
    if C.FLOW_UNIT == "m3h":
        return ft101.astype(float)
    if C.FLOW_UNIT == "m3d":
        return ft101.astype(float) / 24.0
    raise ValueError(f"알 수 없는 FLOW_UNIT: {C.FLOW_UNIT}")


def basin_flows(q_m3h: pd.Series, weights: dict | None = None) -> pd.DataFrame:
    """지별 유량 Q_basin,j = Q_total × w_j (열=지 이름).

    weights=None이면 실측 유량비(§6.2, 2024-08-27 가동 8지)를 사용.
    가동구성 변경 시에는 균등분배 dict를 넘긴다.
    """
    w = C.FLOW_WEIGHTS_8 if weights is None else weights
    return pd.DataFrame({name: q_m3h * wj for name, wj in w.items()},
                        index=q_m3h.index)


def equal_weights(n: int = C.N_ACTIVE_ASSUMED) -> dict:
    """균등분배 가중치 (§6.2 우선순위 4단계)."""
    return {f"균등#{i + 1}": 1.0 / n for i in range(n)}


def stage_hrt_s(q_basin: pd.DataFrame) -> pd.DataFrame:
    """단별 체류시간(초) = 3600 × (776/3) / Q_basin,j — 3단 동일용량 가정 (§6.3)."""
    v_stage = C.BASIN_VOL_M3 / C.N_STAGES
    return 3600.0 * v_stage / q_basin


def t_drop_s(q_m3h: pd.Series) -> pd.Series:
    """낙차부 체류시간(초) = 99.1 × Q_avg / Q(t) (§8.1)."""
    return C.T_DROP_AVG_S * C.FLOW_AVG_M3H / q_m3h


def upstream_lag_min(q_m3h: pd.Series) -> pd.Series:
    """낙차부→플록형성지 지연(분) = 11.3 × Q_avg / Q(t), 클립 (§6.4).

    유량 결측·0은 평균값 11.3분으로 대체 (정렬 자체는 유지, HOLD는 별도 판정).
    """
    lag = C.T_UPSTREAM_AVG_MIN * C.FLOW_AVG_M3H / q_m3h
    lag = lag.clip(*C.UPSTREAM_LAG_CLIP_MIN)
    return lag.fillna(C.T_UPSTREAM_AVG_MIN)
