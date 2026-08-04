# -*- coding: utf-8 -*-
"""체류시간 산정 (§4) — 동적 이론식·추적자 보정·첨두식·정적 후보.

T_outlet(t) = T_상류이송 + T_플록형성지 + T_침전지 + T_센서 (§4.3)
유량 입력 우선순위(§4.3): 지별/계열 실시간 유량 미보유 →
  3순위: FT101 ÷ 가동지수(기본 8) 균등배분  (주 산정식 — 대표 지별유량)
  4순위: 기술진단서 실측 유량비             (지별 편차 범위 보고용)

주의: 유량비 가중 평균 체류시간 Σ w_j·(V/Q_j) (w_j=Q_j/ΣQ)는 수학적으로
균등배분 평균과 동일(V·n/ΣQ)하므로, 4순위는 평균 대체가 아니라 지별
최속(#2-4)~최지(#1-4) T 범위(§4.4 분포형의 폭)를 보고하는 데 쓴다.
"""
import numpy as np
import pandas as pd

from . import config as C


def static_candidates() -> pd.DataFrame:
    """§4.4 초기 후보 4종 재계산 표 — 문서값과 일치 assert."""
    t_up = C.T_UPSTREAM_REF
    rows = [
        ("첨두·최빈 응답", "11.3 + 30 + 80",
         t_up + C.PEAK_FLOC[0] + C.PEAK_SED[0], 121.3),
        ("추적자 보정 평균", "11.3 + 40.9×1.06 + 5.11×60×0.91",
         t_up + 40.9 * C.K_FLOC + 5.11 * 60 * C.K_SED, 333.66),
        ("추적자 조건 이론값", "11.3 + 40.9 + 5.11×60",
         t_up + 40.9 + 5.11 * 60, 358.8),
        ("평균유량 설계표", "11.3 + 58.9 + 6.85×60",
         t_up + 58.9 + 6.85 * 60, 481.2),
    ]
    df = pd.DataFrame(rows, columns=["후보", "계산식", "재계산값(분)", "문서값(분)"])
    assert np.allclose(df["재계산값(분)"], df["문서값(분)"], atol=0.01), \
        "정적 후보 재계산이 설계문서 §4.4와 불일치"
    df["정렬중심(분·1분반올림)"] = df["문서값(분)"].round().astype(int)
    return df


def basin_flow_ratio_bounds() -> tuple[float, float]:
    """진단서 지별 유량비(4순위) — (최속지/균등, 최지지/균등) 유량 배율."""
    flows = np.array(list(C.BASIN_FLOWS.values()), float)
    mean = flows.mean()
    return float(flows.max() / mean), float(flows.min() / mean)


def residence_series(q_total: pd.Series) -> pd.DataFrame:
    """유량 시계열 → T_upstream/T_floc/T_sed/T_theoretical/T_mean/T_peak (분).

    FT101 ≤ 0 또는 결측 → NaN (§12: FT101 이상 시 동적 계산 중지).
    T_mean/T_peak 는 [T_CLIP] 범위로 클립.
    """
    q = pd.to_numeric(q_total, errors="coerce").copy()
    q[q <= 0] = np.nan
    qj = q / C.N_ACTIVE_DEFAULT                  # 3순위: 균등배분 지별유량

    t_up = C.T_UPSTREAM_REF * C.Q_REF / q
    t_floc = 60.0 * C.V_FLOC / qj
    t_sed = 60.0 * C.V_SED / qj
    out = pd.DataFrame(index=q.index)
    out["Q_total"] = q
    out["T_upstream"] = t_up
    out["T_floc"] = t_floc
    out["T_sed"] = t_sed
    out["T_theoretical"] = t_up + t_floc + t_sed + C.T_SENSOR
    out["T_mean"] = (t_up + C.K_FLOC * t_floc + C.K_SED * t_sed + C.T_SENSOR).clip(*C.T_CLIP)
    out["T_peak"] = (t_up
                     + C.PEAK_FLOC[0] * C.PEAK_FLOC[1] / qj
                     + C.PEAK_SED[0] * C.PEAK_SED[1] / qj
                     + C.T_SENSOR).clip(*C.T_CLIP)
    return out


def dynamic_lag_minutes(q_total: pd.Series, kind: str) -> pd.Series:
    """행별 동적 지연(분, 1분 반올림·클립). kind: dynmean|dynpeak."""
    t = residence_series(q_total)
    col = {"dynmean": "T_mean", "dynpeak": "T_peak"}[kind]
    return t[col].round()
