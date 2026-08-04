# -*- coding: utf-8 -*-
"""§12 침전수 탁도 연계 사후평가.

RPM 변경효과 평가창은 상류 11.3분을 더하지 않는다(§12) — 플록형성지→출수부
지연 중심 4개(첨두 110·추적자평균 322·추적자이론 348·설계표 470분) × 반폭
(±30/45/60분) 후보를 상관으로 비교하고, 기본 중심(추적자평균)으로 사후평가
테이블을 만든다. 우세 후보는 리포트로 권고만 하고 자동 변경하지 않는다.
"""
import numpy as np
import pandas as pd

from . import alignment
from . import config as C


def metrics(y: pd.Series, yhat: pd.Series,
            eps: float = C.SMAPE_EPS_NTU) -> dict:
    """§13.3 평가식 그대로: RMSE·MAE·sMAPE·R2."""
    m = y.notna() & yhat.notna()
    y, yhat = y[m].to_numpy(), yhat[m].to_numpy()
    if len(y) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "sMAPE(%)": np.nan, "R2": np.nan,
                "행수": 0}
    err = y - yhat
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(yhat) + eps)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "sMAPE(%)": smape,
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "행수": int(len(y))}


def align_outlet(tb: pd.Series, center_min: int, half_min: int) -> pd.Series:
    """TB_result(t) = median(TB_outlet[t+center-Δ : t+center+Δ])."""
    return alignment.rolling_median_shift(tb, center_min, half_min)


def compare_windows(df: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """시간창 후보(4중심 × 3반폭 × 침전지 1·2) × 드라이버 Spearman 상관 비교.

    drivers: 플록형성지 도달 기준으로 정렬된 운전조건 (원수탁도·약품SV·유량).
    상관이 일관되게 높은 창이 인과 지연을 더 잘 포착한다고 본다.
    """
    rows = []
    lo, hi = C.LAG_SEARCH_RANGE_MIN
    for name, center in C.LAG_CENTERS_MIN.items():
        for half in C.LAG_HALF_WIN_MIN:
            if not (lo <= center - half and center + half <= hi):
                continue  # §12 탐색범위(80~530분) 밖 창은 제외
            for sed, col in C.COL_SED_TB.items():
                tb_res = align_outlet(df[col], center, half)
                row = {"시차중심": name, "중심(분)": center, "반폭(분)": half,
                       "침전지": sed, "유효행수": int(tb_res.notna().sum())}
                for dname, drv in drivers.items():
                    m = tb_res.notna() & drv.notna()
                    if m.sum() > 100:
                        # 10분 서브샘플로 자기상관 중복 축소 (순위상관은 견고)
                        sub = m[m].index[::10]
                        row[f"ρ({dname})"] = round(float(
                            tb_res[sub].corr(drv[sub], method="spearman")), 4)
                    else:
                        row[f"ρ({dname})"] = np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    drv_cols = [c for c in out.columns if c.startswith("ρ(")]
    out["ρ평균절대값"] = out[drv_cols].abs().mean(axis=1).round(4)
    return out.sort_values("ρ평균절대값", ascending=False).reset_index(drop=True)


def posthoc_table(out: pd.DataFrame, df: pd.DataFrame,
                  center_min: int | None = None,
                  half_min: int = C.LAG_HALF_WIN_DEFAULT) -> pd.DataFrame:
    """§12 사후평가 스냅샷: 기본 시간창으로 정렬한 침전수 탁도를 추천시점에 연결.

    실측 RPM이 없으므로 '추천 전·후'는 표준 추천 RPM만 기록한다. 반환은 1분
    전기간 (run에서 parquet 저장 + 일별 요약 CSV 생성).
    """
    center = C.LAG_CENTERS_MIN[C.LAG_DEFAULT] if center_min is None else center_min
    snap = out[["q_m3h", "수온_정렬", "원수탁도_정렬", "원수pH_정렬", "알칼리도_정렬",
                "PAC_SV_정렬", "PACS2_SV_정렬", "g_drop",
                "rpm_safe_1", "rpm_safe_2", "rpm_safe_3",
                "gt_min", "gt_max", "reason_code", "reason_bitmask"]].copy()
    for sed, col in C.COL_SED_TB.items():
        snap[f"침전지{sed}_탁도_정렬"] = align_outlet(df[col], center, half_min)
        snap[f"침전지{sed}_pH_정렬"] = align_outlet(df[C.COL_SED_PH[sed]],
                                                   center, half_min)
    snap.attrs["시간창"] = f"{C.LAG_DEFAULT} {center}±{half_min}분"
    return snap


def daily_posthoc(snap: pd.DataFrame) -> pd.DataFrame:
    """사후평가 일별 요약 (중앙값) + HOLD 비율."""
    num = snap.select_dtypes("number")
    day = num.resample("1D").median()
    day["HOLD비율"] = snap["reason_code"].str.startswith("HOLD") \
                                         .resample("1D").mean().round(4)
    return day.round(4)
