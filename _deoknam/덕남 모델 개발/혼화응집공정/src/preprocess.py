# -*- coding: utf-8 -*-
"""덕남 통합 데이터 전처리 파이프라인.

스파이크 대조 분석(plot_spike_compare)에서 확인한 문제를 규칙으로 반영:
  1) 물리 범위 밖 값 제거 (pH<5, 전기전도도<40, 알칼리도<5, PAC SV>20 등 센서 오류)
  2) Hampel 필터로 순간 스파이크 제거 — 롤링 중앙값 대비 MAD 기준.
     지속된 강우 급등은 중앙값이 함께 올라가 자동 보존, 좁은 첨두만 제거됨.
  3) 침전지 탁도(타깃)는 상한 포화(3.0) 플래그 + 15분 중앙값 평활로 노이즈 완화.

산출: dataset/덕남_통합_전처리.parquet (원본 + *_clean 컬럼 + *_flag 컬럼)
실행: python -m src.preprocess
"""
import numpy as np
import pandas as pd

from . import common as C
from .smoothing_compare import kalman_smooth

INTEGRATED = C.DATA_DIR / "덕남_응집제공정_소독공정_통합.parquet"
OUT = C.DATA_DIR / "덕남_통합_전처리.parquet"

# 변수별 물리 유효 범위 [min, max]; 밖은 센서 오류로 NaN 처리.
# 공정 컬럼(PAC 주입/유량/레벨/염소주입 제어)은 제외, 수질 측정값만 대상.
# PAC 주입율 SV만 예외적으로 포함(모델 피처라 범위 검증 필요).
PHYS_RANGE = {
    # 원수(착수정)
    "RCS_6.AI.착수정_PH": (5.5, 9.0),
    "RCS_6.AI.착수정_TB": (0.0, 50.0),
    "RCS_6.AI.착수정_AL": (5.0, 100.0),
    "RCS_6.AI.착수정_온도": (0.0, 35.0),
    "RCS_6.AI.착수정_전기전도도": (40.0, 200.0),
    # 침전지
    "RCS_6.AI.침전지1_PH": (5.5, 9.0),
    C.TARGET_TB[1]: (0.0, 3.0),
    "RCS_6.AI.침전지2_PH": (5.5, 9.0),
    C.TARGET_TB[2]: (0.0, 3.0),
    # 여과지
    "RCS_6.AI.여과지_PH": (5.5, 9.0),
    "RCS_6.AI.여과지_탁도": (0.0, 3.0),
    # 정수/정수지
    "RCS_6.AI.정수_수온": (0.0, 35.0),
    "RCS_6.AI.정수지_PH": (5.5, 9.0),
    "RCS_6.AI.정수지_탁도": (0.0, 3.0),
    # 잔류염소
    "RCS_6.AI.침전지1_잔류염소": (0.0, 3.0),
    "RCS_6.AI.침전지2_잔류염소": (0.0, 3.0),
    "RCS_6.AI.정수지_잔류염소": (0.0, 3.0),
    "RCS_3.FCC_19.AI.통합여과수_잔류염소": (0.0, 3.0),
    "주입변경.정수지_잔류염소": (0.0, 3.0),
    # 응집제 주입율 SV(모델 피처)
    C.COL_DOSE: (10.0, 20.0),
}

# 계측 상한(도달 시 포화 플래그)
SAT_LIMIT = {
    "RCS_6.AI.착수정_TB": 50.0, C.TARGET_TB[1]: 3.0, C.TARGET_TB[2]: 3.0,
    "RCS_6.AI.여과지_탁도": 3.0, "RCS_6.AI.정수지_탁도": 3.0,
}

# Hampel 스파이크 필터 (창 분, n_sigma, 절대 임계 하한). floor는 MAD≈0인
# 상수/계단 구간에서 미세 변동을 스파이크로 오인하지 않도록 하는 최소 편차.
# IQR 탐지 경계 계수 (k*IQR). Hampel과 합집합으로 사용.
IQR_K = 3.0

_PH = (11, 5, 0.15)
_TB = (11, 5, 0.3)
_CL = (11, 5, 0.1)
HAMPEL = {
    "RCS_6.AI.착수정_TB": (11, 5, 1.5),
    "RCS_6.AI.착수정_PH": _PH,
    "RCS_6.AI.착수정_AL": (11, 6, 3.0),
    "RCS_6.AI.착수정_전기전도도": (11, 6, 4.0),
    "RCS_6.AI.착수정_온도": (21, 6, 0.5),
    "RCS_6.AI.침전지1_PH": _PH, "RCS_6.AI.침전지2_PH": _PH,
    C.TARGET_TB[1]: _TB, C.TARGET_TB[2]: _TB,
    "RCS_6.AI.여과지_PH": _PH, "RCS_6.AI.여과지_탁도": _TB,
    "RCS_6.AI.정수_수온": (21, 6, 0.5),
    "RCS_6.AI.정수지_PH": _PH, "RCS_6.AI.정수지_탁도": _TB,
    "RCS_6.AI.침전지1_잔류염소": _CL, "RCS_6.AI.침전지2_잔류염소": _CL,
    "RCS_6.AI.정수지_잔류염소": _CL, "RCS_3.FCC_19.AI.통합여과수_잔류염소": _CL,
    "주입변경.정수지_잔류염소": _CL,
}

# 탁도 보정: 칼만 필터로 평활(r_scale↑ → 더 매끄럽게). 상한 포화 스파이크 흡수력이
# 중앙값 평활보다 우수. 침전지(모델 타깃) + 후단 탁도.
KALMAN_SMOOTH = {
    C.TARGET_TB[1]: 5.0, C.TARGET_TB[2]: 5.0,
    "RCS_6.AI.여과지_탁도": 5.0, "RCS_6.AI.정수지_탁도": 5.0,
}

# 후향 규칙(학습데이터 정제 전용): 원수 탁도 급등 후 체류시간 창에서 침전지 1·2가
# 둘 다 무반응이면 센서오류로 판정. 미래 정보를 쓰므로 실시간 예측엔 부적합.
RAW_SPIKE_RESID = 3.0      # 6h 중앙값 대비 잔차 임계 (급등 검출)
RAW_SPIKE_ABS = 5.0        # 절대 탁도 임계
RESIDENCE = ("2h", "5h")   # 침전지 반응 관측 창 (스파이크 시점 기준)
SED_REACT_MIN = 0.1        # 침전지 상승 반응 임계 (NTU); 둘 다 미만이면 무반응
SPIKE_WIDEN_RESID = 1.5    # 마스킹할 스파이크 구간 경계 (잔차 이 값 초과인 연속 구간)


def hampel_mask(s: pd.Series, window: int, n_sigma: float, floor: float) -> pd.Series:
    """롤링 중앙값 대비 |x-med| > max(n_sigma*1.4826*MAD, floor) 인 지점을 이상치로."""
    med = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=max(3, window // 2)).median()
    thresh = (n_sigma * 1.4826 * mad).clip(lower=floor)
    return (s - med).abs() > thresh


def iqr_mask(s: pd.Series, k: float, floor: float, window: str = "6h") -> pd.Series:
    """롤링 IQR 경계 밖 이상치. 경계폭 k*IQR은 floor로 하한을 둬 상수 구간 과탐 억제."""
    q1 = s.rolling(window, center=True, min_periods=30).quantile(0.25)
    q3 = s.rolling(window, center=True, min_periods=30).quantile(0.75)
    band = (k * (q3 - q1)).clip(lower=floor)
    return ((s < q1 - band) | (s > q3 + band)).fillna(False)


def mask_raw_sensor_errors(raw_clean: pd.Series, sed1: pd.Series, sed2: pd.Series):
    """원수 탁도 급등 후 침전지 1·2 둘 다 무반응인 스파이크를 센서오류로 마스킹.

    반환: (마스킹된 원수 시리즈, 제거 플래그, 판정된 이벤트 수, 마스킹 행 수).
    """
    med = raw_clean.rolling("6h", center=True, min_periods=30).median()
    resid = raw_clean - med
    spike = (resid > RAW_SPIKE_RESID) & (raw_clean > RAW_SPIKE_ABS)

    # 30분 내 연속 스파이크를 1개 이벤트로 병합, 대표시점 = 최대 잔차 시점
    sp_times = raw_clean.index[spike.fillna(False)]
    flag = pd.Series(False, index=raw_clean.index)
    n_events = n_sensor = 0
    if len(sp_times) == 0:
        return raw_clean, flag, 0, 0

    groups, cur = [], [sp_times[0]]
    for t in sp_times[1:]:
        if t - cur[-1] <= pd.Timedelta("30min"):
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)

    wide = (resid > SPIKE_WIDEN_RESID).fillna(False)
    for g in groups:
        n_events += 1
        t = resid.loc[g[0]:g[-1]].idxmax()
        b1 = sed1.loc[t - pd.Timedelta("3h"):t].median()
        b2 = sed2.loc[t - pd.Timedelta("3h"):t].median()
        p1 = sed1.loc[t + pd.Timedelta(RESIDENCE[0]):t + pd.Timedelta(RESIDENCE[1])].max()
        p2 = sed2.loc[t + pd.Timedelta(RESIDENCE[0]):t + pd.Timedelta(RESIDENCE[1])].max()
        rise1 = (p1 - b1) if pd.notna(p1) and pd.notna(b1) else 0.0
        rise2 = (p2 - b2) if pd.notna(p2) and pd.notna(b2) else 0.0
        if rise1 < SED_REACT_MIN and rise2 < SED_REACT_MIN:
            # 둘 다 무반응 → 센서오류. 스파이크 구간(잔차>1.5 연속) 마스킹
            seg = wide.loc[t - pd.Timedelta("1h"):t + pd.Timedelta("1h")]
            on = seg[seg].index
            if len(on):
                flag.loc[on.min():on.max()] = True
            else:
                flag.loc[t] = True
            n_sensor += 1

    out = raw_clean.copy()
    out[flag] = np.nan
    out = out.interpolate(method="time", limit=60, limit_area="inside")
    return out, flag, n_events, int(flag.sum())


def main():
    C.setup_output()
    df = pd.read_parquet(INTEGRATED).sort_index()
    n0 = len(df)
    report = []

    for col, (lo, hi) in PHYS_RANGE.items():
        clean = df[col].copy()

        # 1) 물리 범위 밖 → NaN
        phys_bad = (clean < lo) | (clean > hi)
        clean[phys_bad] = np.nan

        # 상한 포화 플래그 (원값 기준)
        if col in SAT_LIMIT:
            df[f"{col}_sat"] = df[col] >= SAT_LIMIT[col]

        # 2) Hampel + IQR 스파이크 탐지 → 합집합 → NaN
        hampel_bad = pd.Series(False, index=df.index)
        iqr_bad = pd.Series(False, index=df.index)
        if col in HAMPEL:
            w, ns, floor = HAMPEL[col]
            hampel_bad = hampel_mask(clean, w, ns, floor).fillna(False)
            iqr_bad = iqr_mask(clean, IQR_K, floor).fillna(False)
            clean[hampel_bad | iqr_bad] = np.nan

        # 3) 선형 보간(짧은 결측만; 60분 초과 공백은 남겨둠)
        clean = clean.interpolate(method="time", limit=60, limit_area="inside")

        # 4) 탁도 칼만 보정(평활)
        if col in KALMAN_SMOOTH:
            clean = kalman_smooth(clean, r_scale=KALMAN_SMOOTH[col])

        df[f"{col}_clean"] = clean
        report.append({
            "변수": col, "물리범위밖": int(phys_bad.sum()),
            "Hampel스파이크": int(hampel_bad.sum()),
            "IQR스파이크": int(iqr_bad.sum()),
            "Hampel∪IQR": int((hampel_bad | iqr_bad).sum()),
            "상한포화": int(df[f"{col}_sat"].sum()) if col in SAT_LIMIT else 0,
            "제거율(%)": round((phys_bad | hampel_bad | iqr_bad).sum() / n0 * 100, 2),
        })

    rep = pd.DataFrame(report)
    print(f"전처리 대상 {n0:,}행\n")
    print(rep.to_string(index=False))
    C.save_csv(rep, C.RESULTS_DIR / "preprocess_report.csv")

    # 후향 규칙: 원수 탁도 센서오류(침전지 무반응 급등) 마스킹
    raw_col = "RCS_6.AI.착수정_TB"
    masked, flag, n_ev, n_masked = mask_raw_sensor_errors(
        df[f"{raw_col}_clean"], df[f"{C.TARGET_TB[1]}_clean"], df[f"{C.TARGET_TB[2]}_clean"]
    )
    df[f"{raw_col}_clean"] = masked
    df[f"{raw_col}_sensor_error"] = flag
    print(
        f"\n[후향 규칙] 원수 탁도 급등 이벤트 {n_ev}건 중 "
        f"침전지 1·2 둘 다 무반응(센서오류) {int(flag.groupby((flag != flag.shift()).cumsum()).first().sum())}건 판정 "
        f"→ {n_masked}행 마스킹"
    )

    df.to_parquet(OUT)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
