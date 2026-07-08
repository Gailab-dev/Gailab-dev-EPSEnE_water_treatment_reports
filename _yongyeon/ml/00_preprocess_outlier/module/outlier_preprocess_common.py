"""
정수장 통합 Parquet 데이터 이상치 제거 공통 모듈

처리 흐름
1) 정수장별로 삭제해도 무방한 보조 컬럼을 먼저 제거한다.
2) 남은 숫자형 컬럼 중 이상치 탐지 대상 컬럼을 키워드 기반으로 선별한다.
3) 통계 기반 이상치 탐지(IQR, Z-score, Kalman)를 먼저 수행한다.
4) 통계 기반 이상치가 제거된 데이터에 대해 AI 기반 이상치 탐지(IsolationForest)를 수행한다.
5) 이상치가 탐지된 datetime 행은 전체 제거하고, 정제 데이터와 리포트를 저장한다.

주의
- 기존 결측값과 0값은 이상치 기준 계산과 탐지 대상에서 제외한다.
- 결측값과 0값 자체를 이상치로 판단하지 않는다.
- 사용자가 지정한 삭제 컬럼은 출력 parquet에서 제거된다.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# -----------------------------------------------------------------------------
# 1. 이상치 탐지 기준값 설정
# -----------------------------------------------------------------------------
# 기존보다 완만한 1차 권장안 적용
# - IQR 1.5 -> 3.0: 일반 변동은 보존하고 극단값 위주 탐지
# - Z-score 3.0 -> 4.0: 평균에서 매우 멀리 떨어진 값만 탐지
# - Kalman 3.0 -> 5.0: 시계열 잔차 탐지를 완화하여 고탁도/운영상 변동 보존
# - IsolationForest 0.001 -> 0.0005: 유효값 중 약 0.05%만 AI 이상치로 판단
IQR_MULTIPLIER = 3.0
Z_SCORE_THRESHOLD = 4.0
KALMAN_THRESHOLD = 5.0
MIN_STATISTICAL_METHODS = 2
AI_CONTAMINATION = 0.001
AI_RANDOM_STATE = 42

# 제거율 경고 및 저장 차단 기준
WARNING_REMOVAL_RATE = 20.0
BLOCK_SAVE_REMOVAL_RATE = 30.0


# -----------------------------------------------------------------------------
# 2. 정수장별 사전 삭제 컬럼 정의
# -----------------------------------------------------------------------------
# 용연: 사용자가 지정한 flag/src 컬럼 + DCS.*_flag 전체
YONGYEON_EXACT_DROP_COLUMNS = {
    "원수 PH_flag",
    "원수 PH_src",
    "원수 탁도_flag",
    "원수 탁도_src",
    "원수 온도_flag",
    "원수 온도_src",
    "원수 전기전도도_flag",
    "원수 전기전도도_src",
    "원수 알카리도_flag",
    "원수 알카리도_src",
    "정수지 PH_flag",
    "정수지 PH_src",
    "정수지 탁도_flag",
    "정수지 탁도_src",
    "정수지 잔류염소_flag",
    "정수지 잔류염소_src",
    "침전지 PH#1_flag",
    "침전지 PH#1_src",
}

# 덕남: 사용자가 지정한 적산 컬럼
DEOKNAM_EXACT_DROP_COLUMNS = {
    "PAC.AI.주입_유량1_적산",
    "PAC.AI.주입_유량2_적산",
}


# -----------------------------------------------------------------------------
# 3. 이상치 탐지 대상/제외 대상 키워드
# -----------------------------------------------------------------------------
# 핵심 수질/유량 센서로 볼 수 있는 키워드
CORE_SENSOR_KEYWORDS_UPPER = (
    "PH",
    "TB",
    "FLOW",
    "FT",
    "PPM",
    "AL"
)
CORE_SENSOR_KEYWORDS = (
    "탁도",
    "수온",
    "온도",
    "전기전도도",
    "전도도",
    "알카리",
    "알칼",
    "잔류염소",
    "염소",
    "유량",
    "정수량",
    "압력",
    "투입",
    "주입",
    "약품",
    "PAC",
    "PACS"
)

# 이상치 탐지 대상에서 제외할 성격의 키워드
# 주의: 이 목록은 컬럼 삭제가 아니라 이상치 탐지 대상 제외에만 사용된다.
EXCLUDED_KOREAN_KEYWORDS = (
    "적산",
    "목표",
    "설정",
    "상태",
    "제어",
    "누적",
)
EXCLUDED_ENGLISH_KEYWORDS_UPPER = (
    "TARGET",
    "SETTING",
    "SETPOINT",
    "STATUS",
    "CONTROL",
    "ACCUM",
    "CUMULATIVE",
)

# 약품/투입/주입 계열은 후속 AI에서 입력 또는 타깃으로 쓸 수 있으나,
# 여기서는 센서 이상치 탐지 대상에서는 제외한다.
# NON_CORE_PROCESS_KEYWORDS = (
#     "투입",
#     "주입",
#     "약품",
# )
# NON_CORE_PROCESS_KEYWORDS_UPPER = (
#     "PAC",
#     "PACS",
# )


@dataclass
class DetectionResult:
    """이상치 탐지 결과를 저장하는 자료구조."""

    row_mask: pd.Series
    column_report: pd.DataFrame


@dataclass
class DetectionPlan:
    """이상치 탐지 대상/제외 대상 컬럼 계획을 저장하는 자료구조."""

    target_columns: list
    excluded_columns: list
    excluded_reasons: dict
    kalman_columns: list


# -----------------------------------------------------------------------------
# 4. 기본 유틸리티 함수
# -----------------------------------------------------------------------------
def numeric_columns(df):
    """DataFrame에서 숫자형 컬럼명만 반환한다."""
    return list(df.select_dtypes(include=[np.number]).columns)


def valid_numeric_values(series):
    """
    이상치 기준 계산에 사용할 유효값 위치를 반환한다.

    기준:
    - 숫자로 변환 가능한 값
    - NaN/inf가 아닌 값
    - 0이 아닌 값

    기존 결측값과 0값은 제거하지 않고, 기준 계산에서만 제외한다.
    """
    values = pd.to_numeric(series, errors="coerce")
    array = values.to_numpy(dtype=float, copy=False)
    valid_mask = np.isfinite(array) & (array != 0)
    valid_positions = np.flatnonzero(valid_mask)
    return array, valid_positions


def normalize_plant_name(plant_name):
    """정수장명을 비교하기 쉽게 문자열로 정규화한다."""
    return str(plant_name or "").strip().lower()


# -----------------------------------------------------------------------------
# 5. 사전 삭제 컬럼 처리
# -----------------------------------------------------------------------------
def columns_to_drop_for_plant(df, plant_name):
    """
    정수장별로 출력 데이터에서 제거할 컬럼 목록을 계산한다.

    용연:
    - 사용자가 지정한 flag/src 컬럼
    - DCS로 시작하고 _flag로 끝나는 컬럼 전체

    덕남:
    - 사용자가 지정한 적산 컬럼 2개
    """
    plant = normalize_plant_name(plant_name)
    existing_columns = set(df.columns)
    drop_columns = set()

    if "용연" in plant or "yong" in plant or "yy" == plant:
        drop_columns.update(YONGYEON_EXACT_DROP_COLUMNS & existing_columns)
        drop_columns.update(
            column
            for column in df.columns
            if str(column).startswith("DCS.") and str(column).lower().endswith("_flag")
        )

    if "덕남" in plant or "deok" in plant or "dn" == plant:
        drop_columns.update(DEOKNAM_EXACT_DROP_COLUMNS & existing_columns)

    # 원본 컬럼 순서를 유지하기 위해 df.columns 순서대로 반환한다.
    return [column for column in df.columns if column in drop_columns]


def drop_unneeded_columns(df, plant_name):
    """
    사용자가 지정한 삭제 가능 컬럼을 먼저 제거한다.

    행은 절대 제거하지 않고, 컬럼만 제거한다.
    """
    drop_columns = columns_to_drop_for_plant(df, plant_name)
    if not drop_columns:
        return df.copy(), []
    return df.drop(columns=drop_columns), drop_columns


# -----------------------------------------------------------------------------
# 6. 이상치 탐지 대상 컬럼 계획 수립
# -----------------------------------------------------------------------------
def has_core_sensor_keyword(column):
    """컬럼명에 핵심 수질/유량 센서 키워드가 포함되어 있는지 확인한다."""
    upper_column = str(column).upper()
    return any(keyword in upper_column for keyword in CORE_SENSOR_KEYWORDS_UPPER) or any(
        keyword in str(column) for keyword in CORE_SENSOR_KEYWORDS
    )


def exclusion_reasons_for_column(column):
    """
    특정 숫자형 컬럼을 이상치 탐지 대상에서 제외해야 하는 사유를 반환한다.

    제외 사유가 없으면 실제 이상치 탐지 대상 컬럼으로 사용한다.
    """
    column_text = str(column)
    lower_column = column_text.lower()
    upper_column = column_text.upper()
    reasons = []

    if "_flag" in lower_column:
        reasons.append("_flag 보조 컬럼")
    if "_src" in lower_column:
        reasons.append("_src 출처 컬럼")
    if "SV" in upper_column:
        reasons.append("SV 설정/제어 컬럼")
    if "KG" in upper_column:
        reasons.append("KG 투입/누적 성격 컬럼")

    for keyword in EXCLUDED_KOREAN_KEYWORDS:
        if keyword in column_text:
            reasons.append(f"{keyword} 성격 컬럼")

    for keyword in EXCLUDED_ENGLISH_KEYWORDS_UPPER:
        if keyword in upper_column:
            reasons.append(f"{keyword} 성격 컬럼")

    # if any(keyword in column_text for keyword in NON_CORE_PROCESS_KEYWORDS) or any(
    #     keyword in upper_column for keyword in NON_CORE_PROCESS_KEYWORDS_UPPER
    # ):
    #     reasons.append("핵심 수질/유량 센서가 아닌 약품/투입 성격 컬럼")

    if not has_core_sensor_keyword(column):
        reasons.append("핵심 수질/유량 센서 키워드 없음")

    # 중복 사유 제거 후 반환
    return list(dict.fromkeys(reasons))


def build_detection_plan(df, columns=None):
    """숫자형 컬럼을 이상치 탐지 대상과 제외 대상으로 구분한다."""
    columns = numeric_columns(df) if columns is None else list(columns)
    target_columns = []
    excluded_columns = []
    excluded_reasons = {}

    for column in columns:
        reasons = exclusion_reasons_for_column(column)
        if reasons:
            excluded_columns.append(column)
            excluded_reasons[column] = reasons
        else:
            target_columns.append(column)

    # 현재는 실제 탐지 대상 컬럼 전체에 Kalman filter를 적용한다.
    return DetectionPlan(
        target_columns=target_columns,
        excluded_columns=excluded_columns,
        excluded_reasons=excluded_reasons,
        kalman_columns=list(target_columns),
    )


def empty_mask(df):
    """모든 행을 False로 둔 빈 boolean mask를 생성한다."""
    return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)


# -----------------------------------------------------------------------------
# 7. 통계 기반 이상치 탐지
# -----------------------------------------------------------------------------
def kalman_outlier_positions(values, valid_positions, threshold=KALMAN_THRESHOLD):
    """
    단순 1D Kalman filter로 시계열 잔차 기반 이상치 위치를 반환한다.

    보정 내용:
    - 기존 결측값/0값은 이상치 기준 계산에서 제외한다.
    - 다만 유효값만 이어붙이면 결측/0 구간의 시간 간격이 사라지므로,
      유효값 사이의 원본 위치 간격(gap)을 Kalman 예측 불확실성에 반영한다.
    - residual도 원시 잔차가 아니라 예측 불확실성을 반영한 normalized residual로 계산한다.

    절차:
    1) 유효값 시계열을 구성한다.
    2) 유효값 사이의 원본 위치 간격(gap)을 계산한다.
    3) gap이 클수록 estimate_error를 더 크게 증가시킨다.
    4) residual / sqrt(innovation_variance)를 기준으로 이상치를 판단한다.
    """
    if len(valid_positions) < 3:
        return np.array([], dtype=int)

    observed = values[valid_positions].astype(float)
    finite_observed = observed[np.isfinite(observed)]
    if finite_observed.size < 3:
        return np.array([], dtype=int)

    variance = float(np.var(finite_observed))
    if not np.isfinite(variance) or variance == 0:
        return np.array([], dtype=int)

    # 공정 잡음과 측정 잡음 설정
    # process_variance는 시간 경과에 따른 상태 변화 허용량
    # measurement_variance는 측정값 자체의 변동성
    process_variance = max(variance * 1e-5, 1e-9)
    measurement_variance = max(variance, 1e-9)

    # 유효값 사이의 원본 위치 간격 계산
    # 예: 중간에 결측/0이 10개 있으면 gap이 커져 예측 불확실성을 더 크게 반영한다.
    gaps = np.diff(valid_positions, prepend=valid_positions[0]).astype(float)
    gaps[0] = 1.0
    gaps = np.maximum(gaps, 1.0)

    estimate = observed[0]
    estimate_error = measurement_variance

    # 원시 residual이 아니라 normalized residual을 저장한다.
    residuals = np.zeros(len(observed), dtype=float)

    for idx, value in enumerate(observed):
        # 결측/0 구간을 건너뛴 만큼 예측 불확실성을 증가시킨다.
        estimate_error += process_variance * gaps[idx]

        residual = value - estimate
        innovation_variance = estimate_error + measurement_variance

        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            continue

        # 현재 시점의 예측 불확실성을 반영한 잔차
        residuals[idx] = residual / np.sqrt(innovation_variance)

        # Kalman gain 계산 후 추정값 업데이트
        kalman_gain = estimate_error / innovation_variance
        estimate = estimate + kalman_gain * residual
        estimate_error = (1.0 - kalman_gain) * estimate_error

    center = float(np.median(residuals))
    absolute_deviation = np.abs(residuals - center)
    mad = float(np.median(absolute_deviation))
    scale = mad * 1.4826

    # MAD가 0이면 표준편차로 대체한다.
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.std(residuals))

    if not np.isfinite(scale) or scale == 0:
        return np.array([], dtype=int)

    outlier_offsets = np.flatnonzero(absolute_deviation > threshold * scale)
    return valid_positions[outlier_offsets]


def detect_statistical_outliers(
    df,
    columns=None,
    kalman_columns=None,
    iqr_multiplier=IQR_MULTIPLIER,
    z_threshold=Z_SCORE_THRESHOLD,
    kalman_threshold=KALMAN_THRESHOLD,
    min_methods=MIN_STATISTICAL_METHODS,
):
    """
    IQR, Z-score, Kalman 기반 통계 이상치를 컬럼별로 탐지한다.

    최종 제거 기준:
    - 세 방법 중 min_methods개 이상에서 동시에 이상치로 잡힌 위치만 제거 대상
    - 기본값은 2개 이상 동시 탐지
    """
    columns = numeric_columns(df) if columns is None else list(columns)
    kalman_columns = set(columns if kalman_columns is None else kalman_columns)
    row_mask_array = np.zeros(len(df), dtype=bool)
    report_rows = []

    for column in columns:
        values, valid_positions = valid_numeric_values(df[column])
        valid_values = values[valid_positions]

        iqr_positions = np.array([], dtype=int)
        z_positions = np.array([], dtype=int)
        kalman_positions = np.array([], dtype=int)

        if len(valid_values) >= 2:
            # IQR 기준 이상치 탐지
            q1 = float(np.quantile(valid_values, 0.25))
            q3 = float(np.quantile(valid_values, 0.75))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            iqr_offsets = np.flatnonzero((valid_values < lower) | (valid_values > upper))
            iqr_positions = valid_positions[iqr_offsets]

            # Z-score 기준 이상치 탐지
            std = float(np.std(valid_values))
            if np.isfinite(std) and std > 0:
                mean = float(np.mean(valid_values))
                z_scores = np.abs((valid_values - mean) / std)
                z_offsets = np.flatnonzero(z_scores > z_threshold)
                z_positions = valid_positions[z_offsets]

            # Kalman filter 잔차 기준 이상치 탐지
            if column in kalman_columns:
                kalman_positions = kalman_outlier_positions(
                    values,
                    valid_positions,
                    threshold=kalman_threshold,
                )

        # 각 방법이 같은 위치를 몇 번 이상치로 판단했는지 투표한다.
        method_votes = np.zeros(len(df), dtype=np.uint8)
        for positions in (iqr_positions, z_positions, kalman_positions):
            method_votes[positions] += 1

        union_positions = np.flatnonzero(method_votes > 0)
        removal_positions = np.flatnonzero(method_votes >= min_methods)
        row_mask_array[removal_positions] = True
        report_rows.append(
            {
                "컬럼명": column,
                "통계_기준_유효값_개수": int(len(valid_positions)),
                "IQR_이상치_개수": int(len(iqr_positions)),
                "Z-score_이상치_개수": int(len(z_positions)),
                "Kalman_이상치_개수": int(len(kalman_positions)),
                "통계_이상치_합집합_개수": int(len(union_positions)),
                "통계_제거대상_개수": int(len(removal_positions)),
            }
        )

    report = pd.DataFrame(report_rows).set_index("컬럼명") if report_rows else pd.DataFrame()
    return DetectionResult(pd.Series(row_mask_array, index=df.index), report)


# -----------------------------------------------------------------------------
# 8. AI 기반 이상치 탐지
# -----------------------------------------------------------------------------
def detect_ai_outliers(
    df,
    columns=None,
    contamination=AI_CONTAMINATION,
    random_state=AI_RANDOM_STATE,
):
    """
    IsolationForest 기반 AI 이상치를 컬럼별 1차원 feature로 탐지한다.

    통계 기반 제거 후 남은 데이터에 적용되며,
    contamination 기본값은 0.0005로 전체 유효값 중 약 0.05%만 이상치로 본다.
    """
    columns = numeric_columns(df) if columns is None else list(columns)
    row_mask_array = np.zeros(len(df), dtype=bool)
    report_rows = []

    for column in columns:
        values, valid_positions = valid_numeric_values(df[column])
        valid_values = values[valid_positions]
        ai_positions = np.array([], dtype=int)

        if len(valid_positions) >= 2 and float(np.std(valid_values)) > 0:
            model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1,
            )
            predictions = model.fit_predict(valid_values.reshape(-1, 1))
            ai_offsets = np.flatnonzero(predictions == -1)
            ai_positions = valid_positions[ai_offsets]
            row_mask_array[ai_positions] = True

        report_rows.append(
            {
                "컬럼명": column,
                "AI_기준_유효값_개수": int(len(valid_positions)),
                "AI_IsolationForest_이상치_개수": int(len(ai_positions)),
            }
        )

    report = pd.DataFrame(report_rows).set_index("컬럼명") if report_rows else pd.DataFrame()
    return DetectionResult(pd.Series(row_mask_array, index=df.index), report)


# -----------------------------------------------------------------------------
# 9. 리포트 생성용 데이터 구성
# -----------------------------------------------------------------------------
def full_column_report(plant_name, df, plan, statistical_report, ai_report):
    """컬럼별 탐지 대상 여부와 이상치 개수를 하나의 리포트 DataFrame으로 만든다."""
    rows = []
    target_columns = set(plan.target_columns)
    kalman_columns = set(plan.kalman_columns)

    for column in numeric_columns(df):
        stat_row = statistical_report.loc[column] if column in statistical_report.index else {}
        ai_row = ai_report.loc[column] if column in ai_report.index else {}
        rows.append(
            {
                "정수장": plant_name,
                "컬럼명": column,
                "탐지대상여부": "Y" if column in target_columns else "N",
                "Kalman적용여부": "Y" if column in kalman_columns else "N",
                "제외사유": " | ".join(plan.excluded_reasons.get(column, [])),
                "통계_기준_유효값_개수": int(stat_row.get("통계_기준_유효값_개수", 0)),
                "IQR_이상치_개수": int(stat_row.get("IQR_이상치_개수", 0)),
                "Z-score_이상치_개수": int(stat_row.get("Z-score_이상치_개수", 0)),
                "Kalman_이상치_개수": int(stat_row.get("Kalman_이상치_개수", 0)),
                "통계_이상치_합집합_개수": int(stat_row.get("통계_이상치_합집합_개수", 0)),
                "통계_제거대상_개수": int(stat_row.get("통계_제거대상_개수", 0)),
                "AI_기준_유효값_개수": int(ai_row.get("AI_기준_유효값_개수", 0)),
                "AI_IsolationForest_이상치_개수": int(
                    ai_row.get("AI_IsolationForest_이상치_개수", 0)
                ),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 10. 전체 전처리 파이프라인
# -----------------------------------------------------------------------------
def preprocess_dataframe(df, plant_name):
    """
    단일 DataFrame에 대해 사전 컬럼 삭제, 통계 이상치 제거, AI 이상치 제거를 수행한다.
    """
    original_rows = len(df)
    original_columns = list(df.columns)

    # 1) 삭제해도 무방한 보조 컬럼을 먼저 제거한다.
    working_df, dropped_columns = drop_unneeded_columns(df, plant_name)

    # 2) 남은 컬럼 중 실제 이상치 탐지 대상 컬럼을 선정한다.
    plan = build_detection_plan(working_df)

    # 3) 통계 기반 이상치 탐지 및 행 제거를 수행한다.
    statistical_result = detect_statistical_outliers(
        working_df,
        columns=plan.target_columns,
        kalman_columns=plan.kalman_columns,
    )
    after_statistical = working_df.loc[~statistical_result.row_mask].copy()

    # 4) 통계 기반 제거 후 남은 데이터에 AI 기반 이상치 탐지를 수행한다.
    ai_result = detect_ai_outliers(
        after_statistical,
        columns=plan.target_columns,
        contamination=AI_CONTAMINATION,
        random_state=AI_RANDOM_STATE,
    )
    processed = after_statistical.loc[~ai_result.row_mask].copy()

    # 5) 처리 요약 정보를 계산한다.
    statistical_removed_rows = int(statistical_result.row_mask.sum())
    ai_removed_rows = int(ai_result.row_mask.sum())
    final_rows = len(processed)
    total_removed_rows = original_rows - final_rows
    removal_rate = (total_removed_rows / original_rows * 100.0) if original_rows else 0.0

    summary = {
        "정수장": plant_name,
        "원본 행 수": int(original_rows),
        "통계 기반 제거 행 수": statistical_removed_rows,
        "AI 기반 제거 행 수": ai_removed_rows,
        "전체 제거 행 수": int(total_removed_rows),
        "최종 행 수": int(final_rows),
        "제거율(%)": round(removal_rate, 4),
        "원본 컬럼 수": int(len(original_columns)),
        "사전 삭제 컬럼 수": int(len(dropped_columns)),
        "사전 삭제 후 컬럼 수": int(len(working_df.columns)),
        "사전 삭제 컬럼 목록": list(dropped_columns),
        "숫자형 컬럼 수": int(len(numeric_columns(working_df))),
        "실제 탐지 대상 컬럼 수": int(len(plan.target_columns)),
        "탐지 제외 컬럼 수": int(len(plan.excluded_columns)),
        "Kalman 적용 컬럼 수": int(len(plan.kalman_columns)),
        "실제 탐지 대상 컬럼 목록": list(plan.target_columns),
        "탐지 제외 컬럼 목록": list(plan.excluded_columns),
        "Kalman 적용 컬럼 목록": list(plan.kalman_columns),
        "탐지 제외 사유": dict(plan.excluded_reasons),
    }

    # 6) 컬럼별 상세 리포트를 생성한다.
    report = full_column_report(
        plant_name=plant_name,
        df=working_df,
        plan=plan,
        statistical_report=statistical_result.column_report,
        ai_report=ai_result.column_report,
    )

    return processed, summary, report


# -----------------------------------------------------------------------------
# 11. 제거율 경고 및 저장 제어
# -----------------------------------------------------------------------------
def removal_rate_warning(removal_rate):
    """전체 제거율에 따라 경고 메시지를 반환한다."""
    if removal_rate > BLOCK_SAVE_REMOVAL_RATE:
        return (
            f"경고: 전체 제거율 {removal_rate:.4f}%가 {BLOCK_SAVE_REMOVAL_RATE:.0f}%를 "
            "초과하여 결과 파일을 자동 저장하지 않습니다."
        )
    if removal_rate > WARNING_REMOVAL_RATE:
        return (
            f"경고: 전체 제거율 {removal_rate:.4f}%가 목표 범위 상한 "
            f"{WARNING_REMOVAL_RATE:.0f}%를 초과했습니다."
        )
    return ""


def should_save_outputs(removal_rate):
    """제거율이 저장 차단 기준 이하인지 판단한다."""
    return removal_rate <= BLOCK_SAVE_REMOVAL_RATE


# -----------------------------------------------------------------------------
# 12. 파일 입출력 파이프라인
# -----------------------------------------------------------------------------
def run_outlier_pipeline(input_path, output_path, report_csv_path, report_md_path, plant_name):
    """Parquet 파일을 읽어 이상치 제거 후 Parquet/CSV/Markdown 리포트를 저장한다."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_csv_path = Path(report_csv_path)
    report_md_path = Path(report_md_path)

    df = pd.read_parquet(input_path)
    processed, summary, report = preprocess_dataframe(df, plant_name=plant_name)
    warning = removal_rate_warning(summary["제거율(%)"])
    save_outputs = should_save_outputs(summary["제거율(%)"])
    summary["경고"] = warning
    summary["저장 여부"] = save_outputs

    if warning:
        print(warning)

    if not save_outputs:
        print("경고: 제거율이 30%를 초과하여 기존 결과 파일을 변경하지 않았습니다.")
        return summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_parquet(output_path)
    report.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
    write_markdown_report(
        report_md_path=report_md_path,
        plant_name=plant_name,
        input_path=input_path,
        output_path=output_path,
        report_csv_path=report_csv_path,
        summary=summary,
        report=report,
    )

    return summary


# -----------------------------------------------------------------------------
# 13. Markdown 리포트 작성 함수
# -----------------------------------------------------------------------------
def markdown_table(headers, rows):
    """리스트 데이터를 Markdown 표 문자열로 변환한다."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def list_rows(items):
    """단일 컬럼 Markdown 표를 만들기 위한 행 리스트를 생성한다."""
    if not items:
        return [["없음"]]
    return [[item] for item in items]


def excluded_column_rows(summary):
    """탐지 제외 컬럼과 제외 사유를 Markdown 표 행으로 만든다."""
    if not summary["탐지 제외 컬럼 목록"]:
        return [["없음", ""]]

    rows = []
    reasons = summary["탐지 제외 사유"]
    for column in summary["탐지 제외 컬럼 목록"]:
        rows.append([column, " | ".join(reasons.get(column, []))])
    return rows


def write_markdown_report(
    report_md_path,
    plant_name,
    input_path,
    output_path,
    report_csv_path,
    summary,
    report,
):
    """전처리 결과를 사람이 읽기 쉬운 Markdown 리포트로 저장한다."""
    removal_rows = [
        ["원본 행 수", f"{summary['원본 행 수']:,}"],
        ["통계 기반 제거 행 수", f"{summary['통계 기반 제거 행 수']:,}"],
        ["AI 기반 제거 행 수", f"{summary['AI 기반 제거 행 수']:,}"],
        ["전체 제거 행 수", f"{summary['전체 제거 행 수']:,}"],
        ["최종 행 수", f"{summary['최종 행 수']:,}"],
        ["제거율", f"{summary['제거율(%)']:.4f}%"],
        ["저장 여부", "저장함" if summary.get("저장 여부", True) else "저장하지 않음"],
    ]

    dropped_column_rows = [
        ["원본 컬럼 수", f"{summary['원본 컬럼 수']:,}"],
        ["사전 삭제 컬럼 수", f"{summary['사전 삭제 컬럼 수']:,}"],
        ["사전 삭제 후 컬럼 수", f"{summary['사전 삭제 후 컬럼 수']:,}"],
    ]

    column_summary_rows = [
        ["숫자형 컬럼 수", f"{summary['숫자형 컬럼 수']:,}"],
        ["실제 탐지 대상 컬럼 수", f"{summary['실제 탐지 대상 컬럼 수']:,}"],
        ["탐지 제외 컬럼 수", f"{summary['탐지 제외 컬럼 수']:,}"],
        ["Kalman 적용 컬럼 수", f"{summary['Kalman 적용 컬럼 수']:,}"],
    ]

    column_headers = [
        "컬럼명",
        "탐지대상",
        "Kalman",
        "제외사유",
        "통계 기준 유효값",
        "IQR",
        "Z-score",
        "Kalman 이상치",
        "통계 합집합",
        "통계 제거대상",
        "AI 기준 유효값",
        "AI",
    ]
    column_rows = []
    for _, row in report.iterrows():
        column_rows.append(
            [
                row["컬럼명"],
                row["탐지대상여부"],
                row["Kalman적용여부"],
                row["제외사유"],
                f"{int(row['통계_기준_유효값_개수']):,}",
                f"{int(row['IQR_이상치_개수']):,}",
                f"{int(row['Z-score_이상치_개수']):,}",
                f"{int(row['Kalman_이상치_개수']):,}",
                f"{int(row['통계_이상치_합집합_개수']):,}",
                f"{int(row['통계_제거대상_개수']):,}",
                f"{int(row['AI_기준_유효값_개수']):,}",
                f"{int(row['AI_IsolationForest_이상치_개수']):,}",
            ]
        )

    content = "\n".join(
        [
            f"# {plant_name} 이상치 제거 전처리 리포트",
            "",
            "## 입력 및 출력",
            "",
            f"- 입력 파일: `{input_path}`",
            f"- 이상치 제거 파일: `{output_path}`",
            f"- 컬럼별 CSV 리포트: `{report_csv_path}`",
            "",
            "## 처리 원칙",
            "",
            "- datetime 인덱스는 변경하지 않았습니다.",
            "- 원본 컬럼명은 변경하지 않았고, 사용자가 지정한 삭제 가능 컬럼만 사전에 제거했습니다.",
            "- 기존 결측값과 0값은 값 자체를 변경하지 않았고, 기준 계산과 제거 판단에서 제외했습니다.",
            "- 정수장별 지정 컬럼을 먼저 삭제한 뒤 이상치 탐지 대상을 선정했습니다.",
            "- 이상치 탐지 대상은 핵심 수질/유량 센서 컬럼으로 제한했습니다.",
            f"- IQR 기준은 {IQR_MULTIPLIER} * IQR, Z-score 기준은 {Z_SCORE_THRESHOLD}, Kalman 기준은 {KALMAN_THRESHOLD}입니다.",
            f"- 통계 기반 이상치는 IQR, Z-score, Kalman 중 최소 {MIN_STATISTICAL_METHODS}개 이상 방법에서 동시에 이상치로 잡힌 행만 제거했습니다.",
            f"- 통계 기반 제거 후 IsolationForest를 컬럼별로 적용했고 contamination은 {AI_CONTAMINATION}입니다.",
            "- 보간은 수행하지 않았습니다.",
            "",
            "## 제거 행 수",
            "",
            markdown_table(["항목", "값"], removal_rows),
            "",
            "## 사전 삭제 컬럼 요약",
            "",
            markdown_table(["항목", "값"], dropped_column_rows),
            "",
            "## 사전 삭제 컬럼 목록",
            "",
            markdown_table(["컬럼명"], list_rows(summary["사전 삭제 컬럼 목록"])),
            "",
            "## 탐지 컬럼 요약",
            "",
            markdown_table(["항목", "값"], column_summary_rows),
            "",
            "## 실제 탐지 대상 컬럼 목록",
            "",
            markdown_table(["컬럼명"], list_rows(summary["실제 탐지 대상 컬럼 목록"])),
            "",
            "## 제외 컬럼 목록",
            "",
            markdown_table(["컬럼명", "제외 사유"], excluded_column_rows(summary)),
            "",
            "## 컬럼별 이상치 탐지 개수",
            "",
            markdown_table(column_headers, column_rows),
            "",
        ]
    )

    Path(report_md_path).write_text(content, encoding="utf-8")
