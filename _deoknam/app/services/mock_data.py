"""Mock 데이터 테이블·시계열 생성기 (wtp-api-catalog v0.2.5 기준).

정적 테이블(지표 키/한국어명/단위/기준값)은 카탈로그 response_example에서 가져왔고,
시계열은 (plant_id, key) 시드 랜덤워크로 생성한다 — 결정적이면서 현재 시각에 맞춰 움직인다.
플랜트별 차이는 PLANT_VARIANTS 한 곳에서만 분기한다.
"""
from __future__ import annotations

import random
from datetime import timedelta

from app.core.config import settings
from app.services.common import now_kst

PROCESSES = [
    "intake",
    "coagulation",
    "sedimentation",
    "filtration",
    "disinfection",
    "clear_water",
]

PROCESS_LABELS = {
    "intake": "착수",
    "coagulation": "혼화응집",
    "sedimentation": "침전",
    "filtration": "여과",
    "disinfection": "소독",
    "clear_water": "정수",
}

# 플랜트별 차이 — 이 테이블만 plant_id로 분기한다
PLANT_VARIANTS = {
    "deoknam": {
        "basins": ["basin-4", "basin-5"],
        "value_scale": 1.0,
        "intake_location": "착수정",
        "operator": "op_kim",
    },
    "yongyeon": {
        "basins": ["basin-1", "basin-2"],
        "value_scale": 0.92,
        "intake_location": "취수장 착수정",
        "operator": "op_lee",
    },
}


def variant() -> dict:
    return PLANT_VARIANTS.get(settings.PLANT_ID, PLANT_VARIANTS["deoknam"])


# 공정별 모니터링 지표 (key/name/unit/base/target). coagulation은 basin/stage 필드 추가.
PROCESS_INDICATORS: dict[str, list[dict]] = {
    "intake": [
        {"key": "influent_turbidity", "name": "유입수 탁도", "unit": "NTU", "base": 6.9, "target": 6.0, "ndigits": 2},
        {"key": "raw_ph", "name": "원수 pH", "unit": "pH", "base": 7.1, "target": 7.0, "ndigits": 2},
        {"key": "raw_water_temp", "name": "원수 수온", "unit": "℃", "base": 19.2, "target": None, "ndigits": 1},
        {"key": "raw_flow", "name": "취수 유량", "unit": "m³/h", "base": 4200, "target": None, "ndigits": 0},
    ],
    "coagulation": [
        {"key": "g_value", "name": "G값(속도경사)", "unit": "s⁻¹", "base": 720, "target": 740, "ndigits": 0, "stage": 1},
        {"key": "rpm", "name": "교반기 회전수", "unit": "rpm", "base": 142, "target": 145, "ndigits": 0, "stage": 2},
        {"key": "coagulant_dose", "name": "응집제 주입률", "unit": "mg/L", "base": 31.5, "target": 32.0, "ndigits": 1, "stage": 1},
        {"key": "floc_index", "name": "플록 형성 지수", "unit": "-", "base": 0.82, "target": 0.85, "ndigits": 2, "stage": 3},
    ],
    "sedimentation": [
        {"key": "sed_turbidity_1", "name": "1차 침전수 탁도", "unit": "NTU", "base": 1.8, "target": 1.5, "ndigits": 2},
        {"key": "sed_turbidity_2", "name": "2차 침전수 탁도", "unit": "NTU", "base": 1.2, "target": 1.0, "ndigits": 2},
        {"key": "sludge_level", "name": "슬러지 계면 높이", "unit": "m", "base": 1.4, "target": None, "ndigits": 2},
    ],
    "filtration": [
        {"key": "filt_turbidity_1", "name": "1차 여과 유출탁도", "unit": "NTU", "base": 0.08, "target": 0.10, "ndigits": 3},
        {"key": "filt_turbidity_2", "name": "2차 여과 유출탁도", "unit": "NTU", "base": 0.07, "target": 0.10, "ndigits": 3},
        {"key": "filter_head_loss", "name": "여과지 손실수두", "unit": "m", "base": 1.6, "target": None, "ndigits": 2},
    ],
    "disinfection": [
        {"key": "pre_chlorine_dose", "name": "전염소 주입률", "unit": "mg/L", "base": 1.1, "target": 1.2, "ndigits": 2},
        {"key": "post_cl_dose_rate", "name": "후염소 주입률", "unit": "mg/L", "base": 0.28, "target": 0.31, "ndigits": 2},
        {"key": "residual_chlorine_1", "name": "잔류염소(1차)", "unit": "mg/L", "base": 0.21, "target": 0.20, "ndigits": 2},
    ],
    "clear_water": [
        {"key": "clear_turbidity_1", "name": "1차 정수탁도", "unit": "NTU", "base": 0.07, "target": 0.10, "ndigits": 3},
        {"key": "clear_turbidity_2", "name": "2차 정수탁도", "unit": "NTU", "base": 0.06, "target": 0.10, "ndigits": 3},
        {"key": "residual_chlorine_out", "name": "정수지 유출 잔류염소", "unit": "mg/L", "base": 0.22, "target": 0.20, "ndigits": 2},
    ],
}

# 공정별 권고사항 요약 (API-005/011/017/023/028/033)
PROCESS_SUMMARY_RECOS: dict[str, list[dict]] = {
    "intake": [
        {"target": "influent_turbidity", "current_value": 6.9, "recommended_value": 6.2, "unit": "NTU",
         "safe_range": {"min": 4.0, "max": 8.0},
         "expected_effect": "후단 응집제 주입률 안정화", "confidence": 0.84,
         "caution": "강우 유입 시 재확인"},
    ],
    "coagulation": [
        {"basin": None, "stage": 1, "target": "g_value", "current_value": 720, "recommended_value": 745, "unit": "s⁻¹",
         "safe_range": {"min": 700, "max": 780},
         "expected_effect": "플록 형성 지수 0.03 개선", "confidence": 0.86,
         "caution": "수온 급변 시 재확인"},
        {"basin": None, "stage": 2, "target": "rpm", "current_value": 142, "recommended_value": 145, "unit": "rpm",
         "safe_range": {"min": 138, "max": 150},
         "expected_effect": "목표 탁도 달성률 향상", "confidence": 0.86,
         "caution": "수온 급변 시 재확인"},
    ],
    "sedimentation": [
        {"target": "sed_turbidity_1", "current_value": 1.8, "recommended_value": 1.5, "unit": "NTU",
         "safe_range": {"min": 1.0, "max": 2.0},
         "expected_effect": "여과 부하 경감", "confidence": 0.82,
         "caution": "슬러지 인발 주기 확인"},
    ],
    "filtration": [
        {"target": "filt_turbidity_1", "current_value": 0.08, "recommended_value": 0.07, "unit": "NTU",
         "safe_range": {"min": 0.03, "max": 0.10},
         "expected_effect": "역세 주기 최적화", "confidence": 0.83,
         "caution": "손실수두 1.8 m 초과 시 역세 우선"},
    ],
    "disinfection": [
        {"target": "post_cl_dose_rate", "current_value": 0.28, "recommended_value": 0.31, "unit": "mg/L",
         "safe_range": {"min": 0.20, "max": 0.50},
         "expected_effect": "잔류염소 하한 미달 방지", "confidence": 0.88,
         "caution": "수온 반영, THM 생성 주의"},
    ],
    "clear_water": [
        {"target": "residual_chlorine_out", "current_value": 0.22, "recommended_value": 0.21, "unit": "mg/L",
         "safe_range": {"min": 0.20, "max": 0.50},
         "expected_effect": "배급수 잔류염소 균등화", "confidence": 0.85,
         "caution": "정수지 체류시간 변동 확인"},
    ],
}

# 공정별 XAI 판단 근거 (API-008/014/020/025/030/035)
PROCESS_JUDGEMENT: dict[str, dict] = {
    "intake": {
        "interpretation": "원수 탁도가 완만히 상승 중이나 관리기준 이내로 유지되고 있습니다.",
        "operation_judgement": "정상 범위",
        "action_guide": "현 설정 유지, 강우 예보 시 취수량 조정 검토",
        "contributions": [
            {"feature": "influent_turbidity", "weight": 0.42, "direction": "+"},
            {"feature": "raw_flow", "weight": 0.27, "direction": "+"},
            {"feature": "raw_water_temp", "weight": 0.18, "direction": "-"},
        ],
    },
    "coagulation": {
        "interpretation": "G값이 목표 대비 소폭 낮아 플록 형성 지수가 하락 추세입니다.",
        "operation_judgement": "주의 관찰",
        "action_guide": "교반기 회전수 145 rpm 상향 권고 검토",
        "contributions": [
            {"feature": "g_value", "weight": 0.42, "direction": "+"},
            {"feature": "coagulant_dose", "weight": 0.31, "direction": "+"},
            {"feature": "raw_ph", "weight": 0.15, "direction": "-"},
        ],
    },
    "sedimentation": {
        "interpretation": "침전수 탁도가 안정적이며 슬러지 계면도 관리범위 내에 있습니다.",
        "operation_judgement": "정상 범위",
        "action_guide": "현 설정 유지",
        "contributions": [
            {"feature": "sed_turbidity_1", "weight": 0.38, "direction": "+"},
            {"feature": "coagulant_dose", "weight": 0.29, "direction": "-"},
            {"feature": "sludge_level", "weight": 0.21, "direction": "+"},
        ],
    },
    "filtration": {
        "interpretation": "여과 유출탁도는 양호하나 손실수두 상승 속도가 평시보다 빠릅니다.",
        "operation_judgement": "주의 관찰",
        "action_guide": "손실수두 1.8 m 도달 시 역세척 실시",
        "contributions": [
            {"feature": "filter_head_loss", "weight": 0.45, "direction": "+"},
            {"feature": "filt_turbidity_1", "weight": 0.30, "direction": "+"},
            {"feature": "sed_turbidity_2", "weight": 0.14, "direction": "+"},
        ],
    },
    "disinfection": {
        "interpretation": "예측 잔류염소가 하한(0.20 mg/L)에 근접하고 있습니다.",
        "operation_judgement": "권고 적용 필요",
        "action_guide": "후염소 주입률 0.31 mg/L 상향 권고",
        "contributions": [
            {"feature": "post_cl_dose_rate", "weight": 0.48, "direction": "+"},
            {"feature": "raw_water_temp", "weight": 0.26, "direction": "-"},
            {"feature": "chlorine_decay", "weight": 0.16, "direction": "-"},
        ],
    },
    "clear_water": {
        "interpretation": "정수탁도·잔류염소 모두 수질기준을 안정적으로 만족하고 있습니다.",
        "operation_judgement": "정상 범위",
        "action_guide": "현 설정 유지",
        "contributions": [
            {"feature": "clear_turbidity_1", "weight": 0.36, "direction": "+"},
            {"feature": "residual_chlorine_out", "weight": 0.33, "direction": "-"},
            {"feature": "filt_turbidity_1", "weight": 0.19, "direction": "+"},
        ],
    },
}

# 공정별 허용 analysisType 과 기본값 (API-006/007/012/013/018/019/024/029/034)
ANALYSIS_TYPES: dict[str, list[str]] = {
    "intake": ["cluster", "raw_water_recommendation"],
    "coagulation": ["scatter", "mixer_control"],
    "sedimentation": ["efficiency", "coagulant_suitability"],
    "filtration": ["default"],
    "disinfection": ["default"],
    "clear_water": ["default"],
}

_ANALYSIS_CONTENT: dict[tuple[str, str], dict] = {
    ("intake", "cluster"): {
        "metrics": [
            {"key": "cluster_id", "value": 2, "unit": "-"},
            {"key": "influent_turbidity", "value": 6.9, "unit": "NTU"},
            {"key": "cluster_probability", "value": 0.91, "unit": "-"},
        ],
        "chart": {"type": "cluster", "x": "influent_turbidity", "y": "raw_ph",
                  "points": [{"x": 6.8, "y": 7.1, "cluster": 2}, {"x": 5.4, "y": 7.0, "cluster": 1},
                             {"x": 9.2, "y": 6.9, "cluster": 3}]},
        "recommendation": {"value": 2, "basis": "GMM 군집 분석 결과 현재 원수는 군집 C2(평수기 중탁도)에 속함"},
    },
    ("intake", "raw_water_recommendation"): {
        "metrics": [
            {"key": "influent_turbidity", "value": 6.9, "unit": "NTU"},
            {"key": "recommended_intake_flow", "value": 4100, "unit": "m³/h"},
        ],
        "chart": {"type": "line", "series": ["influent_turbidity"],
                  "points": [{"t": "-2h", "v": 6.5}, {"t": "-1h", "v": 6.8}, {"t": "0h", "v": 6.9}]},
        "recommendation": {"value": 4100, "basis": "원수 탁도 상승 추세 반영 취수량 소폭 하향 권고"},
    },
    ("coagulation", "scatter"): {
        "metrics": [
            {"key": "g_value", "value": 720, "unit": "s⁻¹"},
            {"key": "sed_turbidity_corr", "value": -0.63, "unit": "-"},
        ],
        "chart": {"type": "scatter", "x": "g_value", "y": "sed_turbidity_1",
                  "points": [{"x": 700, "y": 2.1}, {"x": 720, "y": 1.8}, {"x": 745, "y": 1.5}]},
        "recommendation": {"value": 745, "basis": "군집/산포 분석 결과 G값 745 s⁻¹ 부근에서 침전탁도 최소"},
    },
    ("coagulation", "mixer_control"): {
        "metrics": [
            {"key": "rpm", "value": 142, "unit": "rpm"},
            {"key": "g_value", "value": 720, "unit": "s⁻¹"},
            {"key": "floc_index", "value": 0.82, "unit": "-"},
        ],
        "chart": {"type": "line", "series": ["rpm", "floc_index"],
                  "points": [{"t": "-2h", "rpm": 140, "floc_index": 0.80},
                             {"t": "-1h", "rpm": 142, "floc_index": 0.81},
                             {"t": "0h", "rpm": 142, "floc_index": 0.82}]},
        "recommendation": {"value": 145, "basis": "교반 강도-플록 형성 관계 분석 결과 145 rpm 권고"},
    },
    ("sedimentation", "efficiency"): {
        "metrics": [
            {"key": "removal_efficiency", "value": 73.9, "unit": "%"},
            {"key": "sed_turbidity_1", "value": 1.8, "unit": "NTU"},
        ],
        "chart": {"type": "bar", "x": "basin", "y": "removal_efficiency",
                  "points": [{"x": "1계열", "y": 73.9}, {"x": "2계열", "y": 76.2}]},
        "recommendation": {"value": 75.0, "basis": "제거효율 목표 75% 대비 1계열 소폭 미달 — 응집 조건 보정 권고"},
    },
    ("sedimentation", "coagulant_suitability"): {
        "metrics": [
            {"key": "coagulant_dose", "value": 31.5, "unit": "mg/L"},
            {"key": "suitability_score", "value": 0.87, "unit": "-"},
        ],
        "chart": {"type": "scatter", "x": "coagulant_dose", "y": "sed_turbidity_1",
                  "points": [{"x": 29.0, "y": 2.2}, {"x": 31.5, "y": 1.8}, {"x": 33.0, "y": 1.7}]},
        "recommendation": {"value": 32.0, "basis": "현재 수질 군집 기준 적정 주입률 32.0 mg/L"},
    },
    ("filtration", "default"): {
        "metrics": [
            {"key": "filt_turbidity_1", "value": 0.08, "unit": "NTU"},
            {"key": "run_time_since_backwash", "value": 42, "unit": "h"},
        ],
        "chart": {"type": "line", "series": ["filter_head_loss"],
                  "points": [{"t": "-12h", "v": 1.3}, {"t": "-6h", "v": 1.5}, {"t": "0h", "v": 1.6}]},
        "recommendation": {"value": 1.8, "basis": "손실수두 추세 분석 결과 약 6시간 후 역세 기준 도달 예상"},
    },
    ("disinfection", "default"): {
        "metrics": [
            {"key": "post_cl_dose_rate", "value": 0.28, "unit": "mg/L"},
            {"key": "residual_chlorine_1", "value": 0.21, "unit": "mg/L"},
        ],
        "chart": {"type": "line", "series": ["residual_chlorine_1"],
                  "points": [{"t": "-2h", "v": 0.23}, {"t": "-1h", "v": 0.22}, {"t": "0h", "v": 0.21}]},
        "recommendation": {"value": 0.31, "basis": "잔류염소 하강 추세 — 후염소 주입률 0.31 mg/L 권고"},
    },
    ("clear_water", "default"): {
        "metrics": [
            {"key": "clear_turbidity_1", "value": 0.07, "unit": "NTU"},
            {"key": "residual_chlorine_out", "value": 0.22, "unit": "mg/L"},
        ],
        "chart": {"type": "line", "series": ["clear_turbidity_1"],
                  "points": [{"t": "-2h", "v": 0.07}, {"t": "-1h", "v": 0.07}, {"t": "0h", "v": 0.07}]},
        "recommendation": {"value": 0.07, "basis": "정수 수질 안정 — 현 운전조건 유지"},
    },
}


def analysis_payload(process: str, analysis_type: str) -> dict:
    content = _ANALYSIS_CONTENT[(process, analysis_type)]
    return {
        "process": process,
        "analysis_type": analysis_type,
        "result": {"metrics": content["metrics"], "chart": content["chart"]},
        "recommendation": content["recommendation"],
    }


# ---------------------------------------------------------------------------
# 시계열 생성기
# ---------------------------------------------------------------------------

_INTERVAL_MINUTES = {"10m": 10, "30m": 30, "1h": 60, "2h": 120, "1d": 1440}


def _rng(key: str) -> random.Random:
    return random.Random(f"{settings.PLANT_ID}:{key}")


def _scaled(base: float) -> float:
    return base * variant()["value_scale"]


def _round(value: float, ndigits: int) -> float:
    return round(value, ndigits) if ndigits > 0 else round(value)


def history_series(indicator: dict, points: int = 24, interval: str = "1h") -> list[dict]:
    """과거 시계열 [{t, v}] — 현재 시각에서 역산한 랜덤워크."""
    minutes = _INTERVAL_MINUTES.get(interval, 60)
    rng = _rng(f"hist:{indicator['key']}")
    base = _scaled(indicator["base"])
    now = now_kst().replace(minute=0, second=0, microsecond=0)
    out = []
    v = base
    for i in range(points, 0, -1):
        v += base * rng.uniform(-0.02, 0.02)
        out.append({
            "t": (now - timedelta(minutes=minutes * i)).isoformat(timespec="seconds"),
            "v": _round(v, indicator["ndigits"]),
        })
    return out


def forecast_series(indicator: dict, points: int = 6, interval: str = "1h",
                    with_bounds: bool = True) -> list[dict]:
    """예측 시계열 [{t, v, lower, upper}]."""
    minutes = _INTERVAL_MINUTES.get(interval, 60)
    rng = _rng(f"fcst:{indicator['key']}")
    base = _scaled(indicator["base"])
    now = now_kst().replace(minute=0, second=0, microsecond=0)
    out = []
    v = base
    for i in range(1, points + 1):
        v += base * rng.uniform(-0.015, 0.025)
        row = {
            "t": (now + timedelta(minutes=minutes * i)).isoformat(timespec="seconds"),
            "v": _round(v, indicator["ndigits"]),
        }
        if with_bounds:
            spread = abs(base) * 0.05 * (1 + 0.3 * i)
            row["lower"] = _round(v - spread, indicator["ndigits"])
            row["upper"] = _round(v + spread, indicator["ndigits"])
        out.append(row)
    return out


def current_value(indicator: dict) -> float:
    rng = _rng(f"cur:{indicator['key']}")
    base = _scaled(indicator["base"])
    return _round(base * (1 + rng.uniform(-0.01, 0.01)), indicator["ndigits"])


def find_indicator(process: str, key: str | None) -> dict | None:
    rows = PROCESS_INDICATORS[process]
    if key is None:
        return rows[0]
    for row in rows:
        if row["key"] == key:
            return row
    return None
