# -*- coding: utf-8 -*-
"""용연 모델 개발 공통 전처리 설정: 경로, 컬럼 그룹, 파라미터.

덕남 파이프라인 이식본. 컬럼은 3개 그룹으로 처리한다 (전체 36개 측정 컬럼
커버를 pipeline에서 assert; 원본의 _flag/_src 보조 컬럼 30개는 로드 시 제외).
  A. SENSOR_COLS   — 연속 수질 센서: 물리범위 → Hampel∪IQR(∪구간제한 ISO) → 칼만 채움
  B. FLOW_COLS     — 유량/약품투입 연속 신호: A와 동일, floor는 런타임 산출
  C. SETPOINT_COLS — 설정치/계단형 신호: 물리범위 밖만 NaN 후 ffill
덕남의 D그룹(적산계)과 전중염소 1:10 분할은 용연 데이터에 해당 없음 → 제거.
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config.py → preprocessing → 00_preprocessing → 용연 모델 개발
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PARQUET = ROOT.parents[1] / "dataset" / "용연_응집제공정_소독공정_통합.parquet"
RAW_PARQUET = ROOT / "dataset" / "raw_data" / "용연_응집제공정_소독공정_통합.parquet"
OUT_PARQUET = ROOT / "dataset" / "preprocessing_data" / "용연_전처리완료.parquet"
OUT_FLAGS = ROOT / "dataset" / "preprocessing_data" / "용연_전처리_채움플래그.parquet"
OUTPUT_DIR = ROOT / "00_preprocessing" / "output"
COMPARE_DIR = OUTPUT_DIR / "compare"

# ── A. 연속 수질 센서 (15) ─────────────────────────────────────────
SENSOR_COLS = [
    "원수 PH",
    "원수 탁도",
    "원수 온도",
    "원수 전기전도도",
    "원수 알카리도",
    "침전지 PH#1",
    "침전지 PH#2",
    "침전지 탁도",
    "여과지 PH",
    "여과지 탁도",
    "정수지 탁도",
    "정수지 PH",
    "정수지 잔류염소",
    "DCS.여과지 잔류염소",
    "RCLT.후처리잔류염소",
]

# ── B. 유량/약품투입 연속 신호 (10) ────────────────────────────────
FLOW_COLS = [
    "RCLT.원수유입유량",
    "착수정 유입유량(㎥/h)",
    "액체 약품 투입량1",
    "액체 약품 투입량2",
    "액체 약품 투입량3",
    "RCLT.염소라인압력",
    "DCS.전염소 투입량",
    "DCS.중염소 투입량",
    "DCS.후염소 투입기 투입량",
    "DCS.기존 분말투입기 투입량",
]

# ── C. 설정치/계단형 (11) ─────────────────────────────────────────
SETPOINT_COLS = [
    "DCS.PAC #1 FLOW",           # dead tag (전 구간 0) — 통과 처리
    "DCS.기존 PAC 주입률 설정 #1",
    "DCS.기존 PAC 주입률 설정 #2",
    "DCS.PAC 투입량 설정 #2",
    "DCS.기존 전염소 주입량 설정(PPM)",
    "DCS.기존 중염소 주입량 설정(PPM)",
    "DCS.기존 후염소 주입량 설정(PPM)",
    "RCLT.투입량PV_1",
    "RCLT.투입량PV_2",
    "RCLT.투입량PV_3",
    "RCLT.투입량PV_4",
]

ACCUM_COLS = []  # 용연에는 적산계 컬럼 없음

# 물리 유효 범위 (lo, hi); None은 미검사. 밖은 센서 오류로 NaN.
PHYS_RANGE = {
    "원수 PH": (5.5, 9.5),
    "원수 탁도": (0.0, 200.0),
    "원수 온도": (0.0, 35.0),
    "원수 전기전도도": (30.0, 250.0),
    "원수 알카리도": (5.0, 100.0),
    "침전지 PH#1": (5.5, 9.0),
    "침전지 PH#2": (5.5, 9.0),
    "침전지 탁도": (0.0, 10.0),
    "여과지 PH": (5.5, 9.0),
    "여과지 탁도": (0.0, 5.0),
    "정수지 탁도": (0.0, 3.0),
    "정수지 PH": (5.5, 9.0),
    "정수지 잔류염소": (0.0, 3.0),
    "DCS.여과지 잔류염소": (0.0, 3.0),
    "RCLT.후처리잔류염소": (0.0, 3.0),
}
# 그 외 B/C 그룹 컬럼은 음수만 배제
DEFAULT_RANGE = (0.0, None)

# Hampel (window 샘플수, n_sigma, floor) — 덕남 검증값 승계
_PH = (11, 5, 0.15)
_TB = (11, 5, 0.3)
_CL = (11, 5, 0.1)
HAMPEL = {
    "원수 PH": _PH,
    "원수 탁도": (11, 5, 1.5),
    "원수 온도": (21, 6, 0.5),
    "원수 전기전도도": (11, 6, 4.0),
    "원수 알카리도": (11, 6, 3.0),
    "침전지 PH#1": _PH,
    "침전지 PH#2": _PH,
    "침전지 탁도": _TB,
    "여과지 PH": _PH,
    "여과지 탁도": _TB,
    "정수지 탁도": _TB,
    "정수지 PH": _PH,
    "정수지 잔류염소": _CL,
    "DCS.여과지 잔류염소": _CL,
    "RCLT.후처리잔류염소": _CL,
}
# B그룹 기본 (window, n_sigma); Hampel floor는 max(FLOW_ABS_MIN, 0.05*비영중앙값).
# 펌프 on/off 이원 신호의 롤링 IQR 오탐 방지를 위해 IQR floor는 비영중앙값 1.0배.
FLOW_HAMPEL = (11, 6)
FLOW_ABS_MIN = 1.0
FLOW_FLOOR_RATIO = 0.05
FLOW_IQR_FLOOR_RATIO = 1.0

# IQR
IQR_K = 3.0
IQR_WINDOW = "6h"
IQR_MIN_PERIODS = 30

# IsolationForest — Hampel∪IQR 탐지 구간의 ±ISO_DILATE//2 샘플 확장 윈도우 안에서만 채택
ISO_PARAMS = dict(contamination=0.005, n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
ISO_DILATE = 61  # ±30분 (1분 데이터)

# 칼만: 탁도 3컬럼은 전구간 평활 대체(r_scale=5.0), 그 외는 NaN 지점만 대체
KALMAN_Q_SCALE = 1e-3
KALMAN_R_DEFAULT = 2.0
KALMAN_SMOOTH_FULL = {
    "침전지 탁도": 5.0,
    "여과지 탁도": 5.0,
    "정수지 탁도": 5.0,
}

# 내부 결측 채움 상한(분, 1분 격자). A(수질센서) 내부 결측 런이 이보다 길면
# 채우지 않고 NaN 유지 — 장기 결측을 상수로 지어내는 것 방지. None이면 무제한(구동작).
MAX_FILL_GAP_MIN = 120

# True면 A(수질센서)만 전처리하고 B/C/D(공정: 유량·약품·설정치·적산)는
# 원시값 그대로 통과(이상치제거·채움 미적용). 공정 신호는 운영 로그라 정제 대상 제외.
WATER_QUALITY_ONLY = True

# 단, 아래 '측정된 유입유량'은 공정이라도 예외로 전처리(연속 센서 신호라 이상치·결측 존재).
# 약품투입량·설정치·적산·염소라인압력은 계속 원시 통과.
PREPROCESS_FLOW_COLS = [
    "RCLT.원수유입유량",
    "착수정 유입유량(㎥/h)",
]
