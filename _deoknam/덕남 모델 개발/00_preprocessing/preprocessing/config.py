# -*- coding: utf-8 -*-
"""덕남 모델 개발 공통 전처리 설정: 경로, 컬럼 그룹, 파라미터.

컬럼은 4개 그룹으로 나눠 처리한다 (전체 40컬럼 커버를 pipeline에서 assert).
  A. SENSOR_COLS   — 연속 수질 센서: 물리범위 → Hampel∪IQR(∪구간제한 ISO) → 칼만 채움
  B. FLOW_COLS     — 유량/수위/주입량 연속 신호: A와 동일, floor는 런타임 산출
  C. SETPOINT_COLS — 설정치/수동 주입변경 계단형 신호: 롤링 탐지기가 정상 스텝을
                     오탐하고 칼만이 엣지를 뭉개므로 Hampel/IQR/ISO/칼만 미적용,
                     물리범위 밖만 NaN 후 ffill (참고 프로젝트는 주입변경.정수지_잔류염소에
                     Hampel을 적용했으나 여기서는 계단형 설정치로 보고 제외)
  D. ACCUM_COLS    — 적산계(단조증가): 롤링 탐지 부적합, diff 글리치 규칙만 적용
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config.py → preprocessing → 00_preprocessing → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PARQUET = ROOT.parents[1] / "dataset" / "덕남_응집제공정_소독공정_통합.parquet"
RAW_PARQUET = ROOT / "dataset" / "raw_data" / "덕남_응집제공정_소독공정_통합.parquet"
OUT_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리완료.parquet"
OUT_FLAGS = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리_채움플래그.parquet"
OUTPUT_DIR = ROOT / "00_preprocessing" / "output"
COMPARE_DIR = OUTPUT_DIR / "compare"

# ── A. 연속 수질 센서 (16) ─────────────────────────────────────────────
SENSOR_COLS = [
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
    "RCS_6.AI.침전지1_PH",
    "RCS_6.AI.침전지1_탁도",
    "RCS_6.AI.침전지2_PH",
    "RCS_6.AI.침전지2_탁도",
    "RCS_6.AI.여과지_PH",
    "RCS_6.AI.여과지_탁도",
    "RCS_6.AI.정수_수온",
    "RCS_6.AI.정수지_PH",
    "RCS_6.AI.정수지_탁도",
    "RCS_6.AI.침전지1_잔류염소",
    "RCS_6.AI.침전지2_잔류염소",
    "RCS_6.AI.정수지_잔류염소",
    "RCS_3.FCC_19.AI.통합여과수_잔류염소",
]

# ── B. 유량/수위/주입량 연속 신호 (9) ──────────────────────────────────
FLOW_COLS = [
    "PAC.AI.주입_유량1",
    "PAC.AI.주입_유량2",
    "PAC.AI.주입량_A1",
    "PAC.AI.주입량_B1",
    "PAC.AI.주입량_A2",
    "PAC.AI.주입량_B2",
    "RCS_1.AI.FT101",
    "RCS_1.AI.LT301",
    "RCS_1.AI.LT302",
]

# ── C. 설정치/계단형 (11) ─────────────────────────────────────────────
SETPOINT_COLS = [
    "PAC.AI.PAC_목표주입량",
    "PAC.AI.PACS2_목표주입량",
    "PAC.AI.PAC_주입율제어_SV",
    "PAC.AI.PAC_주입량제어_SV",
    "PAC.AI.PACS2_주입율제어_SV",
    "PAC.AI.PACS2_주입량제어_SV",
    "주입변경.전중염소_1차_KG",
    "주입변경.전중염소_2차_KG",
    "주입변경.후염소_PPM",
    "주입변경.정수량_천m3d",
    "주입변경.정수지_잔류염소",
]

# ── D. 적산계 (2) ─────────────────────────────────────────────────────
ACCUM_COLS = [
    "PAC.AI.주입_유량1_적산",
    "PAC.AI.주입_유량2_적산",
]

# 물리 유효 범위 (lo, hi); None은 미검사. 밖은 센서 오류로 NaN.
PHYS_RANGE = {
    "RCS_6.AI.착수정_PH": (5.5, 9.0),
    "RCS_6.AI.착수정_TB": (0.0, 50.0),
    "RCS_6.AI.착수정_AL": (5.0, 100.0),
    "RCS_6.AI.착수정_온도": (0.0, 35.0),
    "RCS_6.AI.착수정_전기전도도": (40.0, 200.0),
    "RCS_6.AI.침전지1_PH": (5.5, 9.0),
    "RCS_6.AI.침전지1_탁도": (0.0, 3.0),
    "RCS_6.AI.침전지2_PH": (5.5, 9.0),
    "RCS_6.AI.침전지2_탁도": (0.0, 3.0),
    "RCS_6.AI.여과지_PH": (5.5, 9.0),
    "RCS_6.AI.여과지_탁도": (0.0, 3.0),
    "RCS_6.AI.정수_수온": (0.0, 35.0),
    "RCS_6.AI.정수지_PH": (5.5, 9.0),
    "RCS_6.AI.정수지_탁도": (0.0, 3.0),
    "RCS_6.AI.침전지1_잔류염소": (0.0, 3.0),
    "RCS_6.AI.침전지2_잔류염소": (0.0, 3.0),
    "RCS_6.AI.정수지_잔류염소": (0.0, 3.0),
    "RCS_3.FCC_19.AI.통합여과수_잔류염소": (0.0, 3.0),
    "주입변경.정수지_잔류염소": (0.0, 3.0),
    "PAC.AI.PAC_주입율제어_SV": (10.0, 20.0),
    "PAC.AI.PACS2_주입율제어_SV": (10.0, 20.0),
}
# 그 외 B/C/D 그룹 컬럼은 음수만 배제
DEFAULT_RANGE = (0.0, None)

# Hampel (window 샘플수, n_sigma, floor). floor는 MAD≈0인 상수/계단 구간
# 과탐 억제용 최소 편차이며 IQR band 하한으로도 공용.
_PH = (11, 5, 0.15)
_TB = (11, 5, 0.3)
_CL = (11, 5, 0.1)
HAMPEL = {
    "RCS_6.AI.착수정_TB": (11, 5, 1.5),
    "RCS_6.AI.착수정_PH": _PH,
    "RCS_6.AI.착수정_AL": (11, 6, 3.0),
    "RCS_6.AI.착수정_전기전도도": (11, 6, 4.0),
    "RCS_6.AI.착수정_온도": (21, 6, 0.5),
    "RCS_6.AI.침전지1_PH": _PH,
    "RCS_6.AI.침전지2_PH": _PH,
    "RCS_6.AI.침전지1_탁도": _TB,
    "RCS_6.AI.침전지2_탁도": _TB,
    "RCS_6.AI.여과지_PH": _PH,
    "RCS_6.AI.여과지_탁도": _TB,
    "RCS_6.AI.정수_수온": (21, 6, 0.5),
    "RCS_6.AI.정수지_PH": _PH,
    "RCS_6.AI.정수지_탁도": _TB,
    "RCS_6.AI.침전지1_잔류염소": _CL,
    "RCS_6.AI.침전지2_잔류염소": _CL,
    "RCS_6.AI.정수지_잔류염소": _CL,
    "RCS_3.FCC_19.AI.통합여과수_잔류염소": _CL,
}
# B그룹 기본 (window, n_sigma); Hampel floor는 max(FLOW_ABS_MIN, 0.05*비영중앙값).
# 펌프 on/off 이원 신호에서 롤링 IQR이 정지 구간(q1=q3=0) 대비 정상 가동값을
# 오탐하므로 IQR floor는 비영중앙값 전체(1.0배)로 상향 — 큰 스파이크만 잡음.
FLOW_HAMPEL = (11, 6)
FLOW_ABS_MIN = 1.0
FLOW_FLOOR_RATIO = 0.05
FLOW_IQR_FLOOR_RATIO = 1.0

# IQR
IQR_K = 3.0
IQR_WINDOW = "6h"
IQR_MIN_PERIODS = 30

# IsolationForest — 고정 contamination이 정상 구간을 오탐하므로
# Hampel∪IQR 탐지 구간의 ±ISO_DILATE//2 샘플 확장 윈도우 안에서만 채택.
ISO_PARAMS = dict(contamination=0.005, n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
ISO_DILATE = 61  # ±30분 (1분 데이터)

# 칼만: 이상치 제거로 생긴 NaN을 예측-전파로 채움. 탁도 4컬럼은 참고 프로젝트와
# 동일하게 전구간 평활 대체(r_scale=5.0), 그 외는 NaN 지점만 대체(r_scale=2.0).
KALMAN_Q_SCALE = 1e-3
KALMAN_R_DEFAULT = 2.0
KALMAN_SMOOTH_FULL = {
    "RCS_6.AI.침전지1_탁도": 5.0,
    "RCS_6.AI.침전지2_탁도": 5.0,
    "RCS_6.AI.여과지_탁도": 5.0,
    "RCS_6.AI.정수지_탁도": 5.0,
}

# 내부 결측 채움 상한(분, 1분 격자). A(수질센서) 내부 결측 런이 이보다 길면
# 채우지 않고 NaN 유지 — 장기 결측을 상수로 지어내는 것 방지. None이면 무제한(구동작).
MAX_FILL_GAP_MIN = 120

# True면 A(수질센서)만 전처리하고 B/C/D(공정: 유량·약품·설정치·적산)는
# 원시값 그대로 통과(이상치제거·채움 미적용). 공정 신호는 운영 로그라 정제 대상 제외.
# (전중염소 1:10 분할은 구조적 파생이라 원시값 기준으로 그대로 수행)
WATER_QUALITY_ONLY = True

# 단, 아래 '측정된 유입유량'은 공정이라도 예외로 전처리(연속 센서 신호라 이상치·결측 존재).
# FT101=착수 유입유량. LT301/302(수위)·약품투입·설정치·적산은 계속 원시 통과.
PREPROCESS_FLOW_COLS = [
    "RCS_1.AI.FT101",
]

# 적산계 글리치: diff<0 이후 ACCUM_GLITCH_SPAN 샘플 내 원복되면 텔레메트리 오류
ACCUM_GLITCH_SPAN = 3

# 전중염소 → 전염소:중염소 = 1:10 분할 (전처리 완료된 부모 컬럼 기준)
PRE_RATIO = 1.0 / 11.0
MID_RATIO = 10.0 / 11.0
CHLORINE_SPLIT = {
    "주입변경.전중염소_1차_KG": ("주입변경.전염소_1차_KG", "주입변경.중염소_1차_KG"),
    "주입변경.전중염소_2차_KG": ("주입변경.전염소_2차_KG", "주입변경.중염소_2차_KG"),
}
