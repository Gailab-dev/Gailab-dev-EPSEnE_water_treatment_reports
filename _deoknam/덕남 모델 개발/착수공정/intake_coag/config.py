# -*- coding: utf-8 -*-
"""착수공정 설계문서 1단계 설정 — 수리상수(§4.2)·정렬 그리드(§5)·평가(§7)·피처(§6).

설계문서: 착수공정/설계문서.md. 02_coag(dose_compare·coag)는 무수정 import/이식만.
실제 사용 응집제·실투입량(active_coagulant 등, §3.3)은 미확보 — 주입 관련 입력은
전부 proxy(주입_유량N, 주입량_AN+BN, SV)이며 전 산출물에 proxy로 표기한다.
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config → intake_coag → 착수공정 → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
DATA_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리완료.parquet"
FLAGS_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리_채움플래그.parquet"
COAG_DIR = ROOT / "02_coag"                      # dose_compare·coag import 용
OUTPUT_DIR = ROOT / "착수공정" / "output"
MODELS_DIR = OUTPUT_DIR / "models"
IMAGES_DIR = OUTPUT_DIR / "images"

# ── 컬럼 (§3) ──────────────────────────────────────────────────────────
RAW5 = [
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
]
FLOW_COL = "RCS_1.AI.FT101"                      # 총유량(m³/h)
LEVEL_COLS = ["RCS_1.AI.LT301", "RCS_1.AI.LT302"]
SED_TB = ["RCS_6.AI.침전지1_탁도", "RCS_6.AI.침전지2_탁도"]   # 공정반응 출력(§3.2)
SED_PH = ["RCS_6.AI.침전지1_PH", "RCS_6.AI.침전지2_PH"]
TURBIDITY_COL = RAW5[0]
PH_COL = RAW5[1]
TEMP_COL = RAW5[3]

# 주입률 proxy (§3.3 — 실투입 라벨 미확보. 계열 1/2 = 정수 계열, A/B = 교대 펌프 → 합산)
DOSE_FLOW = ["PAC.AI.주입_유량1", "PAC.AI.주입_유량2"]
DOSE_PUMPS = {"주입량합1": ["PAC.AI.주입량_A1", "PAC.AI.주입량_B1"],
              "주입량합2": ["PAC.AI.주입량_A2", "PAC.AI.주입량_B2"]}
DOSE_SV = ["PAC.AI.PAC_주입율제어_SV", "PAC.AI.PACS2_주입율제어_SV"]
DOSE_TARGET_SP = ["PAC.AI.PAC_목표주입량", "PAC.AI.PACS2_목표주입량"]
AR_SV_COL = DOSE_SV[0]                            # AR(직전 주입률·경과시간·변경폭) 기준

DISINFECT_COLS = ["주입변경.전염소_1차_KG", "주입변경.중염소_1차_KG",
                  "주입변경.전염소_2차_KG", "주입변경.중염소_2차_KG",
                  "주입변경.후염소_PPM", "주입변경.정수량_천m3d"]

# ── 수리상수 (§4.2 — 기술진단서) ───────────────────────────────────────
Q_REF = 227_728.0 / 24.0            # 9,488.667 m³/h (평균유량)
T_UPSTREAM_REF = 11.3               # 주입점→플록형성지 유입부 평균유량 기준(분)
V_FLOC = 776.0                      # 플록형성지 지별 유효용량(m³)
K_FLOC = 1.06                       # 추적자 t/T (플록형성지)
V_SED = 5414.0                      # 침전지 지별 유효용량(m³)
K_SED = 0.91                        # 추적자 t/T (침전지)
PEAK_FLOC = (30.0, 1125.0)          # (첨두시간 분, 시험유량 m³/h)
PEAK_SED = (80.0, 1058.0)
T_SENSOR = 0.0                      # 채수배관·분석기 지연(분) — 미확인, 확인 시 전 후보 가산
N_ACTIVE_DEFAULT = 8                # 2024-08-27 실측 가동지수
# 가동지별 실측 유량(m³/h) — 유량 우선순위 4순위(진단서 유량비) 대체값
BASIN_FLOWS = {"#1-4": 959, "#1-5": 1189, "#1-6": 1116, "#2-1": 1335,
               "#2-2": 1070, "#2-3": 1051, "#2-4": 1461, "#2-5": 1091}

# ── 정렬 그리드 (§4.4·§5.2) ────────────────────────────────────────────
# 정적 후보(분, 1분 반올림): 121.3/333.66/358.8/481.2 → 121/334/359/481
CENTERS_STATIC = {"c121": 121, "c334": 334, "c359": 359, "c481": 481}
CENTERS_DYNAMIC = ["dynmean", "dynpeak"]        # 유량보정형 T_mean(t)/T_peak(t)
DELTAS = [30, 45, 60]                           # 시간창 ±Δ(분)
MODES = ["pt", "med", "max"]                    # 단일점/시간창 median/위험 max
T_CLIP = (90, 540)                              # §4.5 물리 탐색범위(분)
WIN_MIN_FRAC = 0.5                              # 시간창 통계 최소 유효비율

# ── 평가 (§7.3·7.4) ────────────────────────────────────────────────────
# sMAPE ε = 탁도센서 최소 유효 분해능(계측기 사양 확정 전 기본 0.01 NTU).
# 전 모델 동일 고정 — 모델별 조정·저탁도 행 제외로 sMAPE를 낮추는 행위 금지(§7.3).
SMAPE_EPS = 0.01
GATE_SMAPE = 5.0                                # 승인 게이트(%)
SPLIT = dict(train=0.64, val=0.16, test=0.20)   # 시간순 분할
MIN_ROWS = 50

# ── 피처 (§6) ──────────────────────────────────────────────────────────
LAGS = [5, 15, 30, 60]                          # 지연값(분)
DIFFS = [5, 15, 60]                             # 변화량·변화율(분)
ROLLS = [15, 60, 180]                           # 이동통계(분): mean/min/max/std
TB_BINS = [-1.0, 1.0, 2.0, 5.0, 1e9]            # 저/중저/중고/고탁도 구간
TB_BIN_LABELS = ["≤1", "1-2", "2-5", ">5"]
SURGE_D60_NTU = 0.5                             # 급상승: 착수정_TB 60분 상승폭 초과(NTU)

# ── 학습 테이블 ────────────────────────────────────────────────────────
STRIDE_MIN = 10                                 # 학습행 스트라이드(분) — coag 관례
GAP_MIN = 10                                    # 세그먼트 분리 공백(분, 스트라이드 격자)
RAPID_Q_REL = 0.30                              # 체류창 내 유량 상대변동 초과 시 제외(§5.3)
IDLE = dict(roll="6h", std_eps=1e-4)            # 침전지2 운휴(장기 무변동) 마스킹
TARGET_SMOOTH_MIN = 60                          # persistence 기준: 1h 트레일링 평활

# ── 모델풀 (§7.2) — HP는 dose_compare.config 값 그대로(직접 import) ────
MODELS_TABULAR = ["LinearRegression", "XGBoost", "RandomForest"]
MODELS_SEQ = ["GRU", "LSTM"]
SEQ_WINDOWS = [60, 120, 360]                    # DL 입력창(분)
FEATURE_SETS = ["AR포함", "AR제외"]              # AR = 현재·과거 침전지 탁도 피처(과제B)
VIF_MAX = 10.0                                  # 다변수회귀 피처 축약 기준

# ── 스크리닝 (run_05) ─────────────────────────────────────────────────
SCREEN_MODEL = "XGBoost"
TOP_K_ALIGN = 3                                 # 침전지별 상위 정렬조합 수

# 상관분석(run_04) 대표 정렬 — run_02 지지 중심 확인 후 필요 시 변경
REP_ALIGN = ("c334", "med45")

# 시차검증(run_02) — 90~540분, 10분 스텝
LAGV = dict(step_min=10, lag_lo=90, lag_hi=540, band_steps=54,
            event_z=2.5, pre_h=6, post_h=18, min_events=10,
            min_std_src=0.15, trim_frac=0.1)

MODEL_COLORS = {"LinearRegression": "#4878b0", "XGBoost": "#c44e52",
                "RandomForest": "#55a868", "GRU": "#dd8452", "LSTM": "#8172b2",
                "persistence": "#999999"}
