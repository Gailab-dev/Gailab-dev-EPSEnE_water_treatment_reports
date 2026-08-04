# -*- coding: utf-8 -*-
"""혼화·플록형성 G/RPM 추천엔진 설정 — 설계문서(혼화응집공정) 수치의 단일 출처.

추천 기준: 기술진단서 표 2.3-13 수온별 표준 RPM 룩업테이블 선형보간(§10.1).
05_mixing_gvalue의 K0 물리모델은 §10.3 검산·경보 역할로만 사용한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # "덕남 모델 개발"
IN_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리완료.parquet"
FLAG_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리_채움플래그.parquet"
MIXING_G_DIR = ROOT / "05_mixing_gvalue"
OUTPUT_DIR = ROOT / "06_floc_rpm" / "output"
IMAGES_DIR = OUTPUT_DIR / "images"

# ── 컬럼 ──────────────────────────────────────────────────────────────
COL_FLOW = "RCS_1.AI.FT101"
COL_TEMP = "RCS_6.AI.착수정_온도"
COL_TB = "RCS_6.AI.착수정_TB"
COL_PH = "RCS_6.AI.착수정_PH"
COL_AL = "RCS_6.AI.착수정_AL"
COL_EC = "RCS_6.AI.착수정_전기전도도"
COL_PAC_SV = "PAC.AI.PAC_주입율제어_SV"
COL_PACS2_SV = "PAC.AI.PACS2_주입율제어_SV"
COL_SED_TB = {1: "RCS_6.AI.침전지1_탁도", 2: "RCS_6.AI.침전지2_탁도"}
COL_SED_PH = {1: "RCS_6.AI.침전지1_PH", 2: "RCS_6.AI.침전지2_PH"}
# §6.4 상류지연 시간정렬 대상 (수온 포함 — 변화는 느리나 일관성 유지)
ALIGN_COLS = [COL_TEMP, COL_TB, COL_PH, COL_AL, COL_PAC_SV, COL_PACS2_SV]
LOAD_COLS = [COL_FLOW, COL_TEMP, COL_TB, COL_PH, COL_AL, COL_EC,
             COL_PAC_SV, COL_PACS2_SV,
             COL_SED_TB[1], COL_SED_TB[2], COL_SED_PH[1], COL_SED_PH[2]]

# ── 유량 (§6.1) ──────────────────────────────────────────────────────
FLOW_UNIT = "m3h"              # FT101 가정 단위 ("m3d"면 /24). SCADA 정의서로 확정 필요
FLOW_UNIT_CONFIRMED = False    # 확정 전 False → control_available 전 행 False (§6.1)
FLOW_AVG_M3H = 9_488.667       # 평균유량 227,728㎥/d
FLOW_PLAUSIBLE_M3H = (1_000.0, 20_000.0)   # 단위오인 감지 밴드 (행 단위 HOLD)
FLOW_ZERO_EPS = 1.0            # ㎥/h — 이하면 유량 0 취급

# ── 플록형성지 제원 (§3.2) ───────────────────────────────────────────
BASIN_VOL_M3 = 776.0
N_STAGES = 3
N_BASINS_TOTAL = 12
N_ACTIVE_ASSUMED = 8           # 가동지수 태그 없음 → 진단(2024-08-27) 당시 8지 가정
# §6.2 실측 유량비 (2024-08-27, 가동 8지) — 가동구성 동일 가정 하 초기값
FLOW_WEIGHTS_8 = {"#1-4": 0.103, "#1-5": 0.128, "#1-6": 0.120, "#2-1": 0.144,
                  "#2-2": 0.115, "#2-3": 0.113, "#2-4": 0.158, "#2-5": 0.118}

# ── 착수정 낙차부 수류혼화 (§8) ──────────────────────────────────────
HYD_HEAD_M = 0.45
T_DROP_AVG_S = 99.1            # 평균유량 기준 혼화시간
G_DROP_REF = 209.9             # 20℃·평균유량 검증 기준값
G_DROP_RANGE = (169.5, 282.3)  # 운전범위 감시 기준

# ── 상류지연 (§6.4) ──────────────────────────────────────────────────
T_UPSTREAM_AVG_MIN = 11.3      # 낙차부→플록형성지, 평균유량 기준
UPSTREAM_LAG_CLIP_MIN = (5.0, 45.0)   # 유량 이상 시 지연 폭주 방지

# ── 목표 G·Gt (§9) ───────────────────────────────────────────────────
G_BASE = (50.0, 30.0, 10.0)
G_CANDIDATES = {"표준": (50.0, 30.0, 10.0),
                "완화": (45.0, 27.0, 9.0),
                "강화": (50.0, 35.0, 12.0)}
GT_RANGE = (3.0e4, 2.0e5)
GT_CENTER = 1.1e5

# ── 수온별 표준 RPM (기술진단서 표 2.3-13, §10.1) ────────────────────
STD_RPM_TEMPS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
STD_RPM_TABLE = {1: (2.05, 1.95, 1.86, 1.79, 1.72, 1.66),
                 2: (1.46, 1.39, 1.32, 1.27, 1.22, 1.18),
                 3: (0.70, 0.67, 0.64, 0.61, 0.59, 0.57)}
TEMP_VALID_C = (0.0, 35.0)     # 벗어나면 HOLD (§7)
TEMP_STD_C = (5.0, 30.0)       # 벗어나면 경계값 클램프 + WARN_TEMP_CLAMP

# ── RPM 한계 (§10.4 — 제작사 값 확정 전 계산범위) ────────────────────
RPM_LIMITS = {1: (1.0, 5.0), 2: (1.0, 5.0), 3: (0.5, 5.0)}
STAGE3_ALARM_RPM = 1.0         # 3단 1rpm 일반기준은 경보용 — 임의 상향 금지

# ── 유효성 (§4.3) ────────────────────────────────────────────────────
MISS_RUN_MIN = 5               # 5분 이상 연속 결측 → HOLD

# ── 검산 경보 ────────────────────────────────────────────────────────
MODEL_DIVERGENCE_WARN = 0.03   # 룩업보간 vs K0 모델 상대편차 경보 문턱

# ── 제어주기 파라미터 (§11.2 — 현 범위에선 표시·이력용 기록) ─────────
DISPLAY_MEDIAN_MIN = 5
RPM_HOLD_MIN = 30
RPM_STEP_MAX = 0.05

# ── §12 침전수 탁도 정렬 ─────────────────────────────────────────────
LAG_CENTERS_MIN = {"첨두응답": 110, "추적자평균": 322, "추적자이론": 348, "설계표": 470}
LAG_DEFAULT = "추적자평균"     # 실측 추적자 기반 — 단락류·사수역 반영 (기본 중심)
LAG_HALF_WIN_MIN = (30, 45, 60)
LAG_HALF_WIN_DEFAULT = 45
LAG_SEARCH_RANGE_MIN = (80, 530)
SMAPE_EPS_NTU = 0.01           # 저탁도 안정화값 (계측 분해능 확인 후 고정)
