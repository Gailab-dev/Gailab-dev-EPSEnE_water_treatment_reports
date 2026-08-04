# -*- coding: utf-8 -*-
"""덕남 응집제 주입률(SV) 직접예측 — 군집별 5모델 비교 설정 (응집제추천설계문서 1단계).

타깃: PAC_주입율제어_SV (원시 유지값, 8단계 계단, 13~30). 원수성상+파생으로 t 시점 SV 예측.
입력은 t 이전에 알 수 있는 값만(침전/여과/정수지 탁도·미래 SV 금지).
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config → dose_compare → 02_coag → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
LABELED_PARQUET = ROOT / "dataset" / "clustering_data" / "덕남_군집라벨.parquet"
FILL_FLAGS_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리_채움플래그.parquet"
OUTPUT_DIR = ROOT / "02_coag" / "output" / "dose_compare"
IMAGES_DIR = OUTPUT_DIR / "images"

# 타깃·군집. TARGETS = {표시명: (컬럼, 유효최소, 유효최대)}.
# 주입률SV는 test 구간 상수라 퇴화(R² 무의미) — 정직 비교용으로 유지, 목표주입량이 실효 비교.
TARGETS = {
    "목표주입량": ("PAC.AI.PAC_목표주입량", 0.1, 250.0),
    "주입률SV": ("PAC.AI.PAC_주입율제어_SV", 10.0, 40.0),
}
TARGET = TARGETS["목표주입량"][0]   # 기본 타깃(하위호환)
LABEL_COL = "cluster_label"
CLUSTERS = [0, 1, 2]

# 입력 원천 컬럼
RAW5 = [
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
]
FLOW_COL = "RCS_1.AI.FT101"
LEVEL_COLS = ["RCS_1.AI.LT301", "RCS_1.AI.LT302"]
# 소독 설정(변경여부 플래그 생성용)
DISINFECT_COLS = ["주입변경.전중염소_1차_KG", "주입변경.전중염소_2차_KG",
                  "주입변경.후염소_PPM", "주입변경.정수량_천m3d"]
PH_COL = "RCS_6.AI.착수정_PH"
TURBIDITY_COL = "RCS_6.AI.착수정_TB"

# 파생 (분 단위, 1분 격자). shift/diff/rolling 모두 과거만 참조(미래 누설 없음).
LAGS = [5, 15, 60]
ROLL = [5, 15, 60]

# 품질필터 (설계문서 §7)
CORE_REQUIRED = RAW5 + [FLOW_COL]     # 이 원수·유량 결측 행 제외 (타깃은 별도 처리)
TARGET_MIN, TARGET_MAX = 10.0, 40.0   # SV 유효범위 밖 제외

# 시간순 분할 (무작위 금지)
SPLIT = dict(train=0.64, val=0.16, test=0.20)

# 시퀀스 모델: 메모리·시간 현실화를 위해 피처를 10분 리샘플 후 윈도우 구성
# (원 1분 950k행 × 창 = 수십GB → 비현실적. 10분 리샘플로 축소, 창은 분 단위 지정)
SEQ_RESAMPLE = "10min"
SEQ_GAP_MIN = 30                 # 리샘플 후 이 간격 초과 공백은 세그먼트 분리
SEQ_WINDOWS = [120]              # 분. 전체 스윕은 [60, 120, 360]으로 확장 가능

# 모델 하이퍼파라미터
XGB = dict(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8,
           colsample_bytree=0.8, tree_method="hist", random_state=RANDOM_STATE)
RF = dict(n_estimators=200, max_depth=16, max_samples=0.5, n_jobs=-1,
          random_state=RANDOM_STATE)
SEQ = dict(hidden=64, layers=2, dropout=0.2, lr=1e-3, batch=256,
           max_epochs=40, patience=6, inner_val_frac=0.1)

# 실행 토글
MODELS_TABULAR = ["LinearRegression", "XGBoost", "RandomForest"]
MODELS_SEQ = ["GRU", "LSTM"]
FEATURE_SETS = ["AR포함", "AR제외"]   # 직전 SV(AR) 피처 포함/제외 각각 비교

# 시각화
CLUSTER_COLORS = {0: "#c44e52", 1: "#55a868", 2: "#8172b2"}
MODEL_COLORS = {"LinearRegression": "#4878b0", "XGBoost": "#c44e52",
                "RandomForest": "#55a868", "GRU": "#dd8452", "LSTM": "#8172b2",
                "persistence": "#999999"}
