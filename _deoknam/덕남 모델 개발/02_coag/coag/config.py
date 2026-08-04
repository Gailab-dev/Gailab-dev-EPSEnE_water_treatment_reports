# -*- coding: utf-8 -*-
"""02_coag 응집제 투입모델 설정.

원수성상·수질지표·응집제 주입률로 침전지1·2 탁도(t+체류시간)를 예측한다.
체류시간(lag)은 군집×침전지별 DTW 정렬로 추정하고, 군집별로
{LSTM, GRU, XGBoost, RandomForest}를 학습해 SMAPE<5% & R²≥0.9를
만족하는 최소 피처 모델을 선정한다.
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config.py → coag → 02_coag → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "clustering_data"
OUTPUT_DIR = ROOT / "02_coag" / "output"
MODELS_DIR = OUTPUT_DIR / "models"
IMAGES_DIR = OUTPUT_DIR / "images"

CLUSTERS = [0, 1, 2]
RESAMPLE_RULE = "10min"
GAP_MIN = 20            # 리샘플 후 20분 초과 공백 → 새 세그먼트 (군집 교차 대응)

# 타깃 (예측 대상: t+lag 시점의 1시간 트레일링 이동평균 탁도)
# 10분 단위 원시값은 센서 노이즈 스케일 변동이 커서 예측 대상으로 부적합 —
# 운영 목적(주입률 결정)에 맞는 공정 수준(level)을 타깃으로 정의한다.
TARGET_COLS = ["RCS_6.AI.침전지1_탁도", "RCS_6.AI.침전지2_탁도"]
TARGET_SMOOTH_STEPS = 6  # 1h 트레일링 평균 (10분 × 6, 미래 정보 없음)
BASE_COLS = [f"{t}__sm" for t in TARGET_COLS]  # 델타 기준(현재 평활 수준)

# 후보 피처 (19개): 원수성상5 + 응집제SV + 유량 + AR(현재 침전지 탁도)2 + 파생 10
RAW5 = [
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
]
COL_DOSE = "PAC.AI.PAC_주입율제어_SV"
COL_FLOW = "RCS_1.AI.FT101"
DERIVED = [f"{c}__{s}" for c in RAW5 for s in ("mean", "slope")]
# AR 추세: 최근 30/60분 변화량 (세그먼트 내 계산) — 델타 타깃의 동역학 신호
TREND_SRC = TARGET_COLS + ["RCS_6.AI.착수정_TB"]
TREND = [f"{c}__d{w}" for c in TREND_SRC for w in (3, 6)]  # 3=30분, 6=60분
CAND_FEATURES = RAW5 + [COL_DOSE, COL_FLOW] + TARGET_COLS + DERIVED + TREND  # AR = TARGET_COLS 현재값

# 기존 연구 체류시간(분) — DTW 추정값 비교용
PRIOR_LAGS = {0: 180, 1: 150, 2: 210}

# DTW lag 추정 (10분 스텝 단위)
DTW = dict(
    band_steps=36,      # Sakoe-Chiba 밴드 ±36스텝(±6h)
    lag_min=60,         # 유효 lag 하한(분)
    lag_max=360,        # 유효 lag 상한(분)
    min_seg_steps=72,   # 12h 미만 세그먼트 제외
    trim_frac=0.05,     # 경로 양끝 제외 비율
    min_std_src=0.20,   # 원수탁도 평탄 세그먼트 제외 (원단위 std 하한, NTU)
    min_std_tgt=0.02,   # 침전지탁도 평탄 세그먼트 제외 (원단위 std 하한, NTU)
    min_valid_segs=3,   # 유효 세그먼트 미만이면 PRIOR_LAGS 폴백 (단일 세그먼트 이상치 방지)
)

# 동적 lag 정렬: lag(t) = lag_c × Q_ref / Q(t), 10분 배수 반올림, [lag_min, lag_max] 클립.
# Q_ref = 군집 중앙 유량. 총유량 기반 근사이며 침전지 가동 지수 미반영(데이터 부재).
# 군집별 채택은 val 기준 A/B(XGB 동일조건)로 결정 — c0 평탄 겨울(효과 없음·소폭 악화),
# c1/c2는 개선 (c1 평균SMAPE 6.68→5.95, c2 최소R² 0.797→0.811).
DYN_LAG_Q_COL = "주입변경.정수량_천m3d"
LAG_MODE = {0: "fixed", 1: "dynamic", 2: "dynamic"}

# 침전지2 운휴(장기 무변동) 마스킹
IDLE = dict(roll="6h", std_eps=1e-4)

# 시간순 분할
SPLIT = dict(train=0.64, val=0.16, test=0.20)

# 피처 부분집합 크기 그리드 (top-k)
K_GRID = [4, 6, 8, 10, 12, 16, 25]
K_TOL_SMAPE = 0.5   # val 최고 대비 허용 SMAPE 격차(%p)
K_TOL_R2 = 0.01     # val 최고 대비 허용 R² 격차

# 합격 기준 (타깃 2개 모두, 원단위)
PASS = dict(smape_max=5.0, r2_min=0.90)

# 모델 하이퍼파라미터
SEQ = dict(window=36, hidden=64, layers=2, dropout=0.2, lr=1e-3,
           batch=256, max_epochs=100, patience=10, inner_val_frac=0.1)
XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=8,
           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
           device="cuda", random_state=RANDOM_STATE)
XGB_EARLY = 50
RF = dict(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE)

MODEL_NAMES = ["XGBoost", "RandomForest", "LSTM", "GRU"]

# SHAP 표본 (RF는 깊은 트리 다수라 TreeExplainer 비용이 커 별도 상한)
SHAP_N = dict(tree_sample=2000, rf_sample=300, dl_background=200, dl_sample=500)

# 시각화
CLUSTER_COLORS = {0: "#c44e52", 1: "#55a868", 2: "#8172b2"}
MODEL_COLORS = {"XGBoost": "#4878b0", "RandomForest": "#55a868",
                "LSTM": "#c44e52", "GRU": "#dd8452"}
LINE_COLOR = "#4878b0"
