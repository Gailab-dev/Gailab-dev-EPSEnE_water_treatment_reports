# -*- coding: utf-8 -*-
"""응집제 투입량 추천 설정.

방식: 02_coag 데이터 파이프라인을 재사용하되, 주입률(PAC SV)을 강제 포함하고
"주입률↑ → 침전 탁도↓" 단조 제약을 건 XGBoost(침전지별 단일출력 2개)를 재학습.
추천 = 주입률 후보 그리드 what-if 스윕 → 목표 탁도를 만족하는 최소 주입률.
"""
import sys
from pathlib import Path

RANDOM_STATE = 42

# 경로: config.py → recommend → 03_dose_recommend → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
COAG_DIR = ROOT / "02_coag"
if str(COAG_DIR) not in sys.path:
    sys.path.insert(0, str(COAG_DIR))

OUTPUT_DIR = ROOT / "03_dose_recommend" / "output"
MODELS_DIR = OUTPUT_DIR / "models"
IMAGES_DIR = OUTPUT_DIR / "images"

# 02_coag 설정 재사용 (컬럼명·군집·lag 모드)
from coag import config as CC  # noqa: E402

CLUSTERS = CC.CLUSTERS
TARGET_COLS = CC.TARGET_COLS
COL_DOSE = CC.COL_DOSE

# 추천용 모델 피처 (주입률 강제 포함): 원수성상5 + 주입률 + 유량 + AR2 + 추세6 = 15
DOSE_FEATURES = (CC.RAW5 + [CC.COL_DOSE, CC.COL_FLOW]
                 + CC.TARGET_COLS + CC.TREND)

# 단조 제약: 주입률만 감소(-1), 나머지 0
MONOTONE = tuple(-1 if f == CC.COL_DOSE else 0 for f in DOSE_FEATURES)

XGB_DOSE = dict(n_estimators=600, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                device="cuda", random_state=RANDOM_STATE)

# 주입률 후보 그리드 (조견표 운영범위 12~24ppm)
DOSE_GRID_MIN = 12.0
DOSE_GRID_MAX = 24.0
DOSE_GRID_STEP = 0.5

# 데이터 지지 구간 제한: 학습 주입률 분포가 15~16ppm에 밀집(std 0.5ppm)되어
# 그 밖의 주입률 효과는 데이터로 식별 불가(외삽) — 추천 탐색을 학습 분포
# P5~P95 구간으로 제한한다. 구간 확장은 Jar-test/의도적 변경 실험으로 보정 필요.
DOSE_SUPPORT_Q = (0.05, 0.95)

# 목표 침전 탁도 (침전지1, 침전지2) 및 안전마진 (NTU)
# 운영 실적 95% 누적 0.42NTU(기술진단) 참고, 목표 0.5NTU 관리
TARGET_NTU = {1: 0.5, 2: 0.5}
MARGIN_NTU = 0.05

# 변화율 제한: 현재 주입률 대비 1스텝(10분)당 허용 변경폭 (ppm)
RAMP_PPM = 2.0
