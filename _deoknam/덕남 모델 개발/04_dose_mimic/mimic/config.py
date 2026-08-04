# -*- coding: utf-8 -*-
"""응집제 주입률 운영자 모사 모델 설정.

목적: 무인 운전 시 관행 수준의 주입률 결정을 재현(모사). 절감·최적화가 아님
(그건 03_dose_recommend 담당). 일 단위 5클래스(13~17ppm) 분류로 프레임하고,
persistence(어제 값 유지, 97.1%) 대비 개선과 변경일 감지력을 정직하게 평가한다.
"""
from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[2]
IN_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리완료.parquet"
OUTPUT_DIR = ROOT / "04_dose_mimic" / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
MODELS_DIR = OUTPUT_DIR / "models"

# 대상: 계열별 주입률 설정치
SV_COLS = {1: "PAC.AI.PAC_주입율제어_SV", 2: "PAC.AI.PACS2_주입율제어_SV"}

# 일 단위 집계에 쓰는 원천 컬럼
SRC_COLS = [
    "RCS_6.AI.착수정_TB", "RCS_6.AI.착수정_PH", "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도", "RCS_6.AI.착수정_전기전도도",
    "주입변경.정수량_천m3d", "RCS_6.AI.침전지1_탁도", "RCS_6.AI.침전지2_탁도",
]

CLASSES = [13, 14, 15, 16, 17]   # SV 정수 클래스 (ppm)
TEST_FRAC = 0.2                  # 시간순 분할
XGB = dict(n_estimators=400, learning_rate=0.05, max_depth=4,
           subsample=0.9, colsample_bytree=0.9, tree_method="hist",
           random_state=RANDOM_STATE)
