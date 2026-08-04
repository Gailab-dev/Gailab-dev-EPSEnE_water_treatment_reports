# -*- coding: utf-8 -*-
"""덕남 원수성상 슬라이딩 윈도우 군집화 설정.

매 1분 시점 t의 직전 10분(t-9 ~ t) 트레일링 윈도우에서 원수성상 5컬럼의
평균·기울기(분당 변화율)를 계산해 10차원 특징 벡터를 만들고,
StandardScaler + KMeans(k=3)로 군집 라벨을 시점 t에 부여한다.
"""
from pathlib import Path

RANDOM_STATE = 42

# 경로: config.py → clustering → 01_clustering → 덕남 모델 개발
ROOT = Path(__file__).resolve().parents[2]
IN_PARQUET = ROOT / "dataset" / "preprocessing_data" / "덕남_전처리완료.parquet"
CLUSTER_DATA_DIR = ROOT / "dataset" / "clustering_data"
OUT_LABELED = CLUSTER_DATA_DIR / "덕남_군집라벨.parquet"
OUTPUT_DIR = ROOT / "01_clustering" / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODEL_PATH = OUTPUT_DIR / "model" / "덕남_kmeans_k3.joblib"

# 원수성상(착수정) 5컬럼 — 순서가 특징 벡터 순서를 결정
RAW_FEATURES = [
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.착수정_PH",
    "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도",
    "RCS_6.AI.착수정_전기전도도",
]
TURBIDITY_COL = "RCS_6.AI.착수정_TB"
LABEL_COL = "cluster_label"

WINDOW = 10          # 10분 트레일링 윈도우 (1분 × 10샘플)
K = 3                # 군집 수 고정 (기존 cluster0/1/2 체계 유지)
N_INIT = 10
SIL_SAMPLE = 20_000  # silhouette / PCA 산점도 표본 수

# 기존 프로젝트 관례 (덕남_응집제_투입량/plot_timeseries_integrated.py)
CLUSTER_COLORS = {0: "#c44e52", 1: "#55a868", 2: "#8172b2"}
LINE_COLOR = "#4878b0"
