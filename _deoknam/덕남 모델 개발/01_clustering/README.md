# 01_clustering — 원수성상 슬라이딩 윈도우 군집분석

`dataset/preprocessing_data/덕남_전처리완료.parquet` (1분 주기)를 입력으로,
매 1분 시점 t마다 **직전 10분(t-9분 ~ t분, 10샘플) 트레일링 윈도우**의 원수성상
특징을 계산해 KMeans(k=3) 군집 라벨을 시점 t에 부여한다.
윈도우는 1분씩 밀리며 10개 값 중 9개는 직전 윈도우와 겹친다.

## 실행

```bash
cd 01_clustering
python run_clustering.py
```

## 방법

**특징 (10차원)**: 원수성상 5컬럼(착수정 TB/PH/AL/온도/전기전도도) 각각에 대해
- `__mean`: 윈도우 10샘플 평균
- `__slope`: x = 0..9(분)에 대한 OLS 회귀 기울기 = **분당 변화율**.
  `slope = Σ(x-4.5)·y / 82.5` (x 중심화로 ȳ 항 소거)

배치 계산은 `numpy sliding_window_view` + `einsum`으로 벡터화되어 있으며,
매 분 온라인 계산(`model.assign_label`)과 수식이 동일하다 (검증 완료).

**모델**: `StandardScaler → KMeans(n_clusters=3, random_state=42, n_init=10)`
(기존 `water_treatment/clustering.py` 계약 승계). 학습 후 군집 중심의
착수정_TB__mean 오름차순으로 라벨을 재배열해 **cluster0 = 저탁도 상태**로 고정한다.

**NA 정책**: 선두 9행(불완전 윈도우)과 윈도우에 결측이 포함된 시점
(착수정 센서 데이터가 없는 초기 ~10.7만 행 포함)은 라벨 NA(Int64).

## 산출물

| 경로 | 내용 |
|---|---|
| `../dataset/clustering_data/덕남_군집라벨.parquet` | 원본 44 + 피처 10 + cluster_label (950,286행) |
| `../dataset/clustering_data/덕남_cluster{0,1,2}.parquet` | 라벨별 분할 (다운스트림 모델링용) |
| `output/model/덕남_kmeans_k3.joblib` | 모델 번들 (pipeline·label_map·피처정의·window) |
| `output/cluster_report.csv` | 군집별 건수/비율, silhouette(20k 표본) |
| `output/cluster_centers.csv` | 원단위 군집 중심표 |
| `output/plots/*.png` | 시간점유, PCA 산점도, 피처분포 박스플롯, 월별 점유율 |

## 온라인(실시간) 사용

```python
from clustering import model

bundle = model.load_model()               # 덕남_kmeans_k3.joblib
label = model.assign_label(recent_df, bundle)   # recent_df: 최근 10행(1분×10) DataFrame
# 윈도우에 결측이 있으면 None
```

배치 산출 라벨은 이 함수를 매 1분 반복 호출한 결과와 동일하다.

## 패키지 구조

```
clustering/
├── config.py         # 경로·RAW_FEATURES·WINDOW=10·K=3·색상 관례
├── io_utils.py       # setup_output, load_preprocessed(1분 연속성 검증), 저장
├── features.py       # make_window_features(벡터화) / window_features_single(온라인) / naive(검증)
├── model.py          # fit/relabel/predict/centers/silhouette/save/load/assign_label
└── visualization.py  # 시간점유·PCA·박스플롯·월별점유율
```
