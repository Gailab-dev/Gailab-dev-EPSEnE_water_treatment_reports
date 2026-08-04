# 01_clustering — 용연 원수성상 슬라이딩 윈도우 군집분석 (덕남 이식본)

`dataset/preprocessing_data/용연_전처리완료.parquet` (1분 주기)를 입력으로,
매 1분 시점 t마다 **직전 10분(t-9~t) 트레일링 윈도우**의 원수성상
(원수 탁도/PH/알카리도/온도/전기전도도) 평균·기울기 10차원 특징으로
KMeans(k=3) 군집 라벨을 부여한다. 방법·검증 절차는 덕남과 동일
(벡터화 특징, 탁도 오름차순 재라벨, 온라인 `assign_label` 헬퍼).

## 실행

```bash
cd 01_clustering
python run_clustering.py
```

## 결과 (용연 특성)

유효 1,140,448행 (전체의 40.3% — 원수성상 태그가 없는 초기 구간 ~60%는 라벨 NA).

| 군집 | 비율(유효행) | 특성 |
|---|---|---|
| cluster0 | 57.8% | **동절기**: 수온 10.7℃, 탁도 2.20, 전도도 80.5 |
| cluster1 | 41.5% | **하절기**: 수온 21.4℃, 탁도 2.87, 전도도 75.1 |
| cluster2 | 0.7% | **강우 고탁도 이벤트**: 탁도 63.3 NTU(!), pH 6.72, 알칼리도 15.9 |

덕남(계절 3분할)과 달리 용연은 **강우 이벤트가 별도 군집으로 분리**됐다 —
원수 탁도 변동폭(최대 158NTU)이 덕남보다 훨씬 커서 고탁도 상태가 독립
상태로 잡힌 것. cluster2는 8,138분(약 5.7일 분량)에 불과해 후속 모델링
(02_coag)에서는 표본 부족으로 제외된다. silhouette 0.277 (20k 표본).

## 산출물

`../dataset/clustering_data/용연_군집라벨.parquet` (2,829,600행 × 47컬럼),
`용연_cluster{0,1,2}.parquet`, `output/model/용연_kmeans_k3.joblib`,
`output/cluster_report.csv`, `cluster_centers.csv`, `output/plots/*.png`
