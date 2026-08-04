# 00_preprocessing — 덕남 raw_data 공통 전처리 패키지

`dataset/raw_data/덕남_응집제공정_소독공정_통합.parquet` (1분 주기, 40컬럼)의
전 컬럼을 전처리해 `dataset/preprocessing_data/덕남_전처리완료.parquet` (44컬럼)로 저장한다.

## 실행

```bash
cd 00_preprocessing
python run_preprocess.py
```

원본이 `raw_data/`에 없으면 `EPSEnE_water_treatment/dataset/`에서 자동 복사한다.

## 산출물

| 경로 | 내용 |
|---|---|
| `../dataset/preprocessing_data/덕남_전처리완료.parquet` | 정제값(원 컬럼명 유지) + 전/중염소 분할 4컬럼 |
| `output/preprocess_report.csv` | 컬럼별 방법별 이상치 건수·제거율 (utf-8-sig) |
| `output/compare/<컬럼>.png` | 전처리 전/후 비교 시계열 44장 |
| `output/outlier_rate_heatmap.png` | 변수 × 방법 탐지율 히트맵 |

## 전처리 로직

컬럼을 4개 그룹으로 나눠 처리한다 (`preprocessing/config.py`).

| 그룹 | 컬럼 | 처리 |
|---|---|---|
| A. 수질센서 (18) | pH/탁도/온도/전도도/알칼리도/잔류염소 | 물리범위 → Hampel∪IQR(∪구간제한 ISO) → NaN → 칼만 채움 |
| B. 유량/수위 (9) | PAC 주입유량·주입량, FT/LT | A와 동일, floor 런타임 산출 (펌프 on/off 보호) |
| C. 설정치 (11) | PAC 목표/SV, 주입변경.* | 물리범위 밖만 NaN → ffill (계단 엣지 보존, 탐지기·칼만 미적용) |
| D. 적산계 (2) | 주입_유량 적산 | diff<0 글리치(짧은 원복)만 NaN → ffill |

**이상치 판정** (`preprocessing/outliers.py`):
`최종 = (Hampel ∪ IQR) ∪ (IsolationForest ∩ dilate(Hampel ∪ IQR, ±30분))`
— IsolationForest는 고정 contamination(0.005)으로 정상 구간까지 일정 비율을 지목하므로,
Hampel/IQR이 탐지한 구간의 ±30분 확장 윈도우 안에서 잡은 지점만 채택한다.

**결측 채움** (`preprocessing/kalman.py`): 1D 랜덤워크 칼만 필터. NaN 구간은 관측 갱신
없이 상태를 예측-전파해 채운다. 탁도 4컬럼은 참고 프로젝트(덕남_응집제_투입량)와
동일하게 전구간 평활 대체(r_scale=5.0), 그 외는 NaN 지점만 대체. 선두/후미 결측
(예: 잔류염소 초기 ~10.7만 행)은 채우지 않고 유지한다.

**전중염소 분할** (`preprocessing/pipeline.py: split_chlorine`): 전처리 완료된
`주입변경.전중염소_{1,2}차_KG`를 전염소 1 : 중염소 10 비율로 분할해
`주입변경.전염소_N차_KG`(×1/11), `주입변경.중염소_N차_KG`(×10/11) 4컬럼을 추가한다.

## 패키지 구조

```
preprocessing/
├── config.py         # 경로·컬럼그룹·파라미터 (Hampel/IQR/ISO/칼만/분할비)
├── io_utils.py       # setup_output(한글 폰트/Agg), 원본 복사, 로드/저장
├── outliers.py       # hampel_mask / iqr_mask / isoforest_mask / dilate_mask / combined_mask
├── kalman.py         # kalman_smooth / kalman_fill
├── pipeline.py       # 그룹별 클리너 + run_pipeline + split_chlorine
└── visualization.py  # 전/후 비교 PNG + 탐지율 히트맵
```

파라미터(Hampel window/nσ/floor, IQR k=3.0·6h, ISO contamination 등)는
`덕남_응집제_투입량/src/preprocess.py`의 검증된 값을 승계했다.
