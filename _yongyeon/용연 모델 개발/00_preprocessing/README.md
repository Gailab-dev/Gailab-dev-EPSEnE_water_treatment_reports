# 00_preprocessing — 용연 raw_data 공통 전처리 패키지 (덕남 이식본)

`dataset/raw_data/용연_응집제공정_소독공정_통합.parquet` (1분 주기, 2,829,600행,
2020-08-15 ~ 2025-12-31)의 측정 36컬럼을 전처리해
`dataset/preprocessing_data/용연_전처리완료.parquet`로 저장한다.
원본의 `_flag`/`_src` 보조 컬럼 30개는 로드 시 제외한다.

## 실행

```bash
cd 00_preprocessing
python run_preprocess.py    # 전체 약 1~1.5시간 (2.8M행)
```

원본이 `raw_data/`에 없으면 `EPSEnE_water_treatment/dataset/`에서 자동 복사한다.

## 전처리 로직 (덕남과 동일 알고리즘)

| 그룹 | 컬럼 | 처리 |
|---|---|---|
| A. 수질센서 (15) | 원수 5항목, 침전지 PH#1/#2·탁도, 여과지 PH/탁도, 정수지 탁도/PH/잔류염소, 여과지·후처리 잔류염소 | 물리범위 → Hampel∪IQR(∪구간제한 ISO) → NaN → 칼만 채움 |
| B. 유량/약품 (10) | 원수유입유량, 착수정 유입유량, 액체약품 1~3, 염소 투입량 3종, 분말투입기, 염소라인압력 | A와 동일, floor 런타임 산출 |
| C. 설정치 (11) | PAC 주입률/투입량 설정, 전·중·후염소 주입량 설정(PPM), RCLT.투입량PV_1~4, dead tag | 물리범위 밖만 NaN → ffill (계단 엣지 보존) |

**이상치 판정**: `최종 = (Hampel ∪ IQR) ∪ (IsolationForest ∩ dilate(Hampel∪IQR, ±30분))`
**결측 채움**: 1D 랜덤워크 칼만 (탁도 3컬럼 전구간 평활 r=5.0, 그 외 NaN 지점만)

덕남 대비 제거된 단계: D그룹(적산계 — 용연에 해당 컬럼 없음), 전중염소 1:10 분할
(용연은 전염소/중염소가 별도 컬럼으로 존재).

## 결과

- 2,829,600행 × 36컬럼 보존, A/B 그룹 제거율 전 컬럼 **< 0.32%**
- 침전지/여과지 탁도·액체약품(~60%), RCLT.*(~34%)의 원본 결측 블록은
  유효구간 밖으로 유지 (칼만은 유효구간 내부만 채움)

## 산출물

`../dataset/preprocessing_data/용연_전처리완료.parquet`,
`output/preprocess_report.csv`, `output/compare/*.png`, `output/outlier_rate_heatmap.png`
