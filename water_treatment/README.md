# water_treatment 전처리·군집 파이프라인

원수 수질 데이터에 대한 **1차 전처리 → 군집분류 → 2차 전처리** 3단계 파이프라인의
공통 구현 모듈이다. 덕남·용연 두 정수장에 동일 로직을 적용하며, 정수장별 차이는
config JSON(컬럼 매핑·경로)으로만 분리한다.

## 구성

```text
water_treatment/
├── stage1_outlier.py   # 1차: 규칙 기반 스파이크 탐지 + 결정론적 Kalman(RTS) 보정
├── clustering.py       # 군집: 10분 고정구간 집계 + StandardScaler+KMeans(k=3)
└── stage2_outlier.py   # 2차: 군집 내부 IQR·robust Z·IsolationForest 플래그, 가드 삭제

_deoknam/ml/00_preprocess_outlier/
├── config/stage1.json, config/stage2.json   # 덕남 컬럼 매핑·입출력 경로
└── scripts/run_stage1.py, scripts/run_stage2.py
_deoknam/ml/01_clustering/scripts/train_cluster.py

_yongyeon/ml/...                              # 용연 동일 구조
```

## 실행 순서

1. **1차 전처리** — 원수 통합 parquet → `..._1차이상치처리.parquet`

   ```bash
   python _deoknam/ml/00_preprocess_outlier/scripts/run_stage1.py            # 실제 저장
   python _deoknam/ml/00_preprocess_outlier/scripts/run_stage1.py --dry-run  # 분석만
   ```

2. **군집분류** — 1차 산출물 → 10분 라벨 parquet + 군집별 parquet + 모델

   ```bash
   python _deoknam/ml/01_clustering/scripts/train_cluster.py
   python _deoknam/ml/01_clustering/scripts/train_cluster.py --dry-run
   ```

3. **2차 전처리** — 군집별 parquet → 군집별 2차 산출물 + 삭제 로그

   ```bash
   python _deoknam/ml/00_preprocess_outlier/scripts/run_stage2.py               # 전 군집
   python _deoknam/ml/00_preprocess_outlier/scripts/run_stage2.py --cluster 0   # 특정 군집
   ```

용연은 경로의 `_deoknam` → `_yongyeon` 로만 바꿔 동일하게 실행한다.

## 안전 가드

세 단계 모두 보정율/삭제율에 대해 **5% 초과 경고 · 10% 초과 검토필요 · 15% 초과 저장차단**
가드를 적용한다. 15% 초과 시 `GuardBlockedError`로 어떤 산출물도 생성하지 않는다.
결측·0·±inf 값은 탐지/학습에서 제외하고 값·행을 보존한다.

## 테스트

```bash
python -m pytest tests/test_stage1_outlier.py tests/test_raw_water_clustering.py tests/test_stage2_outlier.py -q
```

## 실행 전 확인 사항

- 파이프라인은 외부 의존성 없이 자체 완결적으로 동작한다. 정수장 메타(컬럼 매핑 등)는
  `clustering.py`의 `PLANT_REGISTRY`에 내장되어 있고, `stage1_outlier.py`·`stage2_outlier.py`의
  상대경로는 저장소 루트(`dataset/`·`water_treatment/`가 함께 존재하는 디렉토리)를 기준으로 해석한다.
- 1차 전처리 입력 `dataset/{정수장}_응집제공정_소독공정_통합.parquet`이 존재해야 하며,
  군집분류·2차 전처리는 각각 직전 단계 산출물을 입력으로 사용한다.
