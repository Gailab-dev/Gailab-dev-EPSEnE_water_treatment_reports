# 정수장 AI 지능화 운영 시스템

광주광역시 **덕남·용연 정수장**을 대상으로 한 AI 기반 지능화 운영 플랫폼입니다.  
응집제 주입, 혼화·응집, 소독 공정에 대해 AI 추천 → 운영자 승인 → 단계적 자동화 구조로 운영합니다.

> **⚠️ 독립 서버 분기 구조**  
> 덕남·용연은 각각 **독립 서버**로 운영됩니다. 이 저장소는 사이트별 자립(self-contained) 트리
> [`_deoknam/`](_deoknam/) · [`_yongyeon/`](_yongyeon/) 로 분리되어 있으며, 두 트리는 서로를 참조하지
> 않습니다. 각 트리는 그대로 **별도 git 저장소로 분리**할 수 있습니다. 공통 인프라 코드
> (app·mlops·simulation·data_pipeline·tests 등)는 각 사이트에 **복제**되어 있습니다.

---

## 핵심 설계 원칙

1. **정수장 완전 분리**: 덕남·용연은 공정 구성, 약품 투입 위치, 혼화 방식, 계측 장비가 달라 모델·설정·데이터를 완전히 분리합니다. → 저장소도 사이트별 독립 트리로 분기되어 있습니다.
2. **군집별 모델 분기**: 유입수 수질 특성을 군집분석으로 분류한 뒤 해당 군집 전용 AI 모델을 적용합니다.
3. **단계적 전환**: AI분석 → AI추천 → AI운영 순서로 점진적으로 자동화 수준을 높입니다.

---

## 전체 처리 흐름

```text
실시간 데이터 수집 (정수장별 분리)
→ 데이터 전처리 / 검증
→ 유입수 군집 분류 (정수장별 군집분류기)
→ 군집별 AI 모델 선택
→ 공정별 AI 예측 (약품 / 혼화응집 / 소독 / 수질예측)
→ 안전제약 기반 권고값 산정
→ XAI 근거 생성
→ API / HMI 연계
→ 운영자 승인 또는 제한적 자동적용
→ 결과 피드백 및 MLOps 재학습
```

---

## 디렉토리 구조

저장소는 두 개의 독립 서버 트리로 분기되어 있습니다. 각 트리는 동일한 내부 구조를 가집니다.

```text
EPSEnE_water_treatment/
│
├── _deoknam/                        # ── 덕남 정수장 독립 서버 트리 ──
│   ├── app/                         # FastAPI 기반 AI API 서버
│   │   ├── api/                     # 라우터 (엔드포인트 정의)
│   │   ├── services/                # 비즈니스 로직 (군집분류, 예측, 권고)
│   │   ├── schemas/                 # Pydantic 입출력 스키마
│   │   ├── core/config.py           # 설정 (PLANT_ID=deoknam 고정, configs 로딩)
│   │   └── main.py                  # FastAPI 앱 진입점 (단일 정수장)
│   │
│   ├── configs/
│   │   ├── deoknam.yaml             # 태그 매핑·안전제약·모델 파라미터
│   │   └── common.yaml              # 공통 설정 (DB·API·재학습 스케줄)
│   │
│   ├── data_pipeline/
│   │   ├── collector/               # 덕남 HMI/SCADA 태그 수집
│   │   ├── preprocessing/           # 결측·이상치 처리, 칼만필터 스무싱
│   │   ├── feature_engineering/     # 래그·롤링 피처, HRT 반영
│   │   └── validation/              # 데이터 품질 검사
│   │
│   ├── ml/                          # 모델 학습 작업 공간 (군집별)
│   │   ├── 00_preprocess_outlier/   # 이상치 전처리 + 노트북 생성기
│   │   ├── 01_clustering/           # 유입수 군집분류기 (K-Means)
│   │   ├── 02_coagulant/            # 응집제 주입률 예측 (cluster_c0~c2)
│   │   ├── 03_mixing/               # 혼화·응집 G값/RPM (cluster_c0~c2)
│   │   ├── 04_chlorine/             # 소독 염소 주입률 (cluster_c0~c2)
│   │   ├── 05_process_prediction/   # 침전·여과·정수 수질 예측 (cluster_c0~c2)
│   │   ├── 06_anomaly_detection/    # 이상탐지 (군집 전이 포함)
│   │   └── utils/                   # 공통 유틸 (데이터 로더, 평가 지표)
│   │
│   ├── models/                      # 학습된 모델 산출물
│   │   ├── cluster/classifier/      # 덕남 군집분류기
│   │   ├── coagulant/cluster_c1~c5/
│   │   ├── mixing/                  # 기계식 급속혼화 (Letterman 산식 기반)
│   │   ├── chlorine/
│   │   ├── process_prediction/
│   │   └── anomaly_detection/
│   │
│   ├── dataset/                     # 덕남 학습 데이터 (덕남_*.parquet, .gitignore 대상)
│   ├── docker/                      # Dockerfile.api + docker-compose.yml (단일 서비스)
│   ├── mlops/                       # 학습·평가·레지스트리·드리프트·롤백
│   ├── simulation/                  # What-if 시뮬레이션 / 최적화
│   ├── tests/                       # unit / integration / api
│   ├── requirements.txt
│   └── README.md                    # 덕남 사이트 안내
│
├── _yongyeon/                       # ── 용연 정수장 독립 서버 트리 ──
│   └── ...                          # 덕남과 동일 구조, 용연 전용 콘텐츠
│                                    #  · 응집제
│                                    #  · 혼화
│                                    #  · 군집
│
├── static/                          # UI 프로토타입 (공유 참고)
│   └── AI정수장 화면설계.html
├── ISSUE_TEMPLATE/                  # 이슈 템플릿 (조직 공통)
├── agent.md
└── README.md                        # (본 문서)
```

> 각 사이트 트리는 자기 루트를 기준으로 동작합니다. 학습 스크립트는 사이트 루트
> (`_deoknam/` 또는 `_yongyeon/`)에서 실행하세요.
> 예: `cd _deoknam && python ml/01_clustering/train_cluster_models.py`

---

## 정수장별 공정 비교

| 항목 | 덕남정수장 | 용연정수장 |
|---|---|---|
| 혼화 방식 | 기계식 급속혼화 (단일) | 복수 혼화방식 선택 가능 |
| 응집플록분석장치 | 없음 | 있음 |
| 자동화 적합성 | 중간 | 높음 |
| 초기 모델 전략 | 침전수 탁도 피드백 기반 역산 | 추후 기입 예정 |

---

## 유입수 군집 정의

현재 군집분류 산출물은 정수장별 `KMEANS_k3`를 최종 채택 기준으로 사용합니다.  
따라서 운영 기준 군집은 C1~C5가 아니라 정수장별 3개 군집으로 정의합니다.

| 정수장 | cluster | 군집명 | 주요 특성 |
| --- | ---: | --- | --- |
| 덕남 | 0 | 저탁도_일반운전 | 대부분의 일반 저탁도 운전 구간 |
| 덕남 | 1 | 중탁도 | 중간 탁도 및 일부 상승 구간 |
| 덕남 | 2 | 고탁도_상승급증 | 탁도 상승 또는 급증 이벤트 구간 |
| 용연 | 0 | 저탁도_일반운전 | 대부분의 일반 저탁도 운전 구간 |
| 용연 | 1 | 고탁도 | 고탁도 이벤트 구간 |
| 용연 | 2 | 극단고탁도 | 극단적 고탁도 이벤트 구간 |

---

## 주요 AI 모델

| 모델 | 알고리즘 | 입력 | 출력 |
|---|---|---|---|
| 군집분류기 | K-Means | 원수 탁도·pH·수온·EC·알칼리도·TOC | 군집 레이블 |
| 응집제 주입률 | XGBoost / LightGBM | 원수 수질, 유량, 군집 레이블 | 응집제 권고 주입률 (ppm) |
| G값/RPM | Letterman 산식 + ML 보정 | 수온, 점도, 응집제 주입률, 군집 | 목표 G값, 권고 RPM |
| 소독 염소 | XGBoost / LSTM | 수온·TOC·Mn, 유량, HRT, 군집 | 전/중/후 염소 권고값 |
| 수질 예측 | XGBoost / LSTM | 운영조건 + 약품 주입률 | 침전수·여과수·잔류염소 예측 |
| 이상탐지 | Isolation Forest + Rule | 센서값, 운영이력, 군집 경계 | 이상 플래그, 군집 전이 경고 |

---


## 인프라 구성

각 정수장은 독립 컨테이너 스택으로 배포됩니다 (`<site>/docker/docker-compose.yml`).

| 컨테이너 | 역할 |
|---|---|
| `epsene-<site>` | FastAPI 추론 API (단일 정수장) |
| `data-collector` | HMI/SCADA 실시간 데이터 수집 |
| `preprocessor` | 전처리 + 칼만필터 센서 스무싱 |
| `model-worker` | 군집별 모델 추론 |
| `simulation-worker` | What-if 시뮬레이션 |
| `mlops-server` | MLflow 모델 관리 |
| `scheduler` | 재학습 / 배치 작업 |
| `postgres` | 운영 DB |
| `redis` | 캐시 / 큐 |
| `nginx` | API Gateway |

---

## 모델 버전 규칙

```text
{plant_id}_{process}_{cluster}_{model_type}_{yyyymmdd}_{version}

예시:
deoknam_coagulant_c1_xgb_20260517_v1.0.0
yongyeon_chlorine_c2_lstm_20260517_v1.0.0
deoknam_cluster_classifier_20260517_v1.0.0
```

---

## 안전 제약 (Safety Guardrail)

- 주입률 상한/하한: 과거 운영 이력 q1~q99 또는 감독원 승인값
- 1회 변경폭 제한: 최근 운영 이력 기반 산정
- 군집 전이 감지 시: 자동적용 즉시 중단, 운영자 알림
- 미학습 수질 범위: 자동적용 금지, 추천만 허용
- 센서 이상 감지 시: AI 권고 중지

---

## 개발 단계

| 단계 | 내용 |
|---|---|
| 1단계 | 요구사항 분석 · 태그 정의 · 운영자 인터뷰 |
| 2단계 | 데이터 마트 구축 · EDA · 군집분석 |
| 3단계 | 군집별 모델 개발 (약품 / 혼화 / 소독 / 수질예측 / 이상탐지) |
| 4단계 | API 서버 개발 · HMI 연계 · XAI 표출 |
| 5단계 | 시운전 · MLOps 체계 · 24시간 무중단 테스트 |

---

## 참고 문서

- [덕남 사이트 안내](_deoknam/README.md) · [용연 사이트 안내](_yongyeon/README.md)
- [덕남 군집분류 README](_deoknam/ml/01_clustering/deoknam_README.md)
- [용연 군집분류 README](_yongyeon/ml/01_clustering/yongyeon_README.md)
- [AI 운영화면 설계 프로토타입](static/AI정수장%20화면설계.html)
