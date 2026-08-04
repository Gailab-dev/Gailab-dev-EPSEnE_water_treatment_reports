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
│   ├── 덕남 모델 개발/              # ★ 모델 개발 작업 공간 (오프라인 분석·학습 — 진행 중 산출물)
│   │   ├── 00_preprocessing/        # 이상치 처리·칼만 평활 → 전처리완료 parquet
│   │   ├── 01_clustering/           # 원수 성상 군집분석 (K-Means k=3)
│   │   ├── 02_coag/                 # 침전탁도 예측(coag) + 주입률 SV 비교모델(dose_compare)
│   │   ├── 03_dose_recommend/       # 응집제 주입률 추천·백테스트
│   │   ├── 04_dose_mimic/           # 운전자 주입 모방 모델
│   │   ├── 05_mixing_gvalue/        # 교반강도 G값·RPM 산정 (K0 캘리브레이션 모델)
│   │   ├── 06_floc_rpm/             # 혼화·플록형성 G/RPM 규칙기반 추천엔진 (설계문서 §5~12 구현)
│   │   ├── db/                      # step_aiwtp MySQL 연동 (config/connection/loader)
│   │   ├── 착수공정/                # 응집공정 모델(intake_coag) + 설계문서
│   │   └── 혼화응집공정/            # G/RPM 추천·제어 상세설계문서
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
│                                    #  · 용연 모델 개발/ (00 전처리 · 01 군집 ·
│                                    #    02 응집제 · 03 추천 · 05 G값 · db 미러)
│
├── static/                          # UI 프로토타입 (공유 참고)
│   └── AI정수장 화면설계.html
├── ISSUE_TEMPLATE/                  # 이슈 템플릿 (조직 공통)
└── README.md                        # (본 문서)
```

> 각 사이트 트리는 자기 루트를 기준으로 동작합니다. 학습 스크립트는 사이트 루트
> (`_deoknam/` 또는 `_yongyeon/`)에서 실행하세요.
> 예: `cd _deoknam && python ml/01_clustering/train_cluster_models.py`

---

## 정수장별 공정 비교

| 항목 | 덕남정수장 | 용연정수장 |
|---|---|---|
| 혼화 방식 | 착수정 낙차부 수류혼화 (기계식 혼화기 12대 현재 미사용) | 복수 혼화방식 선택 가능 |
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
| 응집제 주입률 | XGBoost / RF / LSTM·GRU 비교 | 원수 수질, 유량, 군집 레이블 | 응집제 권고 주입률 (ppm) |
| G값/RPM | 규칙기반 — 표준 RPM 룩업보간(표 2.3-13) + K0 물리모델 검산 | 수온, 유량, 목표 G(50/30/10) | 3단 플록큐레이터 권고 RPM, G·Gt, 사유코드 |
| 소독 염소 | XGBoost / LSTM | 수온·TOC·Mn, 유량, HRT, 군집 | 전/중/후 염소 권고값 |
| 수질 예측 | XGBoost / LSTM | 운영조건 + 약품 주입률 | 침전수·여과수·잔류염소 예측 |
| 이상탐지 | Isolation Forest + Rule | 센서값, 운영이력, 군집 경계 | 이상 플래그, 군집 전이 경고 |

---

## 모델 개발 현황

오프라인 모델 개발은 `_deoknam/덕남 모델 개발/`(선행)과 `_yongyeon/용연 모델 개발/`(미러)에서
진행합니다. 입력은 통합 parquet(1분 주기, 덕남 95만 행: 2024-05~2026-02)입니다.

| 모듈 | 내용 | 상태 |
|---|---|---|
| `00_preprocessing` | 이상치 2단계 처리 + 칼만 평활 → 전처리완료 parquet | 덕남·용연 완료 |
| `01_clustering` | 원수 성상 K-Means k=3 군집 (저탁도/중탁도/고탁도) | 덕남·용연 완료, API 실연동 |
| `02_coag` | 응집제→침전탁도 예측(coag) + 주입률 SV 5모델 비교(dose_compare) | 덕남 완료, 용연 dose_compare 진행 |
| `03_dose_recommend` | 군집별 응집제 주입률 추천 + 백테스트 | 덕남·용연 완료 |
| `04_dose_mimic` | 운전자 주입 모방 모델 | 덕남 완료 |
| `05_mixing_gvalue` | G=K0·√(ρ/μ)·rpm^1.5 캘리브레이션 — 기술진단서 표 재현 오차 <2% | 덕남·용연 완료 |
| `06_floc_rpm` | 혼화·플록형성 3단 RPM 규칙기반 추천엔진 (아래 상세) | 덕남 완료 |
| `착수공정` | 응집공정 체류시간 정렬·과제A/B 모델 (intake_coag) | 덕남 완료 |

**06_floc_rpm — 혼화·플록형성 G/RPM 추천엔진** (내부 설계문서 §5~12 구현):

- 수온별 표준 RPM 룩업보간(표 2.3-13) + 목표 G^(2/3) 보정을 추천 기준으로,
  05의 K0 물리모델은 독립 검산(편차 경보 3%)으로 이중화
- 착수정 낙차부 수류혼화 G 감시(§8), 상류지연 시간정렬(§6.4), 상하한·점감·Gt
  제약검사(§10.4), 사유코드 비트마스크(§15.3), FT101 단위 미확정 시 제어 차단(§6.1)
- 검증: 단위시험 25건 통과, 전기간(950,286행) 백테스트 가용률 99.1%,
  점감·한계 위반 0건 — 상세는 [06_floc_rpm README](_deoknam/덕남%20모델%20개발/06_floc_rpm/README.md)
- 현재는 화면추천 전용(1~2단계). 인버터·RPM 실측 태그 연결 후 운전자 승인형
  제어(3단계)로 확장

---

## 핵심 API

WTP AI API **카탈로그 v0.2.5(61종) 전체 구현** 완료 (현재 mock 단계 — 유입수
군집 분석은 실 DB·군집모델 연동, 나머지는 모의 데이터). 상세 설명은
**[API_GUIDE.md](API_GUIDE.md)**, 접속 주소·Swagger·호출 예시는
**[API_ACCESS_INFO.md](API_ACCESS_INFO.md)** 참고.

- Base URL: `http://<host>:8001/api/v1/ai/deoknam` (덕남) · `:8002/api/v1/ai/yongyeon` (용연)
- 응답 봉투: `{ "success", "data", "metadata": { "generated_at", "plant_id" } }`

| 그룹 | 대표 엔드포인트 | 기능 |
|---|---|---|
| 상황판 | `GET /dashboard/recommendations` · `/dashboard/concentration-forecast` | 공정별 AI 추천 목록, 농도 시평별 예측(+1h/+3h/+6h) |
| 공정 (6공정 공통) | `GET /processes/{process}/monitoring/forecast` · `/analysis` · `/operation-judgement` · `/anomaly-timeseries` · `/recommendations/summary` | 실측+예측, 분석진단(공정별 analysisType), XAI 판단근거, 이상탐지, 권고 요약 |
| 권고 승인 | `GET /recommendations/pending` · `POST /recommendations/{id}/decision` · `GET /recommendations/decisions` | 승인 대기 목록, 승인/반려/보류, 결정 이력 |
| 시뮬레이션 | `POST /simulations` · `GET /simulations/{id}` · `POST /simulations/{id}/apply` | 시나리오 실행·결과 조회·권고 생성 |
| MLOps | `GET /models/current` · `/models/{id}/performance`·`/drift` · `POST /retraining/jobs` · `/models/{id}/deploy`·`/rollback` | 배포 모델·성능·드리프트, 재학습, 후보 승인·배포·롤백 |
| 이벤트 | `GET /events` · `GET /events/stream`(SSE) · `PATCH /events/{id}/ack`·`/close` | 이벤트 목록·실시간 스트림·확인/종료 |
| AI 모드 | `GET /ai-modes` · `PUT /processes/{process}/ai-mode` | 공정별 AI 운영모드 조회/변경 |

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

- [API 설명서](API_GUIDE.md) · [API 접속 정보 (발주처 전달용)](API_ACCESS_INFO.md)
- [덕남 사이트 안내](_deoknam/README.md) · [용연 사이트 안내](_yongyeon/README.md)
- 설계문서: [착수공정 (응집공정 AI)](_deoknam/덕남%20모델%20개발/착수공정/설계문서.md) ·
  [응집제 추천 v2](_deoknam/덕남%20모델%20개발/02_coag/응집제추천설계문서.md) ·
  [혼화응집공정 모델링 리포트](_deoknam/덕남%20모델%20개발/혼화응집공정/보고서/덕남_응집공정_모델링_리포트.md)
- [AI 운영화면 설계 프로토타입](static/AI정수장%20화면설계.html)
