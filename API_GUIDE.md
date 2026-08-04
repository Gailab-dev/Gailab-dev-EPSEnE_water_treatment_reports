# WTP AI API 설명서

덕남·용연 정수장 AI API의 구조와 엔드포인트 설명입니다. WTP AI API 카탈로그
**v0.2.5(61종)** 를 전부 구현했으며, 현재 **mock 단계**(유입수 군집 분석 일부만
실 DB 연동)입니다. 접속 주소·Swagger·호출 예시는
[API_ACCESS_INFO.md](API_ACCESS_INFO.md) 참고.

## 1. 공통 규격

| 항목 | 내용 |
| --- | --- |
| Base URL | `http://<host>:8001/api/v1/ai/deoknam` (덕남) · `:8002/api/v1/ai/yongyeon` (용연) |
| 서버 구조 | 정수장별 **독립 서버** — 경로변수 `{plantId}`가 서버의 정수장과 다르면 404 (`PLANT_NOT_FOUND`) |
| 응답 봉투 | `{ "success": bool, "data": {...}, "metadata": { "generated_at", "plant_id" } }` |
| 오류 봉투 | `{ "success": false, "error": { "code", "message" } }` — 예: `INVALID_ANALYSIS_TYPE`(400), `PLANT_NOT_FOUND`(404) |
| 인증 | 없음 (mock 단계 — 운영 전환 시 Bearer 토큰 예정) |
| CORS | 전체 허용 (운영 전환 시 발주처 도메인으로 제한 예정) |
| 상태 저장 | POST/PUT/PATCH 결과는 서버 메모리 반영 — **재시작 시 초기화** |
| 공정 키 `{process}` | `intake` 착수 · `coagulation` 혼화응집 · `sedimentation` 침전 · `filtration` 여과 · `disinfection` 소독 · `clear_water` 정수 |

헬스체크는 봉투 없이 `GET /health` → `{"status":"ok","plant_id":"deoknam"}`.

## 2. 엔드포인트 그룹

### 상황판 (Dashboard)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /dashboard/recommendations` | API-001 | 공정별 AI 추천/판단 목록 (현재값·목표값·권고값·적용가능 여부·confidence) |
| `GET /dashboard/concentration-forecast` | API-003 | 주요 농도의 시평별 예측 — `?horizons=1,3,6` (현재/+1h/+3h/+6h) |

### 공정 모니터링·분석 (Processes) — 6개 공정 공통

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /processes/{process}/monitoring/forecast` | API-004~032 | 공정 실측 + 예측 시계열 |
| `GET /processes/{process}/recommendations/summary` | API-005~033 | 공정 AI 권고사항 요약 (혼화응집은 지·단별 `basin`/`stage` 포함) |
| `GET /processes/{process}/analysis` | API-006~034 | 분석진단 — `?analysisType=` 공정별 상이(아래 표), `?period=7d` |
| `GET /processes/{process}/operation-judgement` | API-008~035 | XAI 운영 판단 근거 (해석·판단·조치가이드·기여도) |
| `GET /processes/{process}/anomaly-timeseries` | API-009~036 | 이상탐지 시계열 — `?indicator=`, `?past_hours=1~168` |

`analysisType` 허용값 (미지정 시 첫 번째가 기본):

| 공정 | analysisType |
| --- | --- |
| intake | `cluster`(유입수 군집 — **실 DB·군집모델 연동**, 실패 시 mock 폴백) · `raw_water_recommendation` |
| coagulation | `scatter` · `mixer_control` |
| sedimentation | `efficiency` · `coagulant_suitability` |
| filtration / disinfection / clear_water | `default` |

### 권고 승인 (Recommendations)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /recommendations/pending` | API-052 | 승인 대기 권고 목록 |
| `POST /recommendations/{recommendationId}/decision` | API-002 | 승인/반려/보류 결정 — 승인·반려 시 pending에서 제거, 이력에 기록 |
| `GET /recommendations/decisions` | API-051 | 승인/적용 결정 이력 |

### 시뮬레이션 (Simulations)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `POST /simulations` | API-037 | 시나리오 시뮬레이션 실행 요청 |
| `GET /simulations` | API-053 | 시뮬레이션 이력 목록 |
| `GET /simulations/{simulationId}` | API-038 | 시뮬레이션 상세/결과 |
| `POST /simulations/{simulationId}/apply` | API-054 | 결과 적용 → 승인 대기 권고 생성 |

### MLOps (모델 관리)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /models/current` | API-039 | 현재 배포 모델 목록 |
| `GET /models/{modelId}/performance` | API-040 | 모델 성능 지표 |
| `GET /models/{modelId}/drift` | API-041 | 드리프트 현황 (7일·1개월·분기 3구간) |
| `POST /retraining/jobs` · `GET /retraining/jobs` | API-042·055 | 재학습 잡 생성/목록 |
| `GET /models/{modelId}/candidates` | API-056 | 후보 모델 목록 |
| `POST /models/{modelId}/candidates/{candidateId}/approve` | API-057 | 후보 모델 승인 |
| `POST /models/{modelId}/deploy` | API-058 | 승인 후보 배포 — 기존 버전은 롤백 지점으로 보존 |
| `POST /models/{modelId}/rollback` | API-059 | 직전 버전으로 롤백 |
| `GET /mlops/settings` · `PUT /mlops/settings` | API-060·061 | MLOps 설정 조회/변경(부분 갱신) |

### 이벤트 (Events)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /events` | API-043 | 이벤트 목록 |
| `POST /events` | API-044 | 이벤트 등록 — SSE 스트림에도 송출 |
| `GET /events/stream` | API-048 | 실시간 이벤트 스트림 (SSE, `text/event-stream`) |
| `GET /events/{eventId}` | API-047 | 이벤트 상세 |
| `PATCH /events/{eventId}/ack` · `/close` | API-045·046 | 이벤트 확인/종료 |

### AI 운영모드 (AI Modes)

| Method·경로 | 카탈로그 | 설명 |
| --- | --- | --- |
| `GET /ai-modes` | API-049 | 공정별 AI 운영모드 목록 |
| `PUT /processes/{process}/ai-mode` | API-050 | 공정 AI 운영모드 변경 |

## 3. 구현 위치와 실 데이터 연동 현황

서버 코드는 사이트 트리별 [`_deoknam/app/`](_deoknam/app/) ·
[`_yongyeon/app/`](_yongyeon/app/) (동일 구조).

```text
app/main.py            # FastAPI 진입점 — /health, 라우터 등록, CORS
app/api/deps.py        # API_PREFIX(/api/v1/ai/{plantId}), plantId 검증, 공정 enum
app/api/*.py           # 그룹별 라우터 (dashboard·processes·recommendations·
                       #   simulations·mlops·events·ai_modes)
app/services/          # mock 데이터·상태, 군집모델 실 DB 연동(clustering.py)
```

| 구분 | 상태 |
| --- | --- |
| 유입수 군집 분석 (`intake` + `analysisType=cluster`) | **실 연동** — KMeans k=3 군집모델 + step_aiwtp DB 조회, 초기화 실패 시 mock 폴백 |
| 그 외 60종 | mock — 호출 시각 기준 모의 시계열 생성 |

모델 개발 결과를 API에 연결하는 작업은 `덕남 모델 개발/`·`용연 모델 개발/`의
모듈(전처리→군집→응집제→G/RPM 추천) 순으로 진행 중이며, 각 모듈의 산출물
스키마는 해당 폴더 README 참고.
