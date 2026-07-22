# EPSEnE 정수장 AI API 접속 정보 (mock 단계)

> 발주처 전달용 문서 — WTP AI API 카탈로그 v0.2.5 기준

## 1. 기본 정보

| 항목 | 내용 |
| :--- | :--- |
| **덕남 Base URL** | `http://59.3.103.182:8001/api/v1/ai/deoknam` |
| **용연 Base URL** | `http://59.3.103.182:8002/api/v1/ai/yongyeon` |
| 인증 | 없음 (mock 단계 — 추후 Bearer 토큰 적용 예정) |
| 응답 형식 | `{ "success": bool, "data": {...}, "metadata": { "generated_at", "plant_id" } }` |
| API 규격 | WTP AI API 카탈로그 v0.2.5 — 61종 전체 구현 |
| CORS | 전체 허용 (브라우저에서 직접 호출 가능) |

## 2. API 문서 (Swagger)

전체 엔드포인트 목록 확인 및 화면에서 직접 호출 테스트가 가능합니다.

- 덕남: <http://59.3.103.182:8001/docs>
- 용연: <http://59.3.103.182:8002/docs>

## 3. 연결 확인 (Health Check)

| 정수장 | URL | 정상 응답 |
| :--- | :--- | :--- |
| 덕남 | <http://59.3.103.182:8001/health> | `{"status":"ok","plant_id":"deoknam"}` |
| 용연 | <http://59.3.103.182:8002/health> | `{"status":"ok","plant_id":"yongyeon"}` |

## 4. 호출 예시

```http
# 상황판 AI 권고 목록
GET http://59.3.103.182:8001/api/v1/ai/deoknam/dashboard/recommendations
GET http://59.3.103.182:8002/api/v1/ai/yongyeon/dashboard/recommendations

# 상황판 농도 예측 (현재/+1h/+3h/+6h)
GET http://59.3.103.182:8001/api/v1/ai/deoknam/dashboard/concentration-forecast?horizons=1,3,6

# 공정 모니터링 실측+예측
# {process}: intake | coagulation | sedimentation | filtration | disinfection | clear_water
GET http://59.3.103.182:8001/api/v1/ai/deoknam/processes/coagulation/monitoring/forecast

# 공정 분석진단 (공정별 analysisType 상이 — 카탈로그 참조)
GET http://59.3.103.182:8001/api/v1/ai/deoknam/processes/intake/analysis?analysisType=cluster

# 승인 대기 권고 목록 / 권고 결정(승인·반려·보류)
GET  http://59.3.103.182:8001/api/v1/ai/deoknam/recommendations/pending
POST http://59.3.103.182:8001/api/v1/ai/deoknam/recommendations/{recommendationId}/decision

# 이벤트 목록 / 실시간 이벤트 스트림 (SSE, text/event-stream)
GET http://59.3.103.182:8001/api/v1/ai/deoknam/events
GET http://59.3.103.182:8001/api/v1/ai/deoknam/events/stream
```

응답 예시 (상황판 권고):

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "process": "coagulation", "control": "rpm",
        "current_value": 142, "target_value": 145,
        "recommended_value": 145, "predicted_value": 0.34,
        "unit": "rpm", "applicable": true,
        "recommendation_id": "REC-20260722-001", "confidence": 0.86
      }
    ]
  },
  "metadata": { "generated_at": "2026-07-22T09:05:00+09:00", "plant_id": "deoknam" }
}
```

## 5. 참고사항

- **mock 데이터**입니다. 시계열은 호출 시각 기준으로 생성되는 모의 값이며, 실제 계측/AI 모델 결과가 아닙니다.
- 승인·이벤트 등 상태 변경(POST/PUT/PATCH)은 서버 메모리에 반영되며, **서버 재시작 시 초기 상태로 리셋**됩니다.
- 경로변수 `{plantId}`는 각 서버의 정수장과 일치해야 합니다 (8001은 `deoknam`, 8002는 `yongyeon` — 불일치 시 404).

