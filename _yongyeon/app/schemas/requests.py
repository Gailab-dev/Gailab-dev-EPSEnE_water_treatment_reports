"""요청 바디 Pydantic 모델 (wtp-api-catalog v0.2.5 request 테이블 기준).

응답은 카탈로그 예시를 계약으로 하는 dict + envelope 방식이라 모델을 두지 않는다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["approve", "reject", "hold"]
AiMode = Literal["analysis", "recommend", "operate", "none"]
EventLevel = Literal["긴급", "경고", "정보", "AI검토", "기록"]
RetrainTrigger = Literal["scheduled", "drift", "manual"]
ScenarioType = Literal["water_quality", "operation", "combined"]


class DecisionRequest(BaseModel):
    """API-002 권고 결정."""
    decision: Decision
    applied_value: float | None = None
    operator_id: str
    comment: str | None = None


class Scenario(BaseModel):
    scenario_type: ScenarioType
    process: str
    processes: list[str] | None = None
    water_quality: dict[str, float] | None = None
    overrides: dict[str, float]
    horizon: int | None = None


class SimulationCreate(BaseModel):
    """API-037 시뮬레이션 실행."""
    scenario: Scenario


class SimulationApply(BaseModel):
    """API-054 시뮬레이션 결과 적용."""
    plan: str
    operator_id: str


class RetrainJobCreate(BaseModel):
    """API-042 재학습 잡 생성."""
    model_id: str
    trigger: RetrainTrigger
    dataset_range: dict[str, str] | None = None
    requested_by: str


class CandidateApprove(BaseModel):
    """API-057 후보 모델 승인."""
    approver_id: str
    comment: str | None = None


class ModelDeploy(BaseModel):
    """API-058 모델 배포."""
    candidate_id: str
    operator_id: str


class ModelRollback(BaseModel):
    """API-059 모델 롤백."""
    operator_id: str
    reason: str


class MlopsSettingsUpdate(BaseModel):
    """API-061 MLOps 설정 변경 — 전달된 필드만 반영."""
    data_drift_threshold: float | None = None
    performance_drift_threshold: float | None = None
    mape_target: float | None = None
    deviation_condition: str | None = None
    retraining_policy: str | None = None
    deploy_policy: str | None = None
    operator_id: str


class EventCreate(BaseModel):
    """API-044 이벤트 등록."""
    level: EventLevel
    process: str | None = None
    title: str
    message: str
    source: str


class EventAck(BaseModel):
    """API-045 이벤트 확인."""
    operator_id: str
    note: str | None = None


class EventClose(BaseModel):
    """API-046 이벤트 종료."""
    operator_id: str
    action_detail: str | None = None


class AiModeUpdate(BaseModel):
    """API-050 AI 운영모드 변경."""
    mode: AiMode
    operator_id: str
    reason: str | None = None


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=200)
