"""권고 결정/이력 — API-002 결정, API-051 결정 이력, API-052 대기 목록."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import API_PREFIX, paginate, verify_plant
from app.schemas.requests import DecisionRequest
from app.services.common import MockAPIError, envelope, now_kst_iso
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["권고"])


@router.post("/recommendations/{recommendationId}/decision")
async def decide_recommendation(recommendationId: str, body: DecisionRequest) -> dict:
    """API-002 권고 승인/반려/보류 — 승인·반려 시 pending에서 제거되고 이력에 남는다."""
    rec = state.pending.get(recommendationId)
    if rec is None:
        raise MockAPIError(404, "RECOMMENDATION_NOT_FOUND",
                           f"대기 중인 권고가 없습니다: '{recommendationId}'")
    decided_at = now_kst_iso()
    audit_id = state.new_id("AUD")
    if body.decision != "hold":
        state.pending.pop(recommendationId)
        state.decisions.insert(0, {
            "recommendation_id": recommendationId,
            "process": rec["process"],
            "control": rec["control"],
            "recommended_value": rec["recommended_value"],
            "applied_value": body.applied_value if body.decision == "approve" else None,
            "decision": body.decision,
            "operator_id": body.operator_id,
            "comment": body.comment,
            "decided_at": decided_at,
        })
    return envelope({
        "recommendation_id": recommendationId,
        "decision": body.decision,
        "applied_value": body.applied_value,
        "audit_id": audit_id,
        "decided_at": decided_at,
    })


@router.get("/recommendations/decisions")
async def decision_history(
    process: str | None = Query(None),
    decision: str | None = Query(None, description="approve/reject/hold"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
) -> dict:
    """API-051 승인/적용 결정 이력."""
    rows = state.decisions
    if process:
        rows = [r for r in rows if r["process"] == process]
    if decision:
        rows = [r for r in rows if r["decision"] == decision]
    if from_:
        rows = [r for r in rows if r["decided_at"] >= from_]
    if to:
        rows = [r for r in rows if r["decided_at"] <= to]
    return envelope({"total": len(rows), "items": paginate(rows, page, size)})


@router.get("/recommendations/pending")
async def pending_recommendations() -> dict:
    """API-052 승인 대기 권고 목록."""
    items = [
        {
            "recommendation_id": rec["recommendation_id"],
            "process": rec["process"],
            "title": rec["title"],
            "control": rec["control"],
            "current_value": rec["current_value"],
            "recommended_value": rec["recommended_value"],
            "unit": rec["unit"],
            "registered_at": rec["registered_at"],
        }
        for rec in state.pending.values()
    ]
    return envelope({"items": items})
