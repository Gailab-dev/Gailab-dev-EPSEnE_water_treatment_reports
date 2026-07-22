"""AI 운영모드 — API-049 조회, API-050 변경."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import API_PREFIX, Process, verify_plant
from app.schemas.requests import AiModeUpdate
from app.services.common import envelope, now_kst_iso
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["AI모드"])


@router.get("/ai-modes")
async def get_ai_modes() -> dict:
    """API-049 공정별 AI 운영모드 목록."""
    return envelope({"modes": list(state.ai_modes.values())})


@router.put("/processes/{process}/ai-mode")
async def set_ai_mode(process: Process, body: AiModeUpdate) -> dict:
    """API-050 공정 AI 운영모드 변경."""
    changed_at = now_kst_iso()
    state.ai_modes[process.value] = {
        "process": process.value,
        "mode": body.mode,
        "changed_at": changed_at,
        "changed_by": body.operator_id,
    }
    return envelope({"process": process.value, "mode": body.mode, "changed_at": changed_at})
