"""AI 시뮬레이션 — API-037 실행, API-053 목록, API-038 상세, API-054 적용."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import API_PREFIX, paginate, verify_plant
from app.schemas.requests import SimulationApply, SimulationCreate
from app.services.common import MockAPIError, envelope, now_kst, now_kst_iso
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["시뮬레이션"])


@router.post("/simulations")
async def create_simulation(body: SimulationCreate) -> dict:
    """API-037 시나리오 시뮬레이션 실행 요청."""
    sid = state.new_id("SIM")
    sc = body.scenario
    state.simulations[sid] = {
        "simulation_id": sid,
        "status": "queued",
        "created_at": now_kst(),
        "executed_at": None,
        "executed_by": "api_user",
        "input": {
            "scenario_type": sc.scenario_type,
            "processes": sc.processes or [sc.process],
            "overrides": sc.overrides,
        },
        "meets_criteria": True,
    }
    return envelope({"simulation_id": sid, "status": "queued"})


@router.get("/simulations")
async def list_simulations(
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
) -> dict:
    """API-053 시뮬레이션 이력 목록."""
    rows = []
    for sid, sim in state.simulations.items():
        executed_at = sim["executed_at"] or sim["created_at"].isoformat(timespec="seconds")
        rows.append({
            "simulation_id": sid,
            "scenario_type": sim["input"]["scenario_type"],
            "status": state.simulation_status(sid),
            "executed_at": executed_at,
            "executed_by": sim["executed_by"],
            "meets_criteria": sim["meets_criteria"],
        })
    rows.sort(key=lambda r: r["executed_at"], reverse=True)
    if from_:
        rows = [r for r in rows if r["executed_at"] >= from_]
    if to:
        rows = [r for r in rows if r["executed_at"] <= to]
    return envelope({"total": len(rows), "items": paginate(rows, page, size)})


def _get_sim(simulationId: str) -> dict:
    sim = state.simulations.get(simulationId)
    if sim is None:
        raise MockAPIError(404, "SIMULATION_NOT_FOUND",
                           f"시뮬레이션을 찾을 수 없습니다: '{simulationId}'")
    return sim


@router.get("/simulations/{simulationId}")
async def get_simulation(simulationId: str) -> dict:
    """API-038 시뮬레이션 상세/결과."""
    sim = _get_sim(simulationId)
    status = state.simulation_status(simulationId)
    data: dict = {
        "simulation_id": simulationId,
        "status": status,
        "input": sim["input"],
    }
    if status == "done":
        data.update({
            "predictions": [
                {"key": "sed_turbidity_2", "predicted_value": 0.32, "unit": "NTU"},
                {"key": "residual_chlorine_1", "predicted_value": 0.21, "unit": "mg/L"},
            ],
            "comparison": {
                "current": {"clear_turbidity": 0.08, "residual_chlorine": 0.19, "thm": 0.021},
                "scenario": {"clear_turbidity": 0.07, "residual_chlorine": 0.21, "thm": 0.019},
                "optimal": {"clear_turbidity": 0.07, "residual_chlorine": 0.21, "thm": 0.018},
            },
            "economics": {
                "chlorine_saving": 0.05,
                "cost_saving_monthly": 1200000,
                "thm_reduction": 0.003,
            },
            "recommendation": {"summary": "여과속도 -10%, 염소 주입량 -0.05 mg/L 조정"},
            "meets_criteria": sim["meets_criteria"],
        })
    return envelope(data)


@router.post("/simulations/{simulationId}/apply")
async def apply_simulation(simulationId: str, body: SimulationApply) -> dict:
    """API-054 시뮬레이션 결과 적용 → 승인 대기 권고 생성."""
    sim = _get_sim(simulationId)
    if state.simulation_status(simulationId) != "done":
        raise MockAPIError(400, "SIMULATION_NOT_DONE",
                           f"완료되지 않은 시뮬레이션은 적용할 수 없습니다: '{simulationId}'")
    rec_id = state.new_id("REC")
    proc = sim["input"]["processes"][0]
    control, value = next(iter(sim["input"]["overrides"].items()), ("rpm", 145))
    state.pending[rec_id] = {
        "recommendation_id": rec_id,
        "process": proc,
        "title": f"시뮬레이션 {simulationId} {body.plan} 안 적용 요청",
        "control": control,
        "current_value": value,
        "recommended_value": value,
        "predicted_value": value,
        "target_value": value,
        "unit": "-",
        "confidence": 0.85,
        "registered_at": now_kst_iso(),
    }
    return envelope({
        "simulation_id": simulationId,
        "applied_plan": body.plan,
        "recommendation_id": rec_id,
        "status": "pending_approval",
    })
