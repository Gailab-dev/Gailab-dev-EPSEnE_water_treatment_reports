"""MLOps — API-039~042, 055~061 (모델 현황/성능/드리프트, 재학습, 후보/배포/롤백, 설정)."""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Query

from app.api.deps import API_PREFIX, verify_plant
from app.schemas.requests import (
    CandidateApprove,
    MlopsSettingsUpdate,
    ModelDeploy,
    ModelRollback,
    RetrainJobCreate,
)
from app.services.common import MockAPIError, envelope, now_kst, now_kst_iso
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["MLOps"])


def _get_model(modelId: str) -> dict:
    model = state.models.get(modelId)
    if model is None:
        raise MockAPIError(404, "MODEL_NOT_FOUND", f"모델을 찾을 수 없습니다: '{modelId}'")
    return model


@router.get("/models/current")
async def current_models() -> dict:
    """API-039 현재 배포 모델 목록."""
    models = [
        {
            "model_id": m["model_id"],
            "name": m["name"],
            "version": m["version"],
            "process": m["process"],
            "deploy_status": m["deploy_status"],
            "metrics": m["metrics"],
        }
        for m in state.models.values()
    ]
    return envelope({"models": models})


@router.get("/models/{modelId}/performance")
async def model_performance(
    modelId: str,
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
) -> dict:
    """API-040 모델 성능 지표."""
    _get_model(modelId)
    now = now_kst()
    default_from = (now - timedelta(days=30)).date().isoformat()
    timeseries = [
        {"date": (now - timedelta(days=d)).date().isoformat(),
         "metrics": {"MAPE": round(4.6 + 0.1 * (d % 3), 1), "R2": round(0.91 - 0.005 * (d % 4), 3)}}
        for d in range(6, -1, -1)
    ]
    return envelope({
        "model_id": modelId,
        "period": {"from": from_ or default_from, "to": to or now.date().isoformat()},
        "metrics": {"MAE": 0.03, "RMSE": 0.05, "R2": 0.91, "MAPE": 4.6},
        "pass_criteria": {"R2": 0.85, "MAPE": 5.0},
        "passed": True,
        "timeseries": timeseries,
        "next_evaluation_at": (now + timedelta(days=13)).replace(
            hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds"),
    })


@router.get("/models/{modelId}/drift")
async def model_drift(modelId: str) -> dict:
    """API-041 데이터/모델 드리프트 현황."""
    _get_model(modelId)
    now = now_kst()
    history = [
        {"t": (now - timedelta(days=d)).date().isoformat(),
         "data_drift_score": round(3.2 - 0.1 * d, 1),
         "model_drift_score": round(1.8 - 0.1 * d, 1)}
        for d in range(6, 0, -1)
    ]
    return envelope({
        "model_id": modelId,
        "data_drift_score": 3.2,
        "model_drift_score": 1.8,
        "status": "normal",
        "alarm": False,
        "thresholds": {
            "data_drift": state.mlops_settings["data_drift_threshold"],
            "performance_drift": state.mlops_settings["performance_drift_threshold"],
        },
        "history": history,
    })


@router.post("/retraining/jobs")
async def create_retraining_job(body: RetrainJobCreate) -> dict:
    """API-042 재학습 잡 생성."""
    _get_model(body.model_id)
    job_id = state.new_id("JOB")
    state.retraining_jobs.insert(0, {
        "job_id": job_id,
        "model_id": body.model_id,
        "trigger": body.trigger,
        "status": "created",
        "requested_by": body.requested_by,
        "started_at": None,
        "finished_at": None,
    })
    return envelope({"job_id": job_id, "status": "created"})


@router.get("/retraining/jobs")
async def list_retraining_jobs(
    model_id: str | None = Query(None),
    status: str | None = Query(None, description="created/running/done/failed"),
) -> dict:
    """API-055 재학습 잡 목록."""
    jobs = state.retraining_jobs
    if model_id:
        jobs = [j for j in jobs if j["model_id"] == model_id]
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return envelope({"jobs": jobs})


@router.get("/models/{modelId}/candidates")
async def model_candidates(modelId: str) -> dict:
    """API-056 후보 모델 목록."""
    model = _get_model(modelId)
    return envelope({"candidates": model["candidates"]})


def _get_candidate(model: dict, candidateId: str) -> dict:
    for cand in model["candidates"]:
        if cand["candidate_id"] == candidateId:
            return cand
    raise MockAPIError(404, "CANDIDATE_NOT_FOUND",
                       f"후보 모델을 찾을 수 없습니다: '{candidateId}'")


@router.post("/models/{modelId}/candidates/{candidateId}/approve")
async def approve_candidate(modelId: str, candidateId: str, body: CandidateApprove) -> dict:
    """API-057 후보 모델 승인."""
    cand = _get_candidate(_get_model(modelId), candidateId)
    cand["status"] = "approved"
    cand["approved_at"] = now_kst_iso()
    cand["approved_by"] = body.approver_id
    return envelope({
        "candidate_id": candidateId,
        "status": "approved",
        "approved_at": cand["approved_at"],
    })


@router.post("/models/{modelId}/deploy")
async def deploy_model(modelId: str, body: ModelDeploy) -> dict:
    """API-058 승인된 후보 배포 — 기존 버전은 롤백 지점으로 보존."""
    model = _get_model(modelId)
    cand = _get_candidate(model, body.candidate_id)
    if cand["status"] != "approved":
        raise MockAPIError(400, "CANDIDATE_NOT_APPROVED",
                           f"승인되지 않은 후보는 배포할 수 없습니다: '{body.candidate_id}'")
    model["rollback_point"] = model["version"]
    model["version"] = cand["version"].replace("-rc1", "")
    model["deploy_status"] = "active"
    cand["status"] = "deployed"
    return envelope({
        "model_id": modelId,
        "version": model["version"],
        "deploy_status": "active",
        "rollback_point": model["rollback_point"],
        "deployed_at": now_kst_iso(),
    })


@router.post("/models/{modelId}/rollback")
async def rollback_model(modelId: str, body: ModelRollback) -> dict:
    """API-059 직전 버전으로 롤백."""
    model = _get_model(modelId)
    if not model["rollback_point"]:
        raise MockAPIError(400, "NO_ROLLBACK_POINT",
                           f"롤백 지점이 없습니다: '{modelId}'")
    model["version"], model["rollback_point"] = model["rollback_point"], None
    return envelope({
        "model_id": modelId,
        "restored_version": model["version"],
        "rolled_back_at": now_kst_iso(),
    })


@router.get("/mlops/settings")
async def get_mlops_settings() -> dict:
    """API-060 MLOps 설정 조회."""
    return envelope(dict(state.mlops_settings))


@router.put("/mlops/settings")
async def update_mlops_settings(body: MlopsSettingsUpdate) -> dict:
    """API-061 MLOps 설정 변경 — 전달된 필드만 반영."""
    updates = body.model_dump(exclude_none=True)
    updates.pop("operator_id", None)
    state.mlops_settings.update(updates)
    return envelope({"saved_at": now_kst_iso()})
