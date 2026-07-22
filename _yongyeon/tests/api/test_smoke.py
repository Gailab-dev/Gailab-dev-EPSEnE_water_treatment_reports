"""wtp-api-catalog v0.2.5 — 61개 mock 엔드포인트 스모크 테스트.

실행 (사이트 루트에서): pytest tests/api -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.mock_state import state

client = TestClient(app)
P = settings.PLANT_ID
BASE = f"/api/v1/ai/{P}"

PROCESSES = ["intake", "coagulation", "sedimentation", "filtration", "disinfection", "clear_water"]

ANALYSIS_CASES = [
    ("intake", "cluster"), ("intake", "raw_water_recommendation"),
    ("coagulation", "scatter"), ("coagulation", "mixer_control"),
    ("sedimentation", "efficiency"), ("sedimentation", "coagulant_suitability"),
    ("filtration", None), ("disinfection", None), ("clear_water", None),
]


def _ok(res, status=200):
    assert res.status_code == status, res.text
    body = res.json()
    assert body["success"] is True
    assert body["metadata"]["plant_id"] == P
    return body["data"]


# ---------------------------------------------------------------- 상황판
def test_dashboard_recommendations():
    data = _ok(client.get(f"{BASE}/dashboard/recommendations"))
    assert data["items"], "대기 권고가 시드되어 있어야 한다"
    item = data["items"][0]
    for key in ("process", "control", "current_value", "recommended_value",
                "recommendation_id", "confidence", "applicable"):
        assert key in item


def test_concentration_forecast():
    data = _ok(client.get(f"{BASE}/dashboard/concentration-forecast", params={"horizons": "1,3,6"}))
    assert data["horizons"] == [1, 3, 6]
    fc = data["forecasts"][0]
    assert [v["horizon"] for v in fc["values"]] == [1, 3, 6]


# ---------------------------------------------------------------- 공정 30종
@pytest.mark.parametrize("process", PROCESSES)
def test_monitoring_forecast(process):
    data = _ok(client.get(f"{BASE}/processes/{process}/monitoring/forecast",
                          params={"horizon": 6, "interval": "1h"}))
    assert data["process"] == process
    ind = data["indicators"][0]
    assert len(ind["forecast"]) == 6
    assert {"lower", "upper"} <= set(ind["forecast"][0])
    if process == "coagulation":
        assert "basin" in ind and "stage" in ind


@pytest.mark.parametrize("process", PROCESSES)
def test_recommendations_summary(process):
    data = _ok(client.get(f"{BASE}/processes/{process}/recommendations/summary"))
    reco = data["recommendations"][0]
    assert {"target", "safe_range", "expected_effect", "confidence"} <= set(reco)
    assert ("basin" in reco) == (process == "coagulation")


@pytest.mark.parametrize("process,atype", ANALYSIS_CASES)
def test_analysis(process, atype):
    params = {"analysisType": atype} if atype else {}
    data = _ok(client.get(f"{BASE}/processes/{process}/analysis", params=params))
    assert data["analysis_type"] == (atype or ANALYSIS_DEFAULTS[process])
    assert data["result"]["metrics"]
    assert "recommendation" in data


ANALYSIS_DEFAULTS = {
    "intake": "cluster", "coagulation": "scatter", "sedimentation": "efficiency",
    "filtration": "default", "disinfection": "default", "clear_water": "default",
}


def test_analysis_invalid_type():
    res = client.get(f"{BASE}/processes/intake/analysis", params={"analysisType": "nope"})
    assert res.status_code == 400
    assert res.json()["success"] is False


@pytest.mark.parametrize("process", PROCESSES)
def test_operation_judgement(process):
    data = _ok(client.get(f"{BASE}/processes/{process}/operation-judgement"))
    assert data["contributions"]
    assert {"interpretation", "operation_judgement", "action_guide"} <= set(data)


@pytest.mark.parametrize("process", PROCESSES)
def test_anomaly_timeseries(process):
    data = _ok(client.get(f"{BASE}/processes/{process}/anomaly-timeseries",
                          params={"past_hours": 12, "forecast_hours": 3}))
    assert len(data["past"]) == 12
    assert len(data["forecast"]) == 3
    assert {"anomaly_score", "is_anomaly", "causes"} <= set(data)


# ---------------------------------------------------------------- 권고 흐름
def test_decision_flow():
    pending = _ok(client.get(f"{BASE}/recommendations/pending"))["items"]
    assert pending
    rid = pending[0]["recommendation_id"]

    data = _ok(client.post(
        f"{BASE}/recommendations/{rid}/decision",
        json={"decision": "approve", "applied_value": 145, "operator_id": "op_test"},
    ))
    assert data["decision"] == "approve"
    assert data["audit_id"].startswith("AUD-")

    left = _ok(client.get(f"{BASE}/recommendations/pending"))["items"]
    assert rid not in [r["recommendation_id"] for r in left]

    hist = _ok(client.get(f"{BASE}/recommendations/decisions"))
    assert rid in [r["recommendation_id"] for r in hist["items"]]


def test_decisions_filters():
    data = _ok(client.get(f"{BASE}/recommendations/decisions",
                          params={"decision": "approve", "page": 1, "size": 3}))
    assert all(r["decision"] == "approve" for r in data["items"])
    assert len(data["items"]) <= 3


def test_decision_unknown_id():
    res = client.post(f"{BASE}/recommendations/REC-00000000-999/decision",
                      json={"decision": "approve", "operator_id": "op"})
    assert res.status_code == 404
    assert res.json()["success"] is False


# ---------------------------------------------------------------- 시뮬레이션
def test_simulation_flow():
    data = _ok(client.post(f"{BASE}/simulations", json={
        "scenario": {
            "scenario_type": "combined",
            "process": "coagulation",
            "processes": ["coagulation", "disinfection"],
            "water_quality": {"turbidity": 12.5, "ph": 7.1},
            "overrides": {"rpm": 150, "post_cl_dose_rate": 0.31},
            "horizon": 6,
        }
    }))
    sid = data["simulation_id"]
    assert data["status"] == "queued"

    state.simulations[sid]["status"] = "done"  # 경과시간 대기 없이 완료 처리
    detail = _ok(client.get(f"{BASE}/simulations/{sid}"))
    assert detail["status"] == "done"
    assert {"comparison", "economics", "recommendation", "meets_criteria"} <= set(detail)

    listed = _ok(client.get(f"{BASE}/simulations"))
    assert sid in [r["simulation_id"] for r in listed["items"]]

    applied = _ok(client.post(f"{BASE}/simulations/{sid}/apply",
                              json={"plan": "optimal", "operator_id": "op_test"}))
    assert applied["status"] == "pending_approval"
    pending = _ok(client.get(f"{BASE}/recommendations/pending"))["items"]
    assert applied["recommendation_id"] in [r["recommendation_id"] for r in pending]


def test_simulation_unknown_id():
    assert client.get(f"{BASE}/simulations/SIM-00000000-999").status_code == 404


# ---------------------------------------------------------------- MLOps
def test_models_current():
    data = _ok(client.get(f"{BASE}/models/current"))
    assert len(data["models"]) == 6
    assert {"model_id", "version", "process", "deploy_status", "metrics"} <= set(data["models"][0])


def test_model_performance_and_drift():
    perf = _ok(client.get(f"{BASE}/models/MDL-COAG-01/performance"))
    assert perf["passed"] is True and perf["timeseries"]
    drift = _ok(client.get(f"{BASE}/models/MDL-COAG-01/drift"))
    assert drift["status"] == "normal" and drift["history"]


def test_candidate_approve_deploy_rollback():
    cands = _ok(client.get(f"{BASE}/models/MDL-DISF-01/candidates"))["candidates"]
    assert cands
    cid = cands[0]["candidate_id"]

    approved = _ok(client.post(f"{BASE}/models/MDL-DISF-01/candidates/{cid}/approve",
                               json={"approver_id": "manager_park"}))
    assert approved["status"] == "approved"

    deployed = _ok(client.post(f"{BASE}/models/MDL-DISF-01/deploy",
                               json={"candidate_id": cid, "operator_id": "op_test"}))
    assert deployed["version"] == "1.5.0"
    assert deployed["rollback_point"] == "1.4.2"

    rolled = _ok(client.post(f"{BASE}/models/MDL-DISF-01/rollback",
                             json={"operator_id": "op_test", "reason": "성능 저하"}))
    assert rolled["restored_version"] == "1.4.2"


def test_retraining_jobs_flow():
    created = _ok(client.post(f"{BASE}/retraining/jobs", json={
        "model_id": "MDL-COAG-01", "trigger": "manual", "requested_by": "ai_admin",
    }))
    assert created["status"] == "created"
    jobs = _ok(client.get(f"{BASE}/retraining/jobs", params={"model_id": "MDL-COAG-01"}))["jobs"]
    assert created["job_id"] in [j["job_id"] for j in jobs]


def test_mlops_settings_flow():
    before = _ok(client.get(f"{BASE}/mlops/settings"))
    assert "data_drift_threshold" in before
    _ok(client.put(f"{BASE}/mlops/settings",
                   json={"data_drift_threshold": 9.5, "operator_id": "op_test"}))
    after = _ok(client.get(f"{BASE}/mlops/settings"))
    assert after["data_drift_threshold"] == 9.5


# ---------------------------------------------------------------- 이벤트
def test_events_list_and_filters():
    data = _ok(client.get(f"{BASE}/events"))
    assert data["total"] >= 6
    warn = _ok(client.get(f"{BASE}/events", params={"level": "경고"}))
    assert all(e["level"] == "경고" for e in warn["events"])


def test_event_lifecycle():
    created = _ok(client.post(f"{BASE}/events", json={
        "level": "경고", "process": "intake", "title": "테스트 이벤트",
        "message": "테스트 메시지", "source": "pytest",
    }))
    eid = created["event_id"]

    detail = _ok(client.get(f"{BASE}/events/{eid}"))
    assert detail["status"] == "open"

    acked = _ok(client.patch(f"{BASE}/events/{eid}/ack", json={"operator_id": "op_test"}))
    assert acked["status"] == "ack"

    closed = _ok(client.patch(f"{BASE}/events/{eid}/close",
                              json={"operator_id": "op_test", "action_detail": "조치 완료"}))
    assert closed["status"] == "closed"


def test_event_unknown_id():
    assert client.get(f"{BASE}/events/EVT-00000000-999").status_code == 404


def test_event_stream_sse():
    # TestClient의 ASGI 전송은 본문을 전부 모아 반환하므로 cycles로 유한 종료시켜 검증한다.
    with client.stream("GET", f"{BASE}/events/stream",
                       params={"cycles": 1, "heartbeat": 0.1}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        buf = "".join(res.iter_text())
    assert buf.startswith("retry: 5000")
    assert "event: alarm" in buf
    assert '"event_id"' in buf


# ---------------------------------------------------------------- AI 모드
def test_ai_mode_flow():
    modes = _ok(client.get(f"{BASE}/ai-modes"))["modes"]
    assert len(modes) == 6

    updated = _ok(client.put(f"{BASE}/processes/coagulation/ai-mode",
                             json={"mode": "recommend", "operator_id": "op_test"}))
    assert updated["mode"] == "recommend"

    modes = _ok(client.get(f"{BASE}/ai-modes"))["modes"]
    coag = next(m for m in modes if m["process"] == "coagulation")
    assert coag["mode"] == "recommend" and coag["changed_by"] == "op_test"


# ---------------------------------------------------------------- plantId 검증
def test_wrong_plant_returns_404_envelope():
    other = "yongyeon" if P == "deoknam" else "deoknam"
    res = client.get(f"/api/v1/ai/{other}/dashboard/recommendations")
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "PLANT_NOT_FOUND"
