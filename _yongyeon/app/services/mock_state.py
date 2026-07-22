"""In-memory mock 상태 저장소 — import 시 시드된다.

단일 uvicorn 워커 전제(잠금 없음). 서버 재시작 시 초기 상태로 되돌아간다.
POST/PUT/PATCH 가 GET 응답에 반영되도록 흐름을 실제로 연결한다:
  결정(API-002) → pending(052) 제거·decisions(051) 추가
  시뮬레이션(037) → 경과시간 기반 status → 상세(038)·apply(054 → 신규 pending)
  이벤트(044) → 목록/상세/ack/close(043~047)·SSE(048)
  mlops settings(061)·ai-mode(050)·모델 승인/배포/롤백(057~059)·재학습 잡(042→055)
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta

from app.services.common import now_kst, now_kst_iso
from app.services.mock_data import PROCESSES, variant


def _iso(dt) -> str:
    return dt.isoformat(timespec="seconds")


class MockState:
    def __init__(self):
        self._seq: dict[str, int] = {}
        now = now_kst()
        op = variant()["operator"]

        # --- 권고 (pending / decisions) -----------------------------------
        self.pending: dict[str, dict] = {}
        self.decisions: list[dict] = []

        rec1 = self.new_id("REC")  # REC-...-001
        self.pending[rec1] = {
            "recommendation_id": rec1,
            "process": "coagulation",
            "title": "혼화응집 교반기 145 rpm 승인/적용 요청",
            "control": "rpm",
            "current_value": 142,
            "recommended_value": 145,
            "predicted_value": 0.34,
            "target_value": 145,
            "unit": "rpm",
            "confidence": 0.86,
            "registered_at": _iso(now - timedelta(minutes=40)),
        }
        rec2 = self.new_id("REC")
        self.pending[rec2] = {
            "recommendation_id": rec2,
            "process": "disinfection",
            "title": "소독 후염소 0.31 mg/L 승인/적용 요청",
            "control": "post_cl_dose_rate",
            "current_value": 0.28,
            "recommended_value": 0.31,
            "predicted_value": 0.22,
            "target_value": 0.20,
            "unit": "mg/L",
            "confidence": 0.88,
            "registered_at": _iso(now - timedelta(minutes=25)),
        }

        _seed_decisions = [
            ("disinfection", "post_cl_dose_rate", 0.31, 0.31, "approve", "수온 반영 승인"),
            ("coagulation", "rpm", 148, 148, "approve", "야간 탁도 상승 대응"),
            ("coagulation", "coagulant_dose", 33.0, None, "reject", "실측 지수 안정으로 반려"),
            ("filtration", "backwash_interval", 40, None, "hold", "역세 일정 협의 후 재검토"),
            ("intake", "intake_flow", 4100, 4100, "approve", "강우 대비 취수량 하향"),
            ("disinfection", "pre_chlorine_dose", 1.2, 1.2, "approve", "전염소 상향 적용"),
            ("sedimentation", "sludge_draw_cycle", 6, None, "reject", "계면 안정으로 반려"),
            ("clear_water", "outflow_valve", 62, 62, "approve", "배수지 수위 균형 조정"),
        ]
        for i, (proc, control, rv, av, dec, comment) in enumerate(_seed_decisions):
            self.decisions.append({
                "recommendation_id": self.new_id("REC"),
                "process": proc,
                "control": control,
                "recommended_value": rv,
                "applied_value": av,
                "decision": dec,
                "operator_id": op,
                "comment": comment,
                "decided_at": _iso(now - timedelta(hours=30 - i * 3)),
            })

        # --- 이벤트 --------------------------------------------------------
        self.events: dict[str, dict] = {}
        self.sse_queue: deque[dict] = deque()
        _seed_events = [
            ("경고", "intake", "원수 탁도 상승 감지", "원수 탁도 상승 추세 감지", "open",
             7.1, 6.0, variant()["intake_location"], ["강우 유입", "취수원 변동"], "유입탁도 관리기준 6.0 NTU"),
            ("긴급", "disinfection", "잔류염소 하한 미달 예측", "예측 잔류염소 0.18 mg/L", "open",
             0.18, 0.20, "정수지 유출", ["후염소 주입률 부족", "염소소모계수 상승"], "잔류염소 0.20~0.50 mg/L"),
            ("정보", "filtration", "역세척 주기 도래 예상", "손실수두 추세 기준 6시간 후 역세 예상", "open",
             1.8, 1.8, "여과지 1계열", ["여과 지속시간 증가"], "손실수두 관리기준 1.8 m"),
            ("AI검토", "coagulation", "플록 형성 지수 하락", "플록 형성 지수 완만한 하락 추세", "ack",
             0.79, 0.80, "응집지", ["교반 강도 부족", "수온 하강"], "플록 지수 관리기준 0.80"),
            ("기록", "clear_water", "정수탁도 일간 리포트", "정수탁도 24시간 평균 0.07 NTU", "closed",
             0.07, 0.10, "정수지", [], "정수탁도 수질기준 0.10 NTU"),
            ("경고", "sedimentation", "침전수 탁도 상승 감지", "1계열 침전수 탁도 상승", "closed",
             2.1, 2.0, "침전지 1계열", ["응집 불량", "단락류 발생"], "침전탁도 관리기준 2.0 NTU"),
        ]
        for i, (level, proc, title, msg, status, pv, th, loc, causes, criteria) in enumerate(_seed_events):
            eid = self.new_id("EVT")
            occurred = now - timedelta(hours=len(_seed_events) - i)
            self.events[eid] = {
                "event_id": eid,
                "level": level,
                "process": proc,
                "title": title,
                "message": msg,
                "status": status,
                "occurred_at": _iso(occurred),
                "predicted_value": pv,
                "threshold": th,
                "location": loc,
                "cause_candidates": causes,
                "related_criteria": criteria,
                "related_logs": [
                    {"t": occurred.strftime("%H:%M"), "text": "알람 발생"},
                    {"t": (occurred - timedelta(minutes=1)).strftime("%H:%M"), "text": "예측값 갱신"},
                ],
                "decision_history": (
                    [{"recommendation_id": self.decisions[0]["recommendation_id"],
                      "decision": self.decisions[0]["decision"]}]
                    if proc == "disinfection" else []
                ),
                "source": "ai_engine",
            }

        # --- 시뮬레이션 ----------------------------------------------------
        self.simulations: dict[str, dict] = {}
        for i, (stype, procs, overrides, meets) in enumerate([
            ("combined", ["coagulation", "disinfection"], {"rpm": 150, "post_cl_dose_rate": 0.31}, True),
            ("water_quality", ["coagulation"], {"coagulant_dose": 33.0}, False),
        ]):
            sid = self.new_id("SIM")
            self.simulations[sid] = {
                "simulation_id": sid,
                "status": "done",
                "created_at": now - timedelta(hours=3 - i),
                "executed_at": _iso(now - timedelta(hours=3 - i)),
                "executed_by": op,
                "input": {"scenario_type": stype, "processes": procs, "overrides": overrides},
                "meets_criteria": meets,
            }

        # --- AI 운영모드 ---------------------------------------------------
        _seed_modes = {
            "intake": "recommend", "coagulation": "analysis", "sedimentation": "none",
            "filtration": "analysis", "disinfection": "operate", "clear_water": "recommend",
        }
        self.ai_modes: dict[str, dict] = {
            p: {"process": p, "mode": _seed_modes[p],
                "changed_at": _iso(now - timedelta(hours=12)), "changed_by": op}
            for p in PROCESSES
        }

        # --- MLOps ---------------------------------------------------------
        self.mlops_settings: dict = {
            "data_drift_threshold": 8.0,
            "performance_drift_threshold": 5.0,
            "mape_target": 95.0,
            "deviation_condition": "6h",
            "retraining_policy": "manual_only",
            "deploy_policy": "approve_then_manual",
            "approvers": ["운영책임자", "AI 관리자"],
        }

        _seed_models = [
            ("MDL-INTK-01", "착수 원수질 예측", "intake", "1.2.1"),
            ("MDL-COAG-01", "혼화응집 예측", "coagulation", "1.4.0"),
            ("MDL-SEDI-01", "침전 탁도 예측", "sedimentation", "1.3.2"),
            ("MDL-FILT-01", "여과 손실수두 예측", "filtration", "1.1.4"),
            ("MDL-DISF-01", "소독 잔류염소 예측", "disinfection", "1.4.2"),
            ("MDL-CLWT-01", "정수 수질 예측", "clear_water", "1.0.6"),
        ]
        self.models: dict[str, dict] = {}
        for mid, name, proc, ver in _seed_models:
            self.models[mid] = {
                "model_id": mid,
                "name": name,
                "process": proc,
                "version": ver,
                "deploy_status": "active",
                "metrics": {"MAE": 0.03, "R2": 0.91},
                "rollback_point": None,
                "candidates": [],
            }
        cand = self.new_id("CAND")
        self.models["MDL-DISF-01"]["candidates"].append({
            "candidate_id": cand,
            "version": "1.5.0-rc1",
            "metrics": {"MAPE": 4.6},
            "baseline_metrics": {"MAPE": 5.2},
            "improvement": {"MAPE": -0.6},
            "evaluated_at": _iso(now - timedelta(hours=28)),
            "status": "evaluated",
        })

        self.retraining_jobs: list[dict] = [
            {"job_id": self.new_id("JOB"), "model_id": "MDL-DISF-01", "trigger": "drift",
             "status": "done", "requested_by": "ai_admin",
             "started_at": _iso(now - timedelta(hours=31)),
             "finished_at": _iso(now - timedelta(hours=29))},
            {"job_id": self.new_id("JOB"), "model_id": "MDL-COAG-01", "trigger": "scheduled",
             "status": "running", "requested_by": "ai_admin",
             "started_at": _iso(now - timedelta(hours=1)), "finished_at": None},
        ]

    # ------------------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        self._seq[prefix] = self._seq.get(prefix, 0) + 1
        return f"{prefix}-{now_kst():%Y%m%d}-{self._seq[prefix]:03d}"

    def simulation_status(self, sid: str) -> str:
        """생성 후 경과시간으로 queued→running→done 을 흉내낸다."""
        sim = self.simulations[sid]
        if sim["status"] == "done":
            return "done"
        elapsed = (now_kst() - sim["created_at"]).total_seconds()
        if elapsed < 1:
            return "queued"
        if elapsed < 3:
            return "running"
        sim["status"] = "done"
        sim["executed_at"] = now_kst_iso()
        return "done"


state = MockState()
