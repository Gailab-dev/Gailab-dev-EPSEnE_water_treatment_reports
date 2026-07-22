"""이벤트/알람 — API-043~048 (목록·등록·ack·close·상세·SSE 스트림).

주의: /events/stream 은 /events/{eventId} 보다 먼저 선언해야 한다.
"""
from __future__ import annotations

import asyncio
import itertools
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.deps import API_PREFIX, paginate, verify_plant
from app.schemas.requests import EventAck, EventClose, EventCreate
from app.services.common import MockAPIError, envelope, now_kst_iso
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["이벤트"])

_LIST_FIELDS = [
    "event_id", "level", "process", "title", "message", "status", "occurred_at",
    "predicted_value", "threshold", "location", "cause_candidates", "related_criteria",
]


def _list_row(evt: dict) -> dict:
    return {k: evt.get(k) for k in _LIST_FIELDS}


@router.get("/events")
async def list_events(
    q: str | None = Query(None, description="제목/내용 검색어"),
    level: str | None = Query(None, description="긴급/경고/정보/AI검토/기록"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
) -> dict:
    """API-043 이벤트 목록."""
    rows = sorted(state.events.values(), key=lambda e: e["occurred_at"], reverse=True)
    if q:
        rows = [e for e in rows if q in e["title"] or q in e["message"]]
    if level:
        rows = [e for e in rows if e["level"] == level]
    if from_:
        rows = [e for e in rows if e["occurred_at"] >= from_]
    if to:
        rows = [e for e in rows if e["occurred_at"] <= to]
    return envelope({
        "total": len(rows),
        "page": page,
        "events": [_list_row(e) for e in paginate(rows, page, size)],
    })


@router.post("/events")
async def create_event(body: EventCreate) -> dict:
    """API-044 이벤트 등록 — SSE 스트림에도 송출된다."""
    eid = state.new_id("EVT")
    evt = {
        "event_id": eid,
        "level": body.level,
        "process": body.process,
        "title": body.title,
        "message": body.message,
        "status": "open",
        "occurred_at": now_kst_iso(),
        "predicted_value": None,
        "threshold": None,
        "location": None,
        "cause_candidates": [],
        "related_criteria": None,
        "related_logs": [],
        "decision_history": [],
        "source": body.source,
    }
    state.events[eid] = evt
    state.sse_queue.append(evt)
    return envelope({"event_id": eid, "status": "open"})


def _sse_frame(evt: dict) -> str:
    payload = {
        "event_id": evt["event_id"],
        "level": evt["level"],
        "process": evt["process"],
        "title": evt["title"],
        "status": evt["status"],
        "occurred_at": evt["occurred_at"],
    }
    return (
        f"event: alarm\n"
        f"id: {evt['event_id']}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


@router.get("/events/stream")
async def event_stream(
    level: str | None = Query(None),
    cycles: int | None = Query(
        None, ge=0,
        description="테스트용: 지정 시 해당 횟수만 순환 송출 후 스트림 종료(미지정 시 무한)",
    ),
    heartbeat: float = Query(15.0, ge=0.1, description="순환 송출 간격(초)"),
) -> StreamingResponse:
    """API-048 실시간 이벤트 SSE 스트림 (기본 무한 — 클라이언트 종료 시 취소)."""

    async def gen():
        yield "retry: 5000\n\n"
        recent = [e for e in state.events.values() if e["status"] == "open"]
        for evt in recent[-3:]:
            if level is None or evt["level"] == level:
                yield _sse_frame(evt)
        cycle = itertools.cycle(list(state.events.values()) or [None])
        n = 0
        while cycles is None or n < cycles:
            n += 1
            # API-044 로 등록된 이벤트를 우선 송출
            while state.sse_queue:
                evt = state.sse_queue.popleft()
                if level is None or evt["level"] == level:
                    yield _sse_frame(evt)
            await asyncio.sleep(heartbeat)
            evt = next(cycle)
            if evt is not None and (level is None or evt["level"] == level):
                yield _sse_frame(evt)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _get_event(eventId: str) -> dict:
    evt = state.events.get(eventId)
    if evt is None:
        raise MockAPIError(404, "EVENT_NOT_FOUND", f"이벤트를 찾을 수 없습니다: '{eventId}'")
    return evt


@router.patch("/events/{eventId}/ack")
async def ack_event(eventId: str, body: EventAck) -> dict:
    """API-045 이벤트 확인(ack)."""
    evt = _get_event(eventId)
    evt["status"] = "ack"
    evt["updated_at"] = now_kst_iso()
    return envelope({"event_id": eventId, "status": "ack", "updated_at": evt["updated_at"]})


@router.patch("/events/{eventId}/close")
async def close_event(eventId: str, body: EventClose) -> dict:
    """API-046 이벤트 종료(close)."""
    evt = _get_event(eventId)
    evt["status"] = "closed"
    evt["updated_at"] = now_kst_iso()
    return envelope({"event_id": eventId, "status": "closed", "updated_at": evt["updated_at"]})


@router.get("/events/{eventId}")
async def event_detail(eventId: str) -> dict:
    """API-047 이벤트 상세."""
    evt = _get_event(eventId)
    detail = {k: v for k, v in evt.items() if k != "source"}
    return envelope(detail)
