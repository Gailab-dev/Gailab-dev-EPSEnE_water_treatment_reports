"""API 라우터 등록 — main.py 에서 routers 순회 + register_error_handlers 호출."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import (
    ai_modes,
    dashboard,
    events,
    mlops,
    processes,
    recommendations,
    simulations,
)
from app.services.common import MockAPIError

routers = [
    dashboard.router,
    processes.router,
    recommendations.router,
    simulations.router,
    mlops.router,
    events.router,
    ai_modes.router,
]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MockAPIError)
    async def mock_api_error_handler(request: Request, exc: MockAPIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())
