"""덕남 정수장 AI API — FastAPI 진입점 (골격).

독립 서버: 이 인스턴스는 덕남 정수장만 담당한다.
실행 (사이트 루트 deoknam/ 에서):
    uvicorn app.main:app --host 0.0.0.0 --port 8001
    또는  python -m app.main
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s plant API", settings.PLANT_ID)
    logger.info("Model base path: %s", settings.MODEL_BASE_PATH)
    # TODO: 이 정수장의 군집 분류기·공정별 모델 로드
    yield
    logger.info("Shutting down %s plant API", settings.PLANT_ID)


app = FastAPI(
    title=f"EPSEnE Water Treatment API — {settings.PLANT_ID}",
    description="정수장 공정 최적화 AI API (단일 정수장 독립 서버)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "plant_id": settings.PLANT_ID}


# TODO: 라우터 등록
# from app.api import routers
# for r in routers:
#     app.include_router(r)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
    )
