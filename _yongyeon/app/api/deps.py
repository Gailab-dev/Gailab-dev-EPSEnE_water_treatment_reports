"""라우터 공통 의존성 — plantId 검증, 공정 enum, 페이징 헬퍼."""
from __future__ import annotations

from enum import Enum

from fastapi import Path

from app.core.config import settings
from app.services.common import MockAPIError

API_PREFIX = "/api/v1/ai/{plantId}"


class Process(str, Enum):
    intake = "intake"
    coagulation = "coagulation"
    sedimentation = "sedimentation"
    filtration = "filtration"
    disinfection = "disinfection"
    clear_water = "clear_water"


async def verify_plant(plantId: str = Path(...)) -> str:
    """{plantId} 경로변수를 이 서버의 정수장과 대조한다 (불일치 → 404 봉투)."""
    if plantId != settings.PLANT_ID:
        raise MockAPIError(
            404, "PLANT_NOT_FOUND",
            f"이 서버는 '{settings.PLANT_ID}' 정수장 전용입니다 (요청: '{plantId}')",
        )
    return plantId


def paginate(items: list, page: int, size: int) -> list:
    start = (page - 1) * size
    return items[start:start + size]
