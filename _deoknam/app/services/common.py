"""공통 응답 봉투·시간 유틸·mock 오류 (wtp-api-catalog v0.2.5).

모든 응답은 {success, data, metadata:{generated_at, plant_id}} 봉투를 따른다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.config import settings

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_iso() -> str:
    return now_kst().isoformat(timespec="seconds")


def envelope(data: dict) -> dict:
    return {
        "success": True,
        "data": data,
        "metadata": {
            "generated_at": now_kst_iso(),
            "plant_id": settings.PLANT_ID,
        },
    }


class MockAPIError(Exception):
    """봉투 형식의 오류 응답으로 변환되는 예외."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def to_body(self) -> dict:
        return {
            "success": False,
            "error": {"code": self.code, "message": self.message},
            "metadata": {
                "generated_at": now_kst_iso(),
                "plant_id": settings.PLANT_ID,
            },
        }
