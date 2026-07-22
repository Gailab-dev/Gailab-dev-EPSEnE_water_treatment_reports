"""상황판 — API-001 대시보드 권고, API-003 농도 예측."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import API_PREFIX, verify_plant
from app.services.common import envelope
from app.services.mock_data import current_value, find_indicator
from app.services.mock_state import state

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["상황판"])


@router.get("/dashboard/recommendations")
async def dashboard_recommendations() -> dict:
    """API-001 공정별 AI 추천/판단 목록."""
    items = [
        {
            "process": rec["process"],
            "control": rec["control"],
            "current_value": rec["current_value"],
            "target_value": rec["target_value"],
            "recommended_value": rec["recommended_value"],
            "predicted_value": rec["predicted_value"],
            "unit": rec["unit"],
            "applicable": True,
            "recommendation_id": rec["recommendation_id"],
            "confidence": rec["confidence"],
        }
        for rec in state.pending.values()
    ]
    return envelope({"items": items})


# 상황판 농도 예측 대상 지표 (현재/+1h/+3h/+6h)
_FORECAST_KEYS = [
    ("disinfection", "residual_chlorine_1"),
    ("clear_water", "clear_turbidity_1"),
    ("coagulation", "coagulant_dose"),
]


@router.get("/dashboard/concentration-forecast")
async def concentration_forecast(
    horizons: str = Query("1,3,6", description="예측 시평 목록(콤마 구분, 기본 1,3,6)"),
) -> dict:
    """API-003 주요 농도 시평별 예측."""
    hs = [int(h) for h in horizons.split(",") if h.strip()]
    forecasts = []
    for proc, key in _FORECAST_KEYS:
        ind = find_indicator(proc, key)
        cur = current_value(ind)
        values = [
            {"horizon": h, "predicted_value": round(cur * (1 - 0.015 * h), ind["ndigits"])}
            for h in hs
        ]
        forecasts.append({
            "key": ind["key"],
            "name": ind["name"],
            "unit": ind["unit"],
            "current_value": cur,
            "values": values,
            "target": ind["target"],
        })
    return envelope({"horizons": hs, "forecasts": forecasts})
