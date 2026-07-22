"""공정별 API (30개) — 모니터링 예측·권고 요약·분석진단·XAI 판단·이상탐지.

공정 6종(intake/coagulation/sedimentation/filtration/disinfection/clear_water)을
파라미터화된 5개 라우트로 처리한다. coagulation만 basin/stage 필드·필터가 추가된다.
  API-004/010/016/022/027/032  GET .../monitoring/forecast
  API-005/011/017/023/028/033  GET .../recommendations/summary
  API-006/007/012/013/018/019/024/029/034  GET .../analysis
  API-008/014/020/025/030/035  GET .../operation-judgement
  API-009/015/021/026/031/036  GET .../anomaly-timeseries
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import API_PREFIX, Process, verify_plant
from app.services.common import MockAPIError, envelope
from app.services.mock_data import (
    ANALYSIS_TYPES,
    PROCESS_INDICATORS,
    PROCESS_JUDGEMENT,
    PROCESS_LABELS,
    PROCESS_SUMMARY_RECOS,
    analysis_payload,
    current_value,
    find_indicator,
    forecast_series,
    history_series,
    variant,
)

router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(verify_plant)], tags=["공정"])


def _basin_for(i: int) -> str:
    basins = variant()["basins"]
    return basins[i % len(basins)]


@router.get("/processes/{process}/monitoring/forecast")
async def monitoring_forecast(
    process: Process,
    horizon: int = Query(12, ge=1, le=48, description="예측 시평(기본 12h)"),
    interval: str = Query("1h", description="집계 간격(예: 1h)"),
    basinId: str | None = Query(None, description="응집지 필터(coagulation 전용)"),
    stageNo: int | None = Query(None, ge=1, le=3, description="교반 단계 필터(coagulation 전용)"),
) -> dict:
    """API-004/010/016/022/027/032 공정 모니터링 실측+예측."""
    indicators = []
    for i, ind in enumerate(PROCESS_INDICATORS[process.value]):
        row = {
            "key": ind["key"],
            "name": ind["name"],
        }
        if process is Process.coagulation:
            row["basin"] = _basin_for(i)
            row["stage"] = ind.get("stage", 1)
            if basinId is not None and row["basin"] != basinId:
                continue
            if stageNo is not None and row["stage"] != stageNo:
                continue
        row.update({
            "unit": ind["unit"],
            "current": current_value(ind),
            "history": history_series(ind, points=24, interval=interval),
            "forecast": forecast_series(ind, points=horizon, interval=interval),
        })
        indicators.append(row)
    return envelope({"process": process.value, "indicators": indicators})


@router.get("/processes/{process}/recommendations/summary")
async def recommendations_summary(process: Process) -> dict:
    """API-005/011/017/023/028/033 공정 AI 권고사항 요약."""
    recos = []
    for i, base in enumerate(PROCESS_SUMMARY_RECOS[process.value]):
        row = dict(base)
        if process is Process.coagulation:
            row["basin"] = _basin_for(i)
        else:
            row.pop("basin", None)
            row.pop("stage", None)
        recos.append(row)
    return envelope({
        "process": process.value,
        "summary": f"{PROCESS_LABELS[process.value]} AI 권고사항 요약",
        "recommendations": recos,
    })


@router.get("/processes/{process}/analysis")
async def process_analysis(
    process: Process,
    analysisType: str | None = Query(None, description="분석 유형(공정별 상이, 미지정 시 기본값)"),
    period: str | None = Query(None, description="분석 기간(예: 7d)"),
) -> dict:
    """API-006/007/012/013/018/019/024/029/034 공정 분석진단."""
    allowed = ANALYSIS_TYPES[process.value]
    atype = analysisType or allowed[0]
    if atype not in allowed:
        raise MockAPIError(
            400, "INVALID_ANALYSIS_TYPE",
            f"'{process.value}' 공정의 analysisType은 {allowed} 중 하나여야 합니다 (요청: '{atype}')",
        )
    data = analysis_payload(process.value, atype)
    if period:
        data["period"] = period
    return envelope(data)


@router.get("/processes/{process}/operation-judgement")
async def operation_judgement(process: Process) -> dict:
    """API-008/014/020/025/030/035 XAI 운영 판단 근거."""
    j = PROCESS_JUDGEMENT[process.value]
    return envelope({
        "process": process.value,
        "interpretation": j["interpretation"],
        "operation_judgement": j["operation_judgement"],
        "action_guide": j["action_guide"],
        "contributions": j["contributions"],
    })


@router.get("/processes/{process}/anomaly-timeseries")
async def anomaly_timeseries(
    process: Process,
    indicator: str | None = Query(None, description="지표 키(미지정 시 공정 대표 지표)"),
    past_hours: int = Query(24, ge=1, le=168),
    forecast_hours: int = Query(6, ge=1, le=48),
) -> dict:
    """API-009/015/021/026/031/036 이상탐지 시계열."""
    ind = find_indicator(process.value, indicator)
    if ind is None:
        raise MockAPIError(
            400, "INVALID_INDICATOR",
            f"'{process.value}' 공정에 없는 지표입니다: '{indicator}'",
        )
    forecast = [
        {"t": row["t"], "v": row["v"]}
        for row in forecast_series(ind, points=forecast_hours, with_bounds=False)
    ]
    return envelope({
        "process": process.value,
        "indicator": ind["key"],
        "unit": ind["unit"],
        "past": history_series(ind, points=past_hours),
        "forecast": forecast,
        "anomaly_score": 0.18,
        "is_anomaly": False,
        "causes": ["원수 탁도 상승"],
    })
