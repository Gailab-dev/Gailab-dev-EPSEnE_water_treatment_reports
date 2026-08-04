"""군집 실모델 연동 — 학습된 KMeans(k=3) 로드 + 실 DB(step_aiwtp) 최근 원수성상으로 현재 군집 산출.

자기완결(01_clustering 패키지 미의존): 번들은 joblib로 로드하고 특징 수식은 여기서 재현한다.
알칼리도·전기전도도는 WIDE 계측 테이블에 없으므로 WTP_SMLT_HSTRY(있으면)→학습중앙값 순으로 채운다.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.core import config as CFG
from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine():
    if not (settings.WTP_DB_HOST and settings.WTP_DB_USER and settings.WTP_DB_PASSWORD
            and settings.WTP_DB_NAME):
        raise RuntimeError("WTP_DB_* 환경변수 미설정 — DB 접속정보를 env/.env로 주입하세요")
    url = (f"mysql+pymysql://{settings.WTP_DB_USER}:{settings.WTP_DB_PASSWORD}"
           f"@{settings.WTP_DB_HOST}:{settings.WTP_DB_PORT}/{settings.WTP_DB_NAME}?charset=utf8mb4")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


@lru_cache(maxsize=1)
def load_state() -> dict:
    """번들(pipeline·label_map·raw_cols·window) 로드 (1회 캐시).
    군집 중심 정의는 별도 csv에 의존하지 않고 번들의 KMeans에서 역산한다(배포 견고성)."""
    return {"bundle": joblib.load(CFG.CLUSTER_MODEL_PATH)}


def _window_features(rows: pd.DataFrame, raw_cols: list, window: int) -> np.ndarray:
    """10행×5성상 → 10차원 [c__mean, c__slope, ...]. slope=중심화 OLS 계수(분당 변화율)."""
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    feats = []
    for c in raw_cols:
        y = rows[c].to_numpy(float)
        feats.append(float(np.mean(y)))
        feats.append(float((xc * y).sum() / denom))
    return np.asarray(feats, dtype=float).reshape(1, -1)


def _smlt_value(engine, smlt_col: str):
    """WTP_SMLT_HSTRY 최근값(있으면). 비어있으면 None."""
    try:
        q = (f"SELECT {smlt_col} v FROM {CFG.SMLT_TABLE} "
             f"WHERE LCLGV_CD=:c AND {smlt_col} IS NOT NULL ORDER BY REG_DT DESC LIMIT 1")
        r = pd.read_sql(text(q), engine, params={"c": CFG.LCLGV_CD})
        if len(r) and pd.notna(r["v"].iloc[0]):
            return float(r["v"].iloc[0])
    except Exception as e:  # noqa: BLE001
        logger.debug("SMLT 조회 실패(%s): %s", smlt_col, e)
    return None


def read_recent_raw(engine, raw_cols: list, window: int):
    """WIDE 최근 window행(탁도·PH·온도) + 알칼리도·전기전도도(SMLT→중앙값). (정렬 rows, 원본, 소스)."""
    cols = ", ".join(CFG.DB2RAW.keys())
    # 화면 표시용 부가 성상(망간·조류·현재약품)도 함께 조회 — WIDE에 존재
    q = (f"SELECT MSRMT_DT, {cols}, INT_MANG, INT_ALGA, INT_PAC "
         f"FROM {CFG.MEASR_TABLE} ORDER BY MSRMT_DT DESC LIMIT {int(window)}")
    df = pd.read_sql(text(q), engine).sort_values("MSRMT_DT").reset_index(drop=True)
    out = pd.DataFrame(index=range(len(df)))
    for dbcode, rawcol in CFG.DB2RAW.items():
        out[rawcol] = pd.to_numeric(df[dbcode], errors="coerce").to_numpy()
    src = "측정(탁도·PH·온도)+중앙값(알칼리도·전기전도도)"
    for rawcol, fillval in CFG.MISSING_FILL.items():
        v = _smlt_value(engine, CFG.SMLT_COLS[rawcol])
        out[rawcol] = v if v is not None else fillval
        if v is not None:
            src = "측정(탁도·PH·온도)+SMLT(알칼리도·전기전도도)"
    return out[raw_cols], df, src


def current_cluster() -> dict | None:
    """실 DB 최근 10분으로 현재 군집 산출. 데이터 부족/결측이면 None."""
    st = load_state()
    b = st["bundle"]
    pipe, raw_cols, window, label_map = b["pipeline"], b["raw_cols"], int(b["window"]), b["label_map"]
    rows, raw_df, src = read_recent_raw(get_engine(), raw_cols, window)
    if len(rows) < window or not np.isfinite(rows.to_numpy(float)).all():
        return None
    feat = _window_features(rows, raw_cols, window)
    raw_label = int(pipe.predict(feat)[0])
    cid = int(np.asarray(label_map)[raw_label])
    # pseudo-probability: 스케일 공간 중심거리 softmax
    z = pipe[:-1].transform(feat)
    d = np.linalg.norm(pipe[-1].cluster_centers_ - z, axis=1)
    p = np.exp(-d); p = p / p.sum()
    def _last(col):
        try:
            return round(float(pd.to_numeric(raw_df[col]).iloc[-1]), 3)
        except Exception:  # noqa: BLE001
            return None
    return {
        "cluster_id": cid,
        "cluster_label": f"C{cid}",
        "probability": round(float(p[raw_label]), 3),
        "turbidity": _last("INT_TRBD"),
        "ph": _last("INT_PH"),
        "manganese": _last("INT_MANG"),
        "algae": _last("INT_ALGA"),
        "current_dose": _last("INT_PAC"),
        "at": str(raw_df["MSRMT_DT"].iloc[-1]),
        "source": src,
    }


def cluster_definitions() -> list[dict]:
    """군집별 원단위 중심(탁도·PH·온도) + 의미 라벨 — 번들 KMeans에서 역산(csv 미의존)."""
    b = load_state()["bundle"]
    pipe, label_map = b["pipeline"], np.asarray(b["label_map"])
    km = pipe[-1]
    # 스케일 공간 중심 → 원단위 역변환. 특징순서 [c0__mean, c0__slope, c1__mean, ...]
    centers = pipe[:-1].inverse_transform(km.cluster_centers_)
    tb_i, ph_i, tmp_i = 0, 1, 3  # raw_cols 인덱스: 탁도/PH/온도
    defs = []
    for raw_label, c in enumerate(centers):
        cid = int(label_map[raw_label])
        defs.append({
            "cluster_id": cid,
            "meaning": CFG.CLUSTER_MEANINGS.get(cid, ""),
            "turbidity_mean": round(float(c[2 * tb_i]), 3),
            "ph_mean": round(float(c[2 * ph_i]), 3),
            "temp_mean": round(float(c[2 * tmp_i]), 2),
        })
    defs.sort(key=lambda d: d["cluster_id"])
    return defs


def cluster_analysis_payload() -> dict:
    """intake/cluster 엔드포인트용 실데이터 payload. 실패 시 예외(호출부가 mock 폴백)."""
    cur = current_cluster()
    defs = cluster_definitions()
    if cur is None:
        basis = ("실 KMeans(k=3) 군집 — 최근 10분 원수성상 데이터 부족/결측으로 현재 군집 산출 불가. "
                 "군집 정의만 제공.")
        metrics = [{"key": "cluster_id", "value": None, "unit": "-"}]
        points = [{"x": d["turbidity_mean"], "y": d["ph_mean"], "cluster": d["cluster_id"]} for d in defs]
        return {
            "metrics": metrics,
            "chart": {"type": "cluster", "x": "influent_turbidity", "y": "raw_ph",
                      "points": points, "clusters": defs},
            "recommendation": {"value": None, "basis": basis},
        }
    meaning = next((d["meaning"] for d in defs if d["cluster_id"] == cur["cluster_id"]), "")
    basis = (f"실 KMeans(k=3) 군집 — 현재 원수는 C{cur['cluster_id']}({meaning}), "
             f"확률 {cur['probability']}. 소스: {cur['source']} @ {cur['at']}. "
             f"(알칼리도·전기전도도는 DB 계측 부재로 학습중앙값 대체)")
    metrics = [
        {"key": "cluster_id", "value": cur["cluster_id"], "unit": "-"},
        {"key": "cluster_label", "value": cur["cluster_label"], "unit": "-"},
        {"key": "influent_turbidity", "value": cur["turbidity"], "unit": "NTU"},
        {"key": "cluster_probability", "value": cur["probability"], "unit": "-"},
    ]
    for key, unit in (("manganese", "mg/L"), ("algae", "cells/mL"), ("current_dose", "mg/L")):
        if cur.get(key) is not None:
            metrics.append({"key": key, "value": cur[key], "unit": unit})
    points = [{"x": d["turbidity_mean"], "y": d["ph_mean"], "cluster": d["cluster_id"]} for d in defs]
    return {
        "metrics": metrics,
        "chart": {"type": "cluster", "x": "influent_turbidity", "y": "raw_ph",
                  "points": points, "clusters": defs},
        "recommendation": {"value": cur["cluster_id"], "cluster_label": cur["cluster_label"],
                           "basis": basis},
    }
