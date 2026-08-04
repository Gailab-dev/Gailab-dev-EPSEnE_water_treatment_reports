# -*- coding: utf-8 -*-
"""덕남 계측/교반 데이터 로딩 — MSRMT_DT 인덱스 DataFrame 반환."""
import pandas as pd

from . import config as C
from .connection import query


def _where(start, end):
    conds, params = [], {}
    if start is not None:
        conds.append(f"{C.TIME_COL} >= :start")
        params["start"] = str(start)
    if end is not None:
        conds.append(f"{C.TIME_COL} <= :end")
        params["end"] = str(end)
    return (" WHERE " + " AND ".join(conds)) if conds else "", params


def load_measurement(start=None, end=None, rename: str = "kor",
                     cols=None) -> pd.DataFrame:
    """계측 시계열 로드.

    start/end: 'YYYY-MM-DD' 또는 datetime (포함 경계).
    rename: 'kor' 표준 한글명 / 'pipeline' 기존 parquet 컬럼명(겹치는 것만) /
            None 원본 DB 코드.
    cols: 특정 DB 컬럼만 (리스트).
    """
    sel = "*" if cols is None else ", ".join([C.TIME_COL] + list(cols))
    w, params = _where(start, end)
    df = query(f"SELECT {sel} FROM {C.MEASR_TABLE}{w} ORDER BY {C.TIME_COL}", params)
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = df.set_index(C.TIME_COL).sort_index()
    df.index.name = "datetime"
    if rename == "kor":
        df = df.rename(columns=C.COLUMN_MAP)
    elif rename == "pipeline":
        df = df.rename(columns=C.PIPELINE_MAP)
    return df


def load_mixing(start=None, end=None, value: str = "both") -> pd.DataFrame:
    """교반기 지별 G값·RPM 로드.

    value: 'gval' G값만 / 'rpm' RPM만 / 'both' 둘 다.
    컬럼명은 '{지}_{단}_G' / '{지}_{단}_RPM' 형태로 표준화.
    """
    w, params = _where(start, end)
    df = query(f"SELECT * FROM {C.MXC_TABLE}{w} ORDER BY {C.TIME_COL}", params)
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = df.set_index(C.TIME_COL).sort_index()
    df.index.name = "datetime"
    ren = {}
    for c in df.columns:
        if c.startswith("MXC_GVAL_"):
            code = c.replace("MXC_GVAL_", "")
            ren[c] = f"{code[:-1]}_{code[-1]}단_G"
        elif c.startswith("MXC_RPM_"):
            code = c.replace("MXC_RPM_", "")
            ren[c] = f"{code[:-1]}_{code[-1]}단_RPM"
    df = df.rename(columns=ren)
    if value == "gval":
        df = df[[c for c in df.columns if c.endswith("_G")]]
    elif value == "rpm":
        df = df[[c for c in df.columns if c.endswith("_RPM")]]
    return df


def latest(n: int = 10, table: str = "measr", **kw) -> pd.DataFrame:
    """최근 n행 (표시·점검용)."""
    tbl = C.MEASR_TABLE if table == "measr" else C.MXC_TABLE
    df = query(f"SELECT * FROM {tbl} ORDER BY {C.TIME_COL} DESC LIMIT {int(n)}")
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = df.set_index(C.TIME_COL).sort_index()
    if table == "measr" and kw.get("rename", "kor") == "kor":
        df = df.rename(columns=C.COLUMN_MAP)
    return df


def time_range(table: str = "measr") -> tuple:
    """(최소일시, 최대일시, 행수)."""
    tbl = C.MEASR_TABLE if table == "measr" else C.MXC_TABLE
    r = query(f"SELECT MIN({C.TIME_COL}) mn, MAX({C.TIME_COL}) mx, COUNT(*) n FROM {tbl}")
    return (pd.to_datetime(r.mn[0]), pd.to_datetime(r.mx[0]), int(r.n[0]))
