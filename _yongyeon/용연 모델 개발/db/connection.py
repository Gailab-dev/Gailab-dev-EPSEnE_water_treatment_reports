# -*- coding: utf-8 -*-
"""step_aiwtp MySQL 접속 (덕남·용연 공통 로직).

자격증명은 **환경변수 WTP_DB_* 로만** 주입한다(소스에 기본값 두지 않음).
미설정 시 engine()이 RuntimeError. 예: WTP_DB_HOST/PORT/USER/PASSWORD/NAME.
"""
import os
import warnings

import pandas as pd
from sqlalchemy import create_engine, text

DB = dict(
    host=os.environ.get("WTP_DB_HOST", ""),
    port=int(os.environ.get("WTP_DB_PORT", "44306")),
    user=os.environ.get("WTP_DB_USER", ""),
    password=os.environ.get("WTP_DB_PASSWORD", ""),
    database=os.environ.get("WTP_DB_NAME", ""),
)

_ENGINE = None


def engine():
    """SQLAlchemy 엔진 (프로세스 내 싱글턴)."""
    global _ENGINE
    if _ENGINE is None:
        if not (DB["host"] and DB["user"] and DB["password"] and DB["database"]):
            raise RuntimeError(
                "WTP_DB_* 환경변수 미설정 — WTP_DB_HOST/USER/PASSWORD/NAME 을 주입하세요")
        url = (f"mysql+pymysql://{DB['user']}:{DB['password']}@"
               f"{DB['host']}:{DB['port']}/{DB['database']}?charset=utf8mb4")
        _ENGINE = create_engine(url, pool_pre_ping=True,
                                connect_args={"connect_timeout": 8})
    return _ENGINE


def query(sql: str, params: dict = None) -> pd.DataFrame:
    """SELECT → pandas DataFrame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})


def ping() -> str:
    """연결 확인 — 서버 버전·DB·사용자 문자열 반환."""
    r = query("SELECT VERSION() AS v, DATABASE() AS db, CURRENT_USER() AS u")
    return f"{r.v[0]} / {r.db[0]} / {r.u[0]}"
