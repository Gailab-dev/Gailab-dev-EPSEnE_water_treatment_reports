# -*- coding: utf-8 -*-
"""용연정수장 DB 접속 모듈 (step_aiwtp).

사용 예:
    from db import ping, load_measurement, load_mixing, latest, time_range
    print(ping())                                   # 연결 확인
    df = load_measurement("2026-07-08", "2026-07-15")   # 계측(표준 한글명)
    dfp = load_measurement(rename="pipeline")            # 기존 parquet 컬럼명
    g = load_mixing(value="gval")                        # 지별 G값
"""
from . import config
from .connection import engine, ping, query
from .loader import (latest, load_measurement, load_mixing, time_range)

__all__ = ["config", "engine", "ping", "query",
           "load_measurement", "load_mixing", "latest", "time_range"]
