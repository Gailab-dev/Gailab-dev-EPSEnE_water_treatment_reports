# -*- coding: utf-8 -*-
"""사유코드(§15.3) — uint32 비트마스크 + 우선순위 단일코드 조립, 벡터화.

WARN_MODEL_DIVERGENCE는 §15.3에 없는 확장코드(룩업보간 vs K0 검산 편차 경보).
"""
import numpy as np
import pandas as pd

from . import config as C

REASON_BITS = {
    "HOLD_DATA_MISSING": 1 << 0,      # 필수 데이터 결측·유량0·범위외 (§4.3)
    "HOLD_LIMIT": 1 << 1,             # RPM·Gt 한계 위반 (§10.4, §9.2)
    "HOLD_TAPER_VIOLATION": 1 << 2,   # 점감조건 위반 (§9.1)
    "HOLD_VFD_NOT_READY": 1 << 3,     # 인버터 제어 불가 (태그 연결 후 발동)
    "WARN_TEMP_CLAMP": 1 << 8,        # 5~30℃ 밖 표준값 고정 (§7)
    "WARN_FLOW_EST": 1 << 9,          # 지별 유량 추정값 사용 (§6.2)
    "WARN_NO_ACTIVE_CHEM": 1 << 10,   # 약품 미확정 — Letterman G 미산정 (§8.3)
    "WARN_MODEL_DIVERGENCE": 1 << 11, # 확장: 룩업 vs K0 편차 > 문턱
    "DISPLAY_ONLY_NO_VFD": 1 << 16,   # 인버터 미설치·태그 부재 — 화면표출만
    "OK_TEMP_INTERP": 1 << 24,        # 수온보간 적용
    "OK_STANDARD": 1 << 25,           # 표준 격자점 RPM 그대로
}

# 단일 표시코드 우선순위 (HOLD > WARN > 표시제한 > OK)
PRIORITY = ["HOLD_DATA_MISSING", "HOLD_LIMIT", "HOLD_TAPER_VIOLATION",
            "HOLD_VFD_NOT_READY",
            "WARN_TEMP_CLAMP", "WARN_MODEL_DIVERGENCE", "WARN_NO_ACTIVE_CHEM",
            "DISPLAY_ONLY_NO_VFD", "OK_TEMP_INTERP", "OK_STANDARD"]

# 현 단계에서 전 행 참인 정보성 코드 — 비트마스크에는 기록하되 primary에서 제외
# (제외하지 않으면 모든 정상 행의 표시코드가 이 코드로 덮여 OK/WARN이 가려짐)
INFO_CODES = {"WARN_FLOW_EST", "DISPLAY_ONLY_NO_VFD"}

HOLD_MASK_BITS = sum(REASON_BITS[c] for c in REASON_BITS if c.startswith("HOLD"))


def assemble(masks: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    """코드별 bool 마스크 → (primary 단일코드, uint32 비트마스크).

    masks에 없는 코드는 미발동 취급. 루프는 코드 10여 개 대상이며 각 반복은
    벡터 연산 (행 루프 아님).
    """
    index = next(iter(masks.values())).index
    zeros = np.zeros(len(index), dtype=bool)
    as_np = {code: (m.fillna(False).to_numpy(dtype=bool)
                    if (m := masks.get(code)) is not None else zeros)
             for code in REASON_BITS}
    bitmask = np.zeros(len(index), dtype=np.uint32)
    for code, bit in REASON_BITS.items():   # 비트마스크는 전체 코드 기록
        bitmask |= np.where(as_np[code], bit, 0).astype(np.uint32)
    conds = [as_np[c] for c in PRIORITY if c not in INFO_CODES]
    labels = [c for c in PRIORITY if c not in INFO_CODES]
    primary = np.select(conds, labels, default="OK_STANDARD")
    return (pd.Series(primary, index=index, name="reason_code"),
            pd.Series(bitmask, index=index, name="reason_bitmask"))


def is_hold(bitmask: pd.Series) -> pd.Series:
    """HOLD_* 비트가 하나라도 서 있으면 True → 추천 RPM NaN 마스킹 대상."""
    return (bitmask & HOLD_MASK_BITS).gt(0).rename("hold")


def control_available(bitmask: pd.Series, vfd_tag_present: bool = False) -> pd.Series:
    """제어명령 생성 가능 여부 (§6.1, §14.1).

    유량단위 미확정(FLOW_UNIT_CONFIRMED=False) 또는 인버터 태그 부재 시
    전 행 False. 태그 연결·단위 확정 후에만 HOLD 아님 조건으로 활성화.
    """
    if not (C.FLOW_UNIT_CONFIRMED and vfd_tag_present):
        return pd.Series(False, index=bitmask.index, name="control_available")
    return (~is_hold(bitmask)).rename("control_available")
