# -*- coding: utf-8 -*-
"""데이터 유효성 검사기 (§4.3) — HOLD/WARN bool 마스크, 전부 벡터화.

HOLD: 추천 중지(추천 RPM NaN 마스킹). WARN: 추천은 유지하되 사유코드에 기록.
채움플래그 True(전처리 시 채워진 값)는 계산에는 쓰되 연속결측 판정의 결측으로
집계한다 — 5분 이상 채움이 이어지면 원천 데이터 부재로 보고 HOLD.
"""
import pandas as pd

from . import config as C


def run_length_mask(missing: pd.Series, min_run: int = C.MISS_RUN_MIN) -> pd.Series:
    """연속 True 구간 길이가 min_run 이상인 행만 True (run-length 벡터화)."""
    grp = (missing != missing.shift()).cumsum()
    runlen = missing.groupby(grp).transform("size")
    return missing & (runlen >= min_run)


def hold_masks(df: pd.DataFrame, flags: pd.DataFrame,
               q_m3h: pd.Series) -> pd.DataFrame:
    """추천 중지(HOLD) 사유별 bool 마스크 (§4.3)."""
    miss_flow = df[C.COL_FLOW].isna() | flags[C.COL_FLOW]
    miss_temp = df[C.COL_TEMP].isna() | flags[C.COL_TEMP]
    temp = df[C.COL_TEMP]
    return pd.DataFrame({
        # 5분 이상 연속 결측(채움 포함) — 단기 결측은 전처리 채움값으로 계속
        "miss_flow": run_length_mask(miss_flow),
        "miss_temp": run_length_mask(miss_temp),
        "flow_zero": q_m3h.le(C.FLOW_ZERO_EPS).fillna(False),
        # 단위오인·계측이상 감지 밴드 밖 (§6.1 안전조항의 행 단위 버전)
        "flow_implausible": (~q_m3h.between(*C.FLOW_PLAUSIBLE_M3H)).fillna(False)
                            & q_m3h.notna() & q_m3h.gt(C.FLOW_ZERO_EPS),
        # 수온 유효범위 0~35℃ 이탈 (§7; NaN은 miss_temp가 담당)
        "temp_out_of_range": (~temp.between(*C.TEMP_VALID_C)) & temp.notna(),
    }, index=df.index)


def warn_masks(df: pd.DataFrame, temp_floc: pd.Series) -> pd.DataFrame:
    """경고(WARN) 사유별 bool 마스크 — 추천은 계속 산출."""
    pac_on = df[C.COL_PAC_SV].gt(0).fillna(False)
    pacs2_on = df[C.COL_PACS2_SV].gt(0).fillna(False)
    return pd.DataFrame({
        # 5~30℃ 표준표 범위 밖 → 경계값 클램프 적용 (§7, §15.3)
        "temp_clamp": (~temp_floc.between(*C.TEMP_STD_C)) & temp_floc.notna(),
        # PAC·PACS2 동시 활성 — 합산 금지, Letterman G 미산정 (§8.3)
        "chem_ambiguous": pac_on & pacs2_on,
        # 지별 유량은 실측이 아닌 가중치 추정 (§6.2 우선순위 3·4단계)
        "flow_est": pd.Series(True, index=df.index),
        # 인버터·RPM 태그 부재 — 전 지 화면표출만 가능 (§15.3)
        "no_vfd": pd.Series(True, index=df.index),
    }, index=df.index)
