# -*- coding: utf-8 -*-
"""피처 생성 (§6) — dose_compare/features.py 패턴 이식·확장. 전부 t 이전 정보만.

- 연속신호 14종: 현재값 + lag(5/15/30/60) + 변화량(5/15/60) + 변화율 + 이동통계(15/60/180)
- 시간(§6): hour/month/dow sin·cos / 소독설정변경 / 채움플래그
- 주입 운전상태: 직전 SV·변경 후 경과·변경폭 (proxy — §3.3 실투입 라벨 미확보)
- T_outlet(t) 유량보정값, 탁도 구간·급상승 상태
- sed-AR(과제B 'AR포함' 토글): 현재·과거 침전지 탁도 — t 시점에 알 수 있는 값(§5.3 허용).
  'AR제외'는 순수 feedforward(원수+유량+주입) 구성.
변화율은 diff/k(분당) 대신 상대변화 diff/(|lag값|+ε) — diff와의 완전공선 회피.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import residence as R


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def continuous_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """피처 원천 연속신호 {짧은태그: Series}. 주입량합은 A+B 펌프 합산(교대 대응)."""
    out = {}
    for col in C.RAW5 + [C.FLOW_COL] + C.LEVEL_COLS + C.DOSE_FLOW + C.DOSE_SV:
        out[col.split(".")[-1]] = _num(df, col)
    for tag, cols in C.DOSE_PUMPS.items():
        out[tag] = _num(df, cols[0]).fillna(0) + _num(df, cols[1]).fillna(0)
    return out


def build_features(df: pd.DataFrame, flags: pd.DataFrame | None,
                   stride_idx: pd.DatetimeIndex):
    """(stride 학습행 피처 DataFrame, base_feats, sed_ar_feats) 반환.

    전 컬럼을 1분 전체격자에서 계산 후 stride 행만 저장(메모리 절약, float32).
    """
    cont = continuous_columns(df)
    store: dict[str, pd.Series] = {}
    base_feats: list[str] = []

    def keep(name: str, s: pd.Series, is_feat: bool = True):
        store[name] = s.reindex(stride_idx).astype(np.float32)
        if is_feat:
            base_feats.append(name)

    for tag, s in cont.items():
        keep(tag, s)
        for k in C.LAGS:
            keep(f"{tag}__lag{k}", s.shift(k))
        for k in C.DIFFS:
            d = s - s.shift(k)
            keep(f"{tag}__d{k}", d)
            keep(f"{tag}__r{k}", d / (s.shift(k).abs() + 1e-6))
        for w in C.ROLLS:
            r = s.rolling(w, min_periods=max(3, w // 3))
            keep(f"{tag}__ma{w}", r.mean())
            keep(f"{tag}__min{w}", r.min())
            keep(f"{tag}__max{w}", r.max())
            keep(f"{tag}__sd{w}", r.std())

    # 시간 피처
    idx = df.index
    for name, period, val in [("hour", 24, idx.hour), ("month", 12, idx.month),
                              ("dow", 7, idx.dayofweek)]:
        keep(f"{name}_sin", pd.Series(np.sin(2 * np.pi * val / period), index=idx))
        keep(f"{name}_cos", pd.Series(np.cos(2 * np.pi * val / period), index=idx))

    # 소독 설정 변경(직전 대비) — 과거 정보
    chg = pd.Series(0.0, index=idx)
    for col in C.DISINFECT_COLS:
        if col in df.columns:
            chg = chg + (_num(df, col).diff().abs() > 1e-9).astype(float)
    keep("소독설정변경", (chg > 0).astype(float))

    # 채움 플래그(원수·유량이 채워진 값인지)
    if flags is not None:
        fl = flags.reindex(idx)
        for col in C.RAW5 + [C.FLOW_COL]:
            if col in fl.columns:
                keep(col.split(".")[-1] + "__filled", fl[col].astype(float))

    # 주입 운전상태 AR(§6: 직전 주입률·변경 후 경과시간·변경폭) — SV(계단형) 기준
    sv = _num(df, C.AR_SV_COL)
    prev = sv.shift(1)
    keep("SV_prev", prev)
    changed = (prev != prev.shift(1)) & prev.notna()
    grp = changed.cumsum()
    keep("SV_since_change", pd.Series(
        df.groupby(grp).cumcount().to_numpy(float), index=idx))
    last_mag = prev.diff().where(changed).groupby(grp).transform("first")
    keep("SV_change_mag", last_mag.fillna(0.0))

    # T_outlet 동적값 (§6)
    tres = R.residence_series(df[C.FLOW_COL])
    keep("T_mean", tres["T_mean"])
    keep("T_peak", tres["T_peak"])

    # 탁도 구간·급상승 상태 (§6)
    tb = cont[C.TURBIDITY_COL.split(".")[-1]]
    bins = pd.cut(tb, bins=C.TB_BINS, labels=C.TB_BIN_LABELS)
    for lab in C.TB_BIN_LABELS:
        keep(f"탁도구간_{lab}", (bins == lab).astype(float))
    keep("탁도급상승", ((tb - tb.shift(60)) > C.SURGE_D60_NTU).astype(float))

    # sed-AR (과제B AR포함 토글): 현재·과거 침전지 탁도 + 1h 평활(persistence 원천)
    sed_ar_feats: list[str] = []
    for b, col in enumerate(C.SED_TB, start=1):
        s = _num(df, col)
        sm = s.rolling(C.TARGET_SMOOTH_MIN, min_periods=10).mean()
        for name, series in [(f"sed{b}_tb", s), (f"sed{b}_sm", sm),
                             (f"sed{b}_lag30", s.shift(30)), (f"sed{b}_lag60", s.shift(60)),
                             (f"sed{b}_d60", s - s.shift(60))]:
            keep(name, series)
            base_feats.remove(name)          # base에서 빼고 sed-AR 그룹으로
            sed_ar_feats.append(name)

    table = pd.DataFrame(store, index=stride_idx)
    return table, base_feats, sed_ar_feats


def seq_base_feats(base_feats: list[str]) -> list[str]:
    """시퀀스 모델용 베이스 피처 — 파생(lag/diff/rate/rolling) 제외(창이 시계열 문맥 제공)."""
    return [f for f in base_feats
            if not any(s in f for s in ("__lag", "__d", "__r", "__ma",
                                        "__min", "__max", "__sd"))]
