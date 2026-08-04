# -*- coding: utf-8 -*-
"""검증: §16.1 수식 단위시험(실패 시 assert 중단) + §16.2 백테스트 통계."""
import numpy as np
import pandas as pd

from . import alignment
from . import config as C
from . import constraints, flow, recommend, reasons, validity

_RESULTS: list[dict] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append({"항목": name, "결과": "통과" if cond else "실패", "비고": detail})
    assert cond, f"§16.1 단위시험 실패: {name} — {detail}"


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="1min")


def validate_formulas() -> pd.DataFrame:
    """§16.1 전 항목 실행. 하나라도 실패하면 assert로 즉시 중단."""
    _RESULTS.clear()
    # ① 낙차부 체류시간·G
    q = pd.Series([C.FLOW_AVG_M3H], index=_idx(1))
    td = float(flow.t_drop_s(q).iloc[0])
    _check("낙차부 체류시간(평균유량)=99.1s", abs(td - C.T_DROP_AVG_S) < 1e-9,
           f"{td:.4f}s")
    gd = float(recommend.g_drop(pd.Series([20.0], index=q.index),
                                flow.t_drop_s(q)).iloc[0])
    _check("낙차부 G(20℃·평균유량)≈209.9", abs(gd - C.G_DROP_REF) / C.G_DROP_REF < 0.02,
           f"{gd:.1f}s⁻¹")

    # ② 표 2.3-13 격자점 정확 재현
    temps = pd.Series(C.STD_RPM_TEMPS, index=_idx(6))
    std = recommend.interp_std_rpm(temps)
    for i, k in enumerate((1, 2, 3)):
        err = np.abs(std.iloc[:, i].to_numpy() - np.asarray(C.STD_RPM_TABLE[k])).max()
        _check(f"표 2.3-13 재현({k}단)", err < 1e-9, f"max err {err:.1e}")

    # ③ 중간 수온(17℃) 보간값이 15·20℃ 사이
    m = recommend.interp_std_rpm(pd.Series([17.0], index=_idx(1)))
    for i, k in enumerate((1, 2, 3)):
        lo = min(C.STD_RPM_TABLE[k][2], C.STD_RPM_TABLE[k][3])
        hi = max(C.STD_RPM_TABLE[k][2], C.STD_RPM_TABLE[k][3])
        v = float(m.iloc[0, i])
        _check(f"17℃ 보간({k}단)", lo < v < hi, f"{v:.3f} ∈ ({lo},{hi})")

    # ④ 표준후보 보정계수=1
    rec = recommend.correct_rpm(std, C.G_CANDIDATES["표준"])
    _check("표준후보 보정계수=1",
           np.allclose(rec.to_numpy(), std.to_numpy(), atol=1e-12), "")

    # ⑤ G 단조증가 → RPM 단조증가 (3단 G 40→60 스윕 + 강화>표준)
    sweep = [float(recommend.correct_rpm(std.iloc[[0]], (50, 30, g)).iloc[0, 2])
             for g in np.arange(40, 61, 5)]
    _check("G 스윕 RPM 단조증가", bool(np.all(np.diff(sweep) > 0)), "")
    strong = recommend.correct_rpm(std, C.G_CANDIDATES["강화"])
    _check("강화후보 3단 > 표준후보 3단",
           bool(strong.iloc[:, 2].gt(rec.iloc[:, 2]).all()), "")

    # ⑥ 합성 점감·한계 위반 차단 + HOLD 우선순위
    bad = pd.DataFrame({"rpm_rec_1": [1.2, 0.3], "rpm_rec_2": [1.5, 1.2],
                        "rpm_rec_3": [0.8, 0.6]}, index=_idx(2))
    safe, limviol = constraints.apply_limits(bad)
    _check("하한 위반 감지", bool(limviol.iloc[1]), "1단 0.3rpm < 하한 1.0")
    _check("하한 클립", float(safe.iloc[1, 0]) == C.RPM_LIMITS[1][0], "")
    tap = constraints.taper_violation(safe)
    _check("점감 위반 감지", bool(tap.iloc[0]), "1.2 < 1.5")
    code, bits = reasons.assemble({
        "HOLD_TAPER_VIOLATION": tap, "HOLD_LIMIT": limviol,
        "WARN_TEMP_CLAMP": pd.Series(True, index=bad.index),
        "OK_TEMP_INTERP": pd.Series(True, index=bad.index)})
    _check("HOLD가 WARN·OK 우선", code.iloc[0] == "HOLD_TAPER_VIOLATION"
           and code.iloc[1] == "HOLD_LIMIT", str(code.tolist()))
    _check("비트마스크 복수기록",
           bool((bits & reasons.REASON_BITS["WARN_TEMP_CLAMP"]).gt(0).all()), "")

    # ⑦ 수온 3℃ → 5℃ 클램프, 36℃ → HOLD
    t3 = recommend.interp_std_rpm(pd.Series([3.0], index=_idx(1)))
    _check("3℃ → 5℃ 값 클램프", float(t3.iloc[0, 0]) == C.STD_RPM_TABLE[1][0], "")
    df36 = pd.DataFrame({c: [np.nan] for c in C.LOAD_COLS}, index=_idx(1))
    df36[C.COL_TEMP] = 36.0
    df36[C.COL_FLOW] = C.FLOW_AVG_M3H
    fl36 = pd.DataFrame(False, index=df36.index, columns=C.LOAD_COLS)
    hm = validity.hold_masks(df36, fl36, df36[C.COL_FLOW])
    _check("36℃ → 범위외 HOLD", bool(hm["temp_out_of_range"].iloc[0]), "")

    # ⑧ 상류정렬: 일정유량 Qavg → 정확히 11분(11.3 반올림) 시프트 + 앞머리 NaN
    n = 30
    imp = pd.DataFrame({"x": np.zeros(n)}, index=_idx(n))
    imp.iloc[5, 0] = 1.0
    lag = flow.upstream_lag_min(pd.Series(C.FLOW_AVG_M3H, index=imp.index))
    shifted = alignment.shift_by_lag(imp, lag)
    _check("상류시프트 정확성", float(shifted.iloc[5 + 11, 0]) == 1.0
           and float(shifted.iloc[5 + 10, 0]) == 0.0, "임펄스 11분 이동")
    _check("앞머리 NaN", bool(shifted.iloc[:11, 0].isna().all()), "")

    # ⑨ K0 검산 정합 (5~30℃ 전 구간 편차 < 문턱)
    tgrid = pd.Series(np.arange(5.0, 30.01, 0.5), index=_idx(51))
    stdg = recommend.interp_std_rpm(tgrid)
    ck = recommend.crosscheck_k0(
        stdg.rename(columns=dict(zip(stdg.columns,
                                     ("rpm_rec_1", "rpm_rec_2", "rpm_rec_3")))),
        tgrid)
    dmax = float(ck["k0_dev_max"].max())
    _check("K0 검산 편차<3%", dmax < C.MODEL_DIVERGENCE_WARN, f"max {dmax:.2%}")

    # ⑩ Gt: 평균유량·12지 균등·표준 G → ≈1.1×10⁵, 범위 내
    qb = flow.basin_flows(pd.Series([C.FLOW_AVG_M3H], index=_idx(1)),
                          flow.equal_weights(C.N_BASINS_TOTAL))
    gt = constraints.gt_by_basin(C.G_BASE, flow.stage_hrt_s(qb))
    gtv = float(gt.iloc[0, 0])
    _check("Gt(평균유량·12지·표준G)≈1.1e5",
           abs(gtv - C.GT_CENTER) / C.GT_CENTER < 0.10, f"{gtv:.3e}")
    _check("Gt 승인범위 내", C.GT_RANGE[0] <= gtv <= C.GT_RANGE[1], "")

    # ⑪ 연속결측 run-length: 4분 미발동 / 5·6분 발동
    miss = pd.Series(False, index=_idx(30))
    miss.iloc[2:6] = True    # 4분
    miss.iloc[10:15] = True  # 5분
    miss.iloc[20:26] = True  # 6분
    rl = validity.run_length_mask(miss)
    _check("4분 결측 미발동", not rl.iloc[2:6].any(), "")
    _check("5분·6분 결측 발동", bool(rl.iloc[10:15].all() and rl.iloc[20:26].all()), "")

    out = pd.DataFrame(_RESULTS)
    print(f"§16.1 단위시험 {len(out)}건 전부 통과")
    return out


def backtest_stats(out: pd.DataFrame, elapsed_s: float) -> pd.DataFrame:
    """§16.2 전기간 재현 통계. 위반 0건은 assert."""
    n = len(out)
    hold = out["reason_code"].str.startswith("HOLD")
    required_ok = out["q_m3h"].notna() & out["수온_정렬"].notna()
    rows = [
        ("전체 행수", f"{n:,}"),
        ("백테스트 소요시간", f"{elapsed_s:.1f}초"),
        ("가용률(전체행 분모)", f"{(~hold).mean():.2%}"),
        ("가용률(필수입력 존재행 분모)",
         f"{(~hold[required_ok]).mean():.2%} (행수 {int(required_ok.sum()):,})"),
    ]

    # 위반 0건 assert: HOLD 행 추천값 NaN, 산출 rpm_safe 점감·한계 준수
    safe_cols = ["rpm_safe_1", "rpm_safe_2", "rpm_safe_3"]
    leak = out.loc[hold, safe_cols].notna().any(axis=1).sum()
    assert leak == 0, f"HOLD 행에 추천값 잔존 {leak}건"
    s = out[safe_cols].dropna()
    taper_bad = int((~(s["rpm_safe_1"].ge(s["rpm_safe_2"])
                      & s["rpm_safe_2"].ge(s["rpm_safe_3"]))).sum())
    lim_bad = int(sum((s[f"rpm_safe_{k}"].lt(C.RPM_LIMITS[k][0])
                       | s[f"rpm_safe_{k}"].gt(C.RPM_LIMITS[k][1])).sum()
                      for k in (1, 2, 3)))
    assert taper_bad == 0 and lim_bad == 0, f"점감 {taper_bad}·한계 {lim_bad}건"
    rows += [("산출 추천값 점감 위반", "0건"), ("산출 추천값 한계 위반", "0건"),
             ("HOLD 행 추천값 누출", "0건")]

    # 계절 정합: 저수온기 추천 RPM > 고수온기 (표 2.3-13 온도 의존성)
    w = out.loc[out["수온_정렬"] < 10, "rpm_safe_1"].mean()
    sm = out.loc[out["수온_정렬"] > 20, "rpm_safe_1"].mean()
    assert w > sm, f"계절 정합 실패: 저수온 {w:.2f} ≤ 고수온 {sm:.2f}"
    rows.append(("계절 정합(1단 저수온 vs 고수온)", f"{w:.2f} > {sm:.2f} rpm"))

    # 층화 집계: 고탁도·저수온·고유량
    tb = out["원수탁도_정렬"]
    strata = {"고탁도(상위5%)": tb.ge(tb.quantile(0.95)),
              "저수온(<8℃)": out["수온_정렬"].lt(8),
              "고유량(상위5%)": out["q_m3h"].ge(out["q_m3h"].quantile(0.95))}
    for name, m in strata.items():
        m = m.fillna(False)
        if m.any():
            rows.append((f"{name} 가용률", f"{(~hold[m]).mean():.2%} "
                         f"(행수 {int(m.sum()):,})"))
    return pd.DataFrame(rows, columns=["항목", "값"])
