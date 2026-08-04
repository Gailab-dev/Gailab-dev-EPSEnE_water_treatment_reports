# -*- coding: utf-8 -*-
"""덕남 혼화·플록형성 G/RPM 규칙기반 추천엔진 — 전기간 백테스트 실행.

혼화응집공정 설계문서 §5~12 구현: 단위시험(§16.1) → 전기간 추천 산출(§16.2)
→ 침전수 탁도 시간창 비교·사후평가(§12) → 산출물·시각화 저장.

실행: python run_floc_rpm.py   (06_floc_rpm 디렉터리에서)
"""
import time

import numpy as np
import pandas as pd

from floc_rpm import alignment, checks, constraints, evaluate, flow
from floc_rpm import config as C
from floc_rpm import io_utils, reasons, recommend, validity, visualization

ALIGN_NAMES = {C.COL_TEMP: "수온_정렬", C.COL_TB: "원수탁도_정렬",
               C.COL_PH: "원수pH_정렬", C.COL_AL: "알칼리도_정렬",
               C.COL_PAC_SV: "PAC_SV_정렬", C.COL_PACS2_SV: "PACS2_SV_정렬"}


def build_output(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                 pd.DataFrame]:
    """§11.1 연산 파이프라인의 전기간 벡터화 실행 → (출력, 지별 Gt)."""
    # 1~4) 유량 정규화·상류지연·시간정렬
    q = flow.normalize_flow(df[C.COL_FLOW])
    med = float(q.median())
    if not (C.FLOW_PLAUSIBLE_M3H[0] <= med <= C.FLOW_PLAUSIBLE_M3H[1]):
        raise RuntimeError(
            f"유량 중앙값 {med:,.0f}이 {C.FLOW_UNIT} 타당범위 밖 — "
            f"FT101 단위(config.FLOW_UNIT) 재확인 필요 (§6.1)")
    lag = flow.upstream_lag_min(q)
    aligned = alignment.shift_by_lag(df[C.ALIGN_COLS], lag) \
                       .rename(columns=ALIGN_NAMES)
    temp_floc = aligned["수온_정렬"]

    # 5~6) 체류시간·물성 → 7) 낙차부 G (현재 착수정 수온, §8)
    t_drop = flow.t_drop_s(q)
    g_drop = recommend.g_drop(df[C.COL_TEMP], t_drop)

    # 9~10) 표준 RPM 보간 + 후보별 G 보정 (§10.1~10.2)
    rpm_std = recommend.interp_std_rpm(temp_floc)
    rec_std = recommend.correct_rpm(rpm_std, C.G_CANDIDATES["표준"])
    ck = recommend.crosscheck_k0(rec_std, temp_floc, C.G_CANDIDATES["표준"])

    # 11) 제약검사 (§10.4): 상하한 클립 → 점감 재검사
    rpm_safe, lim_viol = constraints.apply_limits(rec_std)
    taper = constraints.taper_violation(rpm_safe)

    # 완화·강화 후보는 침전수 사후평가용 저장만 (§9.3)
    extra = {}
    for name in ("완화", "강화"):
        rec_c = recommend.correct_rpm(rpm_std, C.G_CANDIDATES[name])
        safe_c, _ = constraints.apply_limits(rec_c)
        for i, k in enumerate((1, 2, 3)):
            extra[f"{name}_rpm_{k}"] = safe_c.iloc[:, i]

    # 8) Gt — 실측 유량비 8지 배분 (§6.2, §9.2)
    q_basin = flow.basin_flows(q)
    gt_basins = constraints.gt_by_basin(C.G_BASE, flow.stage_hrt_s(q_basin))
    gt_viol = constraints.gt_violation(gt_basins)

    # 12) 사유코드 조립 (§15.3)
    holds = validity.hold_masks(df, flags, q)
    warns = validity.warn_masks(df, temp_floc)
    data_missing = holds.any(axis=1) | temp_floc.isna()
    grid_pt = pd.Series(
        np.isclose(temp_floc.to_numpy(dtype=float)[:, None],
                   np.asarray(C.STD_RPM_TEMPS), atol=0.05).any(axis=1),
        index=temp_floc.index).fillna(False)
    normal = ~data_missing & ~lim_viol & ~gt_viol & ~taper & temp_floc.notna()
    code, bits = reasons.assemble({
        "HOLD_DATA_MISSING": data_missing,
        "HOLD_LIMIT": lim_viol | gt_viol,
        "HOLD_TAPER_VIOLATION": taper,
        "WARN_TEMP_CLAMP": warns["temp_clamp"],
        "WARN_MODEL_DIVERGENCE": ck["warn_divergence"],
        "WARN_NO_ACTIVE_CHEM": warns["chem_ambiguous"],
        "WARN_FLOW_EST": warns["flow_est"],
        "DISPLAY_ONLY_NO_VFD": warns["no_vfd"],
        "OK_TEMP_INTERP": normal & ~grid_pt,
        "OK_STANDARD": normal & grid_pt,
    })
    hold = reasons.is_hold(bits)
    rpm_safe = rpm_safe.mask(hold, np.nan)   # HOLD 행 추천 중지 (§4.3)

    out = pd.concat(
        [q.rename("q_m3h"), lag.rename("lag_min"), t_drop.rename("t_drop_s"),
         g_drop, aligned, rpm_std, rec_std, rpm_safe,
         ck[["rpm_phys_1", "rpm_phys_2", "rpm_phys_3", "k0_dev_max"]],
         pd.DataFrame(extra, index=df.index),
         gt_basins.min(axis=1).rename("gt_min"),
         gt_basins.max(axis=1).rename("gt_max"),
         code, bits, reasons.control_available(bits)], axis=1)
    return out, gt_basins


def reason_stats(out: pd.DataFrame) -> pd.DataFrame:
    """사유코드별 발생행수·비율·최장 연속구간(분) — 비트마스크 기준."""
    rows = []
    for code_name, bit in reasons.REASON_BITS.items():
        m = (out["reason_bitmask"] & np.uint32(bit)).gt(0)
        if not m.any():
            rows.append({"사유코드": code_name, "발생행수": 0, "비율": "0.00%",
                         "최장연속(분)": 0})
            continue
        grp = (m != m.shift()).cumsum()
        longest = int(m.groupby(grp).sum().max())
        rows.append({"사유코드": code_name, "발생행수": int(m.sum()),
                     "비율": f"{m.mean():.2%}", "최장연속(분)": longest})
    return pd.DataFrame(rows)


def interp_vs_k0_table() -> pd.DataFrame:
    """0~35℃ 0.5℃ 스텝: 룩업보간 vs K0 검산 조견표 (§10.1 vs §10.3)."""
    t = pd.Series(np.arange(0.0, 35.01, 0.5),
                  index=pd.RangeIndex(71).map(lambda i: pd.Timestamp("2025-01-01")
                                              + pd.Timedelta(minutes=i)))
    std = recommend.interp_std_rpm(t)
    ck = recommend.crosscheck_k0(
        std.set_axis(["rpm_rec_1", "rpm_rec_2", "rpm_rec_3"], axis=1), t)
    tab = pd.DataFrame({"수온(℃)": t.to_numpy()})
    for k in (1, 2, 3):
        tab[f"{k}단 보간rpm"] = std[f"rpm_std_{k}"].round(3).to_numpy()
        tab[f"{k}단 K0검산rpm"] = ck[f"rpm_phys_{k}"].round(3).to_numpy()
    tab["최대편차(%)"] = (ck["k0_dev_max"] * 100).round(2).to_numpy()
    return tab


def gt_check_table(q: pd.Series) -> pd.DataFrame:
    """지별 Gt 점검 — 실측 가중배분 vs 균등배분 시나리오 (§6.2, §9.2)."""
    rows = []
    scen = {"실측가중(8지)": None, "균등(8지)": flow.equal_weights(8),
            "균등(12지)": flow.equal_weights(12)}
    for name, w in scen.items():
        gt = constraints.gt_by_basin(
            C.G_BASE, flow.stage_hrt_s(flow.basin_flows(q, w)))
        for basin in gt.columns:
            s = gt[basin].dropna()
            oob = ((s < C.GT_RANGE[0]) | (s > C.GT_RANGE[1])).mean()
            rows.append({"시나리오": name, "지": basin,
                         "Gt 5%": f"{s.quantile(0.05):.2e}",
                         "Gt 중앙값": f"{s.median():.2e}",
                         "Gt 95%": f"{s.quantile(0.95):.2e}",
                         "범위이탈률": f"{oob:.2%}"})
    return pd.DataFrame(rows)


def run() -> None:
    io_utils.setup_output()
    unit_results = checks.validate_formulas()          # §16.1 — 실패 시 중단

    t0 = time.time()
    df, flags = io_utils.load_grid()
    out, gt_basins = build_output(df, flags)
    elapsed = time.time() - t0
    print(f"전기간 백테스트 {len(out):,}행, {elapsed:.1f}초")

    # ── §16.2 통계 (위반 0건 assert 포함) ──
    bt = checks.backtest_stats(out, elapsed)
    print(bt.to_string(index=False))

    # ── 산출물 저장 ──
    out.to_parquet(C.OUTPUT_DIR / "추천_전기간_시계열.parquet")
    print("저장: 추천_전기간_시계열.parquet")

    daily = out.select_dtypes("number").resample("1D").median()
    daily["HOLD비율"] = out["reason_code"].str.startswith("HOLD") \
                                          .resample("1D").mean().round(4)
    io_utils.save_csv(daily.round(4).reset_index(names="날짜"), "추천_일별요약.csv")
    io_utils.save_csv(reason_stats(out), "사유코드_통계.csv")
    io_utils.save_csv(interp_vs_k0_table(), "수온별_표준RPM_보간표.csv")
    io_utils.save_csv(gt_check_table(out["q_m3h"]), "지별_Gt_점검.csv")

    # ── §12 침전수 탁도 시간창 비교·사후평가 ──
    drivers = pd.DataFrame({"원수탁도": out["원수탁도_정렬"],
                            "PAC_SV": out["PAC_SV_정렬"],
                            "유량": out["q_m3h"]})
    cmp_df = evaluate.compare_windows(df, drivers)
    io_utils.save_csv(cmp_df, "시간창후보_비교.csv")
    best = cmp_df.iloc[0]
    print(f"시간창 우세 후보: {best['시차중심']} {best['중심(분)']}±"
          f"{best['반폭(분)']}분 (침전지{best['침전지']}, "
          f"|ρ|평균 {best['ρ평균절대값']:.3f}) — config.LAG_DEFAULT="
          f"{C.LAG_DEFAULT} 유지, 변경은 수동 승인 (§12)")

    snap = evaluate.posthoc_table(out, df)
    snap.to_parquet(C.OUTPUT_DIR / "침전탁도_사후평가_전기간.parquet")
    posthoc_daily = evaluate.daily_posthoc(snap)

    # 기준선 지표: 24h persistence(전일 동시각) — §13.3 식의 참고 기준선
    met_rows = []
    for sed in (1, 2):
        y = snap[f"침전지{sed}_탁도_정렬"]
        met = evaluate.metrics(y, y.shift(24 * 60))
        met_rows.append({"침전지": sed, "기준선": "24h persistence", **met})
    met_df = pd.DataFrame(met_rows).round(4)
    io_utils.save_csv(posthoc_daily.reset_index(names="날짜"), "침전탁도_사후평가.csv")
    io_utils.save_csv(met_df, "침전탁도_기준선지표.csv")
    print(met_df.to_string(index=False))

    # ── 검증 요약 ──
    summary = pd.concat([unit_results.rename(columns={"결과": "값", "비고": "비고"}),
                         bt.assign(비고="").rename(columns={"항목": "항목", "값": "값"})],
                        ignore_index=True)
    io_utils.save_csv(summary, "검증_요약.csv")

    # ── 시각화 ──
    visualization.plot_seasonal_rpm(daily, C.IMAGES_DIR / "계절별_추천RPM.png")
    visualization.plot_reason_timeline(out, C.IMAGES_DIR / "사유코드_타임라인.png")
    visualization.plot_g_drop(daily, C.IMAGES_DIR / "G_drop_시계열.png")
    visualization.plot_gt_distribution(gt_basins, C.IMAGES_DIR / "Gt_분포.png")
    visualization.plot_window_compare(cmp_df, C.IMAGES_DIR / "시간창_상관비교.png")
    print("시각화 5종 저장 완료")
    print(f"전체 완료 — 산출물: {C.OUTPUT_DIR}")


if __name__ == "__main__":
    run()
