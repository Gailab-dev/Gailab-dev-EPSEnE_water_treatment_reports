# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 01: 체류시간 산정표 (§4).

실행: python 착수공정/run_01_residence.py
산출: output/체류시간_산정표.csv, 체류시간_시계열_통계.csv, images/체류시간_*.png
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import io_utils as IO
from intake_coag import residence as R


def main():
    IO.setup_output()
    df = IO.load_master()
    q = df[C.FLOW_COL]
    print(f"로드: {C.DATA_PARQUET.name} — {len(df):,}행 "
          f"({df.index.min()} ~ {df.index.max()})")

    # ── 정적 후보 (문서값 재계산 assert 포함) ─────────────────────────
    cand = R.static_candidates()

    # ── 동적 시계열 ───────────────────────────────────────────────────
    t = R.residence_series(q)
    valid = t["T_mean"].notna()
    n_bad_q = int((~valid).sum())
    print(f"FT101 유효 {valid.sum():,}행 / 무효(≤0·결측) {n_bad_q:,}행 "
          f"({n_bad_q / len(t) * 100:.2f}%)")

    # sanity: 유량 스케일 (중앙값이 기준유량과 동일 자릿수)
    q_med = float(q[q > 0].median())
    assert 0.3 * C.Q_REF < q_med < 3 * C.Q_REF, \
        f"FT101 중앙값 {q_med:.0f}가 기준유량 {C.Q_REF:.0f}(m³/h)와 스케일 불일치"
    print(f"FT101 중앙값 {q_med:,.0f} m³/h (기준유량 {C.Q_REF:,.1f} — 스케일 일치)")

    # sanity: 수기 검산 — 중앙유량에서 T_mean
    qj = q_med / C.N_ACTIVE_DEFAULT
    t_hand = (C.T_UPSTREAM_REF * C.Q_REF / q_med
              + C.K_FLOC * 60 * C.V_FLOC / qj + C.K_SED * 60 * C.V_SED / qj)
    t_med = float(t["T_mean"].median())
    print(f"T_mean 중앙값 {t_med:.1f}분 (수기 검산 {t_hand:.1f}분)")
    assert abs(t_med - min(max(t_hand, *[C.T_CLIP[0]]), C.T_CLIP[1])) < 30, \
        "T_mean 중앙값이 수기 검산과 30분 이상 괴리"

    # 클립 밖 비율 (클립 전 원값 기준)
    raw_mean = (t["T_upstream"] + C.K_FLOC * t["T_floc"] + C.K_SED * t["T_sed"])
    out_frac = float(((raw_mean < C.T_CLIP[0]) | (raw_mean > C.T_CLIP[1])).mean() * 100)
    print(f"T_mean 클립범위 {C.T_CLIP} 밖 비율: {out_frac:.2f}%")

    # ── 산정표: 정적 후보 + 동적 분포 ─────────────────────────────────
    hi, lo = R.basin_flow_ratio_bounds()
    dyn_rows = []
    for name, col in [("유량보정 평균 T_mean(t)", "T_mean"),
                      ("유량보정 첨두 T_peak(t)", "T_peak"),
                      ("이론 T_theoretical(t)", "T_theoretical")]:
        s = t.loc[valid, col]
        dyn_rows.append({
            "후보": name, "계산식": "§4.3·4.4 동적식 (FT101/가동지수 8)",
            "재계산값(분)": np.nan, "문서값(분)": np.nan,
            "정렬중심(분·1분반올림)": int(s.median().round()),
            "p5(분)": round(float(s.quantile(0.05)), 1),
            "p50(분)": round(float(s.median()), 1),
            "p95(분)": round(float(s.quantile(0.95)), 1),
        })
    table = pd.concat([cand, pd.DataFrame(dyn_rows)], ignore_index=True)
    table["비고"] = ""
    table.loc[0, "비고"] = "출수부 최강 응답 도달시점"
    table.loc[1, "비고"] = "체류시간분포 평균 중심"
    table.loc[6, "비고"] = (f"진단서 지별 유량비 편차: 최속지 T×{1/hi:.2f} ~ 최지지 T×{1/lo:.2f} "
                            f"(분포형 §4.4 폭)")
    p = IO.save_csv(table, "체류시간_산정표.csv")
    print(f"저장: {p.name}")

    # ── 월별·유량구간별 통계 ──────────────────────────────────────────
    tv = t[valid].copy()
    tv["월"] = tv.index.to_period("M").astype(str)
    qbin = pd.qcut(tv["Q_total"], q=[0, .33, .66, 1],
                   labels=["저유량", "중유량", "고유량"], duplicates="drop")
    rows = []
    for key, g in tv.groupby("월"):
        rows.append({"구분": "월", "구간": key, "n": len(g),
                     "Q중앙(m³/h)": round(float(g["Q_total"].median()), 0),
                     "T_mean중앙(분)": round(float(g["T_mean"].median()), 1),
                     "T_peak중앙(분)": round(float(g["T_peak"].median()), 1)})
    for key, g in tv.groupby(qbin, observed=True):
        rows.append({"구분": "유량구간", "구간": str(key), "n": len(g),
                     "Q중앙(m³/h)": round(float(g["Q_total"].median()), 0),
                     "T_mean중앙(분)": round(float(g["T_mean"].median()), 1),
                     "T_peak중앙(분)": round(float(g["T_peak"].median()), 1)})
    p = IO.save_csv(pd.DataFrame(rows), "체류시간_시계열_통계.csv")
    print(f"저장: {p.name}")

    # ── 그림 ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    t10 = t[valid].resample("1D").median()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t10.index, t10["T_mean"], label="T_mean(t) 유량보정 평균", lw=1.2)
    ax.plot(t10.index, t10["T_peak"], label="T_peak(t) 유량보정 첨두", lw=1.2)
    for name, v in C.CENTERS_STATIC.items():
        ax.axhline(v, ls="--", lw=0.8, color="gray")
        ax.text(t10.index[0], v + 4, f"{name}={v}분", fontsize=8, color="gray")
    ax.set_ylabel("체류시간(분)"), ax.set_title("동적 체류시간 (일 중앙값) vs 정적 후보")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "체류시간_시계열.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t.loc[valid, "Q_total"], bins=60, color="#4878b0")
    ax.axvline(C.Q_REF, color="r", ls="--", label=f"기준유량 {C.Q_REF:,.0f}")
    ax.set_xlabel("FT101 (m³/h)"), ax.set_title("총유량 분포"), ax.legend()
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "유량_히스토그램.png", dpi=130)
    plt.close(fig)
    print(f"저장: images/체류시간_시계열.png, 유량_히스토그램.png")


if __name__ == "__main__":
    main()
