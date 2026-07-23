# -*- coding: utf-8 -*-
"""덕남 응집제공정·소독공정 통합 데이터(1분 간격) 시계열 진단 플롯.

- 10분 min–max 엔벨로프로 그려 스파이크를 보존하면서 950k행을 가볍게 렌더링
- 최상단 패널에 cluster0/1/2가 전체 타임라인에서 차지하는 구간을 표시
- 월별 커버리지 표 출력

실행: python -m src.plot_timeseries_integrated
"""
import numpy as np
import pandas as pd

from . import common as C

INTEGRATED = C.DATA_DIR / "덕남_응집제공정_소독공정_통합.parquet"

PANELS = [
    ("RCS_6.AI.착수정_TB", "원수 탁도 (NTU)"),
    ("RCS_6.AI.착수정_PH", "원수 pH"),
    ("RCS_6.AI.착수정_AL", "알칼리도"),
    ("RCS_6.AI.착수정_온도", "수온 (°C)"),
    ("RCS_6.AI.착수정_전기전도도", "전기전도도"),
    (C.COL_DOSE, "PAC 주입율 SV (ppm)"),
    (C.TARGET_TB[1], "침전지1 탁도 (NTU)"),
    (C.TARGET_TB[2], "침전지2 탁도 (NTU)"),
    ("RCS_6.AI.여과지_탁도", "여과지 탁도 (NTU)"),
    ("RCS_6.AI.정수지_탁도", "정수지 탁도 (NTU)"),
    ("RCS_6.AI.정수지_잔류염소", "정수지 잔류염소 (mg/L)"),
    ("주입변경.후염소_PPM", "후염소 (PPM)"),
]

LINE = "#4878b0"
CLUSTER_COLORS = {0: "#c44e52", 1: "#55a868", 2: "#8172b2"}


def main():
    C.setup_output()
    import matplotlib.pyplot as plt

    df = pd.read_parquet(INTEGRATED).sort_index()
    env = df[[c for c, _ in PANELS]].resample("10min").agg(["min", "max"])

    fig, axes = plt.subplots(
        len(PANELS) + 1, 1, figsize=(15, 2.0 * (len(PANELS) + 1)), sharex=True,
        gridspec_kw={"height_ratios": [0.5] + [1] * len(PANELS)},
    )

    # 클러스터 점유 구간
    ax0 = axes[0]
    for cid in range(3):
        cdf = C.load_cluster(cid)
        ax0.scatter(cdf[C.COL_DT], np.full(len(cdf), cid), s=1, color=CLUSTER_COLORS[cid], label=f"cluster{cid}")
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(["cluster0", "cluster1", "cluster2"], fontsize=8)
    ax0.set_ylim(-0.5, 2.5)
    ax0.grid(True, alpha=0.2)
    ax0.set_title(
        f"덕남 응집·소독공정 통합 데이터 시계열 진단 (1분 간격, n={len(df):,}, "
        f"{df.index.min().date()} ~ {df.index.max().date()}) — 상단: 클러스터 점유 구간"
    )

    for ax, (col, label) in zip(axes[1:], PANELS):
        lo, hi = env[(col, "min")], env[(col, "max")]
        ax.fill_between(env.index, lo, hi, color=LINE, alpha=0.55, lw=0)
        ax.plot(env.index, hi, color=LINE, lw=0.3)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = C.PLOTS_DIR / "timeseries_integrated.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"저장: {path}")

    mon = df.index.strftime("%Y-%m")
    cov = pd.Series(mon).value_counts().sort_index().rename("행수(1분격자)")
    cov = cov.to_frame()
    cov["월최대"] = [pd.Period(m).days_in_month * 1440 for m in cov.index]
    cov["커버리지(%)"] = (cov["행수(1분격자)"] / cov["월최대"] * 100).round(1)
    print("\n===== 통합 데이터 월별 커버리지 =====")
    print(cov.to_string())
    C.save_csv(cov.reset_index().rename(columns={"index": "month"}), C.RESULTS_DIR / "monthly_coverage_integrated.csv")


if __name__ == "__main__":
    main()
