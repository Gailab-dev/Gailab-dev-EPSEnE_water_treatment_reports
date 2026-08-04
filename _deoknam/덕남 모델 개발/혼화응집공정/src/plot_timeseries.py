# -*- coding: utf-8 -*-
"""클러스터별 주요 변수 시계열 진단 플롯.

- 원수성상 5개 + PAC 주입율 SV + 침전지 1·2 탁도를 패널로 표시 (x축 공유)
- 30분 초과 갭에서 선을 끊어 결측 구간이 연결선으로 왜곡되지 않게 함
- 50%(백테스트 시작)·80%(기존 시간분할) 지점을 세로선으로 표시
- 월별 데이터 커버리지 표를 콘솔에 출력 (데이터 공백 진단용)

실행: python -m src.plot_timeseries
"""
import numpy as np
import pandas as pd

from . import common as C

PANELS = [
    ("RCS_6.AI.착수정_TB", "원수 탁도 (NTU)"),
    ("RCS_6.AI.착수정_PH", "원수 pH"),
    ("RCS_6.AI.착수정_AL", "알칼리도"),
    ("RCS_6.AI.착수정_온도", "수온 (°C)"),
    ("RCS_6.AI.착수정_전기전도도", "전기전도도"),
    (C.COL_DOSE, "PAC 주입율 SV (ppm)"),
    (C.TARGET_TB[1], "침전지1 탁도 (NTU)"),
    (C.TARGET_TB[2], "침전지2 탁도 (NTU)"),
]

LINE = "#4878b0"
GAP_BREAK = pd.Timedelta(minutes=30)


def with_gap_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """30분 초과 간격 지점에 NaN 행을 삽입해 선을 끊는다."""
    gaps = df[C.COL_DT].diff() > GAP_BREAK
    if not gaps.any():
        return df
    nan_rows = df.loc[gaps, [C.COL_DT]].copy()
    nan_rows[C.COL_DT] = nan_rows[C.COL_DT] - pd.Timedelta(minutes=1)
    return (
        pd.concat([df, nan_rows], ignore_index=True)
        .sort_values(C.COL_DT)
        .reset_index(drop=True)
    )


def plot_cluster(cid: int):
    import matplotlib.pyplot as plt

    df = C.load_cluster(cid)
    dfb = with_gap_breaks(df)
    t50 = df[C.COL_DT].iloc[int(len(df) * 0.5) - 1]
    t80 = df[C.COL_DT].iloc[int(len(df) * 0.8) - 1]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15, 2.1 * len(PANELS)), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        ax.plot(dfb[C.COL_DT], dfb[col], color=LINE, lw=0.6)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        for t, c, name in [(t50, "#c44e52", "50% (백테스트 시작)"), (t80, "#8172b2", "80% (시간분할 컷)")]:
            ax.axvline(t, color=c, ls="--", lw=1)
    axes[0].set_title(
        f"cluster{cid} 시계열 진단 (n={len(df):,}, {df[C.COL_DT].min().date()} ~ {df[C.COL_DT].max().date()})"
        " — 빨강 점선: 50% 백테스트 시작, 보라 점선: 80% 시간분할 컷"
    )
    fig.tight_layout()
    path = C.PLOTS_DIR / f"timeseries_cluster{cid}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"저장: {path}")


def coverage_table():
    rows = []
    for cid in range(3):
        df = C.load_cluster(cid)
        mon = df[C.COL_DT].dt.strftime("%Y-%m")
        counts = mon.value_counts()
        rows.append(counts.rename(f"cluster{cid}"))
    cov = pd.concat(rows, axis=1).fillna(0).astype(int).sort_index()
    cov["월최대(10분격자)"] = [pd.Period(m).days_in_month * 144 for m in cov.index]
    print("\n===== 월별 행 수 (데이터 커버리지) =====")
    print(cov.to_string())
    C.save_csv(cov.reset_index().rename(columns={"index": "month"}), C.RESULTS_DIR / "monthly_coverage.csv")


def main():
    C.setup_output()
    for cid in range(3):
        plot_cluster(cid)
    coverage_table()


if __name__ == "__main__":
    main()
