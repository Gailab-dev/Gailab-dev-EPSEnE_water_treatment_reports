# -*- coding: utf-8 -*-
"""시각화 (Agg + Malgun Gothic — io_utils.setup_output에서 설정)."""
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_seasonal_rpm(daily: pd.DataFrame, path) -> None:
    """일별 수온·단별 추천 RPM — 계절 정합(저수온기 RPM 상승) 확인용."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    for k, c in zip((1, 2, 3), ("tab:blue", "tab:orange", "tab:green")):
        ax1.plot(daily.index, daily[f"rpm_safe_{k}"], c, lw=0.9, label=f"{k}단 추천")
    ax1.set_ylabel("추천 RPM")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["수온_정렬"], "tab:red", lw=0.8, alpha=0.6)
    ax2.set_ylabel("수온(℃)", color="tab:red")
    ax1.set_title("일별 추천 RPM(표준후보)과 수온")
    _save(fig, path)


def plot_reason_timeline(out: pd.DataFrame, path) -> None:
    """일별 primary 사유코드 구성비."""
    share = (out.groupby([pd.Grouper(freq="1D"), "reason_code"]).size()
                .unstack(fill_value=0))
    share = share.div(share.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    share.plot.area(ax=ax, lw=0, alpha=0.85)
    ax.set_ylabel("구성비")
    ax.set_title("일별 사유코드 구성비")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_g_drop(daily: pd.DataFrame, path) -> None:
    """낙차부 G 감시 시계열 + 운전범위 밴드 (§8)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["g_drop"], lw=0.8, color="tab:blue")
    ax.axhspan(*C.G_DROP_RANGE, color="tab:green", alpha=0.15,
               label=f"운전범위 {C.G_DROP_RANGE[0]}~{C.G_DROP_RANGE[1]}")
    ax.axhline(C.G_DROP_REF, color="tab:red", ls="--", lw=0.8,
               label=f"기준 {C.G_DROP_REF}")
    ax.set_ylabel("G_drop (1/s)")
    ax.set_title("착수정 낙차부 수류혼화 G (일별 중앙값)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_gt_distribution(gt_basins: pd.DataFrame, path) -> None:
    """지별 Gt 분포 (상자그림) + 승인범위."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [gt_basins[c].dropna() for c in gt_basins.columns]
    ax.boxplot(data, labels=list(gt_basins.columns), showfliers=False)
    ax.axhspan(*C.GT_RANGE, color="tab:green", alpha=0.12,
               label="승인범위 3e4~2e5")
    ax.axhline(C.GT_CENTER, color="tab:red", ls="--", lw=0.8, label="중심 1.1e5")
    ax.set_yscale("log")
    ax.set_ylabel("Gt")
    ax.set_title("지별 Gt 분포 (실측 유량비 8지 배분)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_window_compare(cmp_df: pd.DataFrame, path) -> None:
    """§12 시간창 후보별 상관 비교."""
    g = (cmp_df.assign(창=lambda d: d["시차중심"] + " ±" + d["반폭(분)"].astype(str))
               .groupby("창")["ρ평균절대값"].mean().sort_values())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    g.plot.barh(ax=ax, color="tab:blue", alpha=0.8)
    ax.set_xlabel("|Spearman ρ| 평균 (드라이버·침전지 평균)")
    ax.set_title("침전수 탁도 시간창 후보 비교")
    _save(fig, path)
