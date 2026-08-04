# -*- coding: utf-8 -*-
"""시각화: G-RPM 곡선(수온별), 일별 권장 rpm 시계열, 지별 G 진단 바차트."""
import numpy as np
import pandas as pd

from . import config as C
from . import gvalue

STAGE_COLORS = {1: "#c44e52", 2: "#dd8452", 3: "#55a868"}


def plot_g_rpm_curves(path) -> None:
    import matplotlib.pyplot as plt

    rpm = np.linspace(0.3, 2.5, 200)
    fig, ax = plt.subplots(figsize=(9, 6))
    for t, ls in [(5, "-"), (15, "--"), (25, ":")]:
        ax.plot(rpm, gvalue.g_from_rpm(rpm, t), ls=ls, color="#4878b0",
                label=f"수온 {t}℃")
    for s, g in C.G_TARGETS.items():
        ax.axhline(g, color=STAGE_COLORS[s], lw=1, ls="-.",
                   label=f"{s}단 목표 G={g:.0f}")
    ax.set_xlabel("회전수 (rpm)")
    ax.set_ylabel("G값 (/sec)")
    ax.set_title("플록큐레이터 G-RPM 곡선 (G = K0·√(ρ/μ)·rpm^1.5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_daily_rpm(daily: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                             height_ratios=[1, 1.4])
    axes[0].plot(daily.index, daily["수온"], color="#4878b0", lw=0.9)
    axes[0].set_ylabel("착수정 수온 (℃)")
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title("일별 수온과 단별 권장 회전수 (목표 G 50/30/10 점감)")
    for s in (1, 2, 3):
        axes[1].plot(daily.index, daily[f"{s}단 권장rpm"],
                     color=STAGE_COLORS[s], lw=0.9, label=f"{s}단")
    axes[1].set_ylabel("권장 회전수 (rpm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_basin_diagnosis(diag: pd.DataFrame, temp_c: float, path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(diag))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, s in enumerate((1, 2, 3)):
        ax.bar(x + (i - 1) * 0.27, diag[f"G {s}단"], 0.27,
               color=STAGE_COLORS[s], label=f"{s}단 G")
        ax.axhline(C.G_TARGETS[s], color=STAGE_COLORS[s], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(diag["지"], fontsize=9)
    ax.set_ylabel("G값 (/sec)")
    ax.set_title(f"지별 운영 G값 (수온 {temp_c:.0f}℃ 기준, 점선 = 단별 목표)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
