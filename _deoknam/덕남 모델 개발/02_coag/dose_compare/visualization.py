# -*- coding: utf-8 -*-
"""비교표·실제vs예측·피처중요도 시각화."""
import numpy as np
import pandas as pd

from . import config as C


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_metric_bars(summary: pd.DataFrame, metric: str, path):
    """군집×모델 metric 막대 (AR제외 기준). summary: 컬럼 군집·모델·피처셋·metric."""
    plt = _setup()
    d = summary[summary["피처셋"] == "AR제외"]
    piv = d.pivot_table(index="모델", columns="군집", values=metric)
    ax = piv.plot(kind="bar", figsize=(10, 5),
                  color=[C.CLUSTER_COLORS.get(c, "#888") for c in piv.columns])
    ax.set_title(f"군집×모델 {metric} (AR제외)")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)


def plot_pred_vs_actual(idx, y_true, preds: dict, cid, path, max_pts=3000):
    """test 구간 실제 vs 예측(여러 모델) 시계열."""
    plt = _setup()
    n = len(idx)
    step = max(1, n // max_pts)
    sl = slice(None, None, step)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(idx[sl], np.asarray(y_true)[sl], color="#222", lw=1.2, label="실제 SV", zorder=5)
    for name, yp in preds.items():
        ax.plot(idx[sl], np.asarray(yp)[sl], lw=0.8, alpha=0.8,
                color=C.MODEL_COLORS.get(name, None), label=name)
    ax.set_title(f"cluster{cid} · 실제 vs 예측 주입률(SV) — test 구간")
    ax.set_ylabel("SV (ppm)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance_series(imp: pd.Series, cid, name, path, topn=20):
    """트리계열 피처중요도 (Series 입력)."""
    imp = imp.sort_values()[-topn:]
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp))))
    ax.barh(imp.index, imp.values, color=C.CLUSTER_COLORS.get(cid, "#4878b0"))
    ax.set_title(f"cluster{cid} · {name} 피처중요도 top{topn}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model, feat_cols, cid, name, path, topn=20):
    """트리계열 피처중요도."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()[-topn:]
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp))))
    ax.barh(imp.index, imp.values, color=C.CLUSTER_COLORS.get(cid, "#4878b0"))
    ax.set_title(f"cluster{cid} · {name} 피처중요도 top{topn}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
