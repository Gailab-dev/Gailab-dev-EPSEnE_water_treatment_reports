# -*- coding: utf-8 -*-
"""군집 결과 시각화: 시간점유, PCA 산점도, 피처 분포, 월별 점유율."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config as C
from .model import _centers_original


def plot_occupancy(labels: pd.Series, turbidity: pd.Series, path) -> None:
    """상단: 시점별 군집 라벨 산점 / 하단: 착수정 탁도 라인."""
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                                   height_ratios=[1, 2])
    step = 5  # 파일 크기 절감용 표본화
    for cid in range(C.K):
        t = labels.index[(labels == cid).fillna(False).to_numpy(dtype=bool)][::step]
        ax0.scatter(t, np.full(len(t), cid), s=1, color=C.CLUSTER_COLORS[cid],
                    label=f"cluster{cid}")
    ax0.set_yticks(range(C.K))
    ax0.set_yticklabels([f"cluster{i}" for i in range(C.K)], fontsize=8)
    ax0.legend(loc="upper right", fontsize=8, ncol=C.K, markerscale=6)
    ax0.set_title("군집 시간 점유 (상) / 착수정 탁도 (하)")
    ax0.grid(True, alpha=0.2)

    env = turbidity.resample("10min").mean()
    ax1.plot(env.index, env, color=C.LINE_COLOR, lw=0.5)
    ax1.set_ylabel("착수정 TB (NTU)", fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(feats: pd.DataFrame, labels: pd.Series, pipe, path,
                     n: int = C.SIL_SAMPLE) -> None:
    """스케일 공간 PCA 2D 산점도 + 군집 중심."""
    import matplotlib.pyplot as plt

    valid = feats.dropna()
    lab = labels.loc[valid.index].astype(int)
    if len(valid) > n:
        idx = valid.sample(n, random_state=C.RANDOM_STATE).index
        valid, lab = valid.loc[idx], lab.loc[idx]
    scaler = pipe.named_steps["scaler"]
    scaled = scaler.transform(valid.to_numpy())
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(scaled)
    pts = pca.transform(scaled)
    centers = pca.transform(scaler.transform(_centers_original(pipe)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(C.K):
        m = (lab == cid).to_numpy()
        ax.scatter(pts[m, 0], pts[m, 1], s=3, alpha=0.3, color=C.CLUSTER_COLORS[cid])
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, color="black",
               edgecolors="white", zorder=5)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title(f"군집 PCA 산점도 (표본 {len(valid):,}점, 스케일 공간)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=C.CLUSTER_COLORS[i],
                      label=f"cluster{i}") for i in range(C.K)]
    handles.append(Line2D([], [], marker="X", ls="", ms=10, color="black",
                          markeredgecolor="white", label="군집 중심"))
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, labels: pd.Series, path) -> None:
    """원수성상 5컬럼 × 군집별 박스플롯."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(C.RAW_FEATURES), figsize=(16, 4.5))
    for ax, col in zip(axes, C.RAW_FEATURES):
        data = [df.loc[(labels == cid).fillna(False).to_numpy(dtype=bool), col].dropna()
                for cid in range(C.K)]
        bp = ax.boxplot(data, tick_labels=[f"c{i}" for i in range(C.K)],
                        patch_artist=True, showfliers=False)
        for patch, cid in zip(bp["boxes"], range(C.K)):
            patch.set_facecolor(C.CLUSTER_COLORS[cid])
            patch.set_alpha(0.7)
        ax.set_title(col.split(".")[-1], fontsize=10)
        ax.grid(True, alpha=0.2)
    fig.suptitle("군집별 원수성상 분포 (이상값 표시 생략)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_share(labels: pd.Series, path) -> None:
    """월별 군집 점유율 스택바."""
    import matplotlib.pyplot as plt

    lab = labels.dropna().astype(int)
    share = (lab.groupby([lab.index.to_period("M"), lab]).size()
             .unstack(fill_value=0))
    share = share.div(share.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 4.5))
    bottom = np.zeros(len(share))
    for cid in share.columns:
        ax.bar(share.index.astype(str), share[cid], bottom=bottom,
               color=C.CLUSTER_COLORS[int(cid)], label=f"cluster{cid}")
        bottom += share[cid].to_numpy()
    ax.set_ylabel("점유율 (%)")
    ax.set_title("월별 군집 점유율")
    ax.legend(loc="upper right", fontsize=8, ncol=C.K)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(df: pd.DataFrame, feats: pd.DataFrame, labels: pd.Series, pipe) -> None:
    jobs = [
        ("군집_시간점유.png", lambda p: plot_occupancy(labels, df[C.TURBIDITY_COL], p)),
        ("군집_PCA산점도.png", lambda p: plot_pca_scatter(feats, labels, pipe, p)),
        ("군집_피처분포_박스플롯.png", lambda p: plot_boxplots(df, labels, p)),
        ("군집_월별점유율.png", lambda p: plot_monthly_share(labels, p)),
    ]
    for name, fn in jobs:
        path = C.PLOTS_DIR / name
        fn(path)
        print(f"저장: {path.name}")
