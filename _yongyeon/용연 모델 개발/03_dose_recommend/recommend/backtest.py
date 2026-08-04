# -*- coding: utf-8 -*-
"""백테스트: test 구간에서 추천 주입률 vs 실제 주입률 비교 + 시각화."""
import numpy as np
import pandas as pd

from . import config as R
from . import dose_model, recommender
from coag import config as CC
from coag import datasets


def run_backtest(cid: int, models: dict, table: pd.DataFrame):
    """test 구간 전체 추천 생성. 탐색은 학습(train+val) 주입률 지지구간으로 제한."""
    df_tr, df_val, df_te = datasets.chrono_split(table)
    dose_hist = pd.concat([df_tr, df_val])[CC.COL_DOSE]
    lo = float(dose_hist.quantile(R.DOSE_SUPPORT_Q[0]))
    hi = float(dose_hist.quantile(R.DOSE_SUPPORT_Q[1]))
    print(f"  주입률 지지구간(P5~P95): {lo:.1f} ~ {hi:.1f} ppm")
    rec = recommender.recommend(models, df_te, dose_lo=lo, dose_hi=hi)
    for i, name in enumerate(R.TARGET_NAMES, start=1):
        rec[f"실제_{name}"] = df_te[f"y{i}_future"].to_numpy()
    rec.attrs["support"] = (lo, hi)
    return rec


def summarize(cid: int, rec: pd.DataFrame, metrics: dict) -> dict:
    diff = rec["추천주입률"] - rec["현재주입률"]
    lower = diff < -0.25
    higher = diff > 0.25
    ok_when_lower = np.ones(int(lower.sum()), dtype=bool)
    exceed = pd.Series(False, index=rec.index)
    for name, tgt in zip(R.TARGET_NAMES, R.TARGET_NTU):
        ok_when_lower &= (rec.loc[lower, f"실제_{name}"] <= tgt).to_numpy()
        exceed |= rec[f"실제_{name}"] > tgt
    return {
        "cluster": cid,
        "test행": len(rec),
        "평균 실제주입률(ppm)": round(rec["현재주입률"].mean(), 2),
        "평균 추천주입률(ppm)": round(rec["추천주입률"].mean(), 2),
        "추천<실제 비율(%)": round(lower.mean() * 100, 1),
        "그때 실제탁도 목표만족(%)": round(ok_when_lower.mean() * 100, 1) if lower.any() else np.nan,
        "추천>실제 비율(%)": round(higher.mean() * 100, 1),
        "실제 목표초과 비율(%)": round(exceed.mean() * 100, 2),
        "목표달성가능 비율(%)": round(rec["목표달성가능"].mean() * 100, 1),
        "모델 test R2": "/".join(f"{metrics[f'{n}_r2']:.2f}" for n in R.TARGET_NAMES),
    }


def plot_backtest(cid: int, rec: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                             height_ratios=[1.2, 1])
    r10 = rec.resample("1h").mean(numeric_only=True)
    axes[0].plot(r10.index, r10["현재주입률"], color="#999999", lw=1.0, label="실제 주입률")
    axes[0].plot(r10.index, r10["추천주입률"], color="#4878b0", lw=1.0, label="추천 주입률")
    axes[0].set_ylabel("주입률 (ppm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title(f"cluster{cid} 응집제 주입률: 실제 vs 추천 (test, 1h 평균)")
    for name, tgt, color in zip(R.TARGET_NAMES, R.TARGET_NTU, ["#c44e52", "#dd8452"]):
        axes[1].plot(r10.index, r10[f"실제_{name}"], color=color, lw=0.8,
                     label=f"실제 {name}")
        axes[1].axhline(tgt, color="red", ls="--", lw=1, label=f"목표 {tgt}NTU")
    axes[1].set_ylabel("침전 탁도 (NTU)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(R.IMAGES_DIR / f"cluster{cid}_추천vs실제_주입률.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist((rec["추천주입률"] - rec["현재주입률"]).clip(-8, 8), bins=33,
            color="#55a868", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("추천 − 실제 주입률 (ppm)")
    ax.set_ylabel("빈도")
    ax.set_title(f"cluster{cid} 추천-실제 주입률 차이 분포 (test)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(R.IMAGES_DIR / f"cluster{cid}_주입률차이_분포.png", dpi=130)
    plt.close(fig)


def plot_response_curves(cid: int, models: dict, table: pd.DataFrame,
                         n_samples: int = 6) -> None:
    """대표 시점의 주입률-예측 침전탁도 응답곡선."""
    import matplotlib.pyplot as plt

    _, _, df_te = datasets.chrono_split(table)
    rng = np.random.default_rng(R.RANDOM_STATE)
    idx = df_te.index[rng.choice(len(df_te), min(n_samples, len(df_te)), replace=False)]
    rows = df_te.loc[idx]
    pred = recommender.sweep_predict(models, rows)
    grid = recommender.dose_grid()

    n = len(R.TARGET_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.5), sharex=True, squeeze=False)
    for b, ax in enumerate(axes[0]):
        for j, t in enumerate(idx):
            ax.plot(grid, pred[j, :, b], lw=1.2, label=str(t))
        ax.axhline(R.TARGET_NTU[b], color="red", ls="--", lw=1)
        ax.set_xlabel("주입률 (ppm)")
        ax.set_title(f"{R.TARGET_NAMES[b]} 예측 탁도 응답곡선")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("예측 탁도 (NTU)")
    fig.suptitle(f"cluster{cid} 주입률 what-if 응답곡선 (단조 제약)")
    fig.tight_layout()
    fig.savefig(R.IMAGES_DIR / f"cluster{cid}_응답곡선.png", dpi=130)
    plt.close(fig)
