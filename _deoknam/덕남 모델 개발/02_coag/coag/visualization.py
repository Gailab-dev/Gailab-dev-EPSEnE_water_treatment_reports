# -*- coding: utf-8 -*-
"""시각화: 상관관계, DTW lag, 특성중요도, SHAP, 모델 비교, 실제vs예측."""
import numpy as np
import pandas as pd

from . import config as C
from . import models_dl

SHORT = {c: c.split(".")[-1] for c in C.CAND_FEATURES + C.TARGET_COLS}


def _short(cols):
    return [SHORT.get(c, c) for c in cols]


def plot_corr_heatmap(table: pd.DataFrame, cid: int) -> None:
    import matplotlib.pyplot as plt

    cols = C.CAND_FEATURES + ["y1_future", "y2_future"]
    corr = table[cols].corr()
    labels = _short(C.CAND_FEATURES) + ["침전지1탁도(t+lag)", "침전지2탁도(t+lag)"]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5)
    ax.set_title(f"cluster{cid} 후보 피처·타깃 상관관계")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"cluster{cid}_상관관계.png", dpi=130)
    plt.close(fig)


def plot_dtw_lag(detail: pd.DataFrame, lags: dict, cid: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, tgt in zip(axes, C.TARGET_COLS):
        name = tgt.split(".")[-1]
        v = detail[detail["타깃"] == name]["DTW_lag(분)"].dropna() if len(detail) else pd.Series(dtype=float)
        if len(v):
            ax.hist(v, bins=np.arange(C.DTW["lag_min"], C.DTW["lag_max"] + 20, 20),
                    color=C.CLUSTER_COLORS[cid], alpha=0.7)
        ax.axvline(lags[tgt], color="black", lw=1.5, label=f"채택 lag {lags[tgt]}분")
        ax.axvline(C.PRIOR_LAGS[cid], color="gray", ls="--", lw=1.2,
                   label=f"기존 연구 {C.PRIOR_LAGS[cid]}분")
        ax.set_title(f"{name} (유효 세그먼트 {len(v)}개)")
        ax.set_xlabel("lag (분)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("세그먼트 수")
    fig.suptitle(f"cluster{cid} DTW 체류시간(lag) 추정")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"cluster{cid}_DTW_lag추정.png", dpi=130)
    plt.close(fig)


def plot_importance(best_name: str, artifact: dict, cid: int) -> None:
    """트리 모델 내장 중요도. DL 모델은 SHAP bar가 중요도를 대신하므로 생략."""
    import matplotlib.pyplot as plt

    if best_name not in ("XGBoost", "RandomForest"):
        print(f"  특성중요도: {best_name}은 SHAP bar로 대체")
        return
    feat = artifact["feat_cols"]
    imp = np.asarray(artifact["model"].feature_importances_, dtype=float)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(feat) + 2))
    ax.barh(np.array(_short(feat))[order], imp[order], color=C.MODEL_COLORS[best_name])
    ax.set_title(f"cluster{cid} {best_name} 특성중요도")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"cluster{cid}_특성중요도.png", dpi=130)
    plt.close(fig)


def plot_shap(best_name: str, artifact: dict, cid: int, df_te: pd.DataFrame) -> bool:
    """SHAP summary/bar. 실패 시 False 반환 (호출부에서 경고만)."""
    import matplotlib.pyplot as plt
    import shap

    from . import datasets

    feat = artifact["feat_cols"]
    labels = _short(feat)
    try:
        if best_name in ("XGBoost", "RandomForest"):
            Xte, _, _ = datasets.to_tabular(df_te, feat, artifact["scaler_x"],
                                            artifact["scaler_y"])
            # RF는 깊은 트리 다수라 TreeExplainer 비용이 커 표본을 줄임
            n_cap = C.SHAP_N["tree_sample"] if best_name == "XGBoost" \
                else C.SHAP_N["rf_sample"]
            n = min(n_cap, len(Xte))
            rng = np.random.default_rng(C.RANDOM_STATE)
            Xs = Xte[rng.choice(len(Xte), n, replace=False)]
            explainer = shap.TreeExplainer(artifact["model"])
            sv = explainer.shap_values(Xs, check_additivity=False)
            # 멀티아웃풋: (2, n, F) 또는 (n, F, 2) → 타깃 평균 |SHAP|
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv2 = sv.mean(axis=0) if sv.shape[0] == 2 else sv.mean(axis=2)
            else:
                sv2 = sv
            shap.summary_plot(sv2, Xs, feature_names=labels, show=False)
            plt.title(f"cluster{cid} {best_name} SHAP summary")
            plt.tight_layout()
            plt.savefig(C.IMAGES_DIR / f"cluster{cid}_SHAP_summary.png", dpi=130)
            plt.close()
            shap.summary_plot(sv2, Xs, feature_names=labels, plot_type="bar", show=False)
            plt.title(f"cluster{cid} {best_name} SHAP 중요도")
            plt.tight_layout()
            plt.savefig(C.IMAGES_DIR / f"cluster{cid}_SHAP_bar.png", dpi=130)
            plt.close()
        else:
            import torch

            Xte, _, _ = datasets.to_sequences(df_te, feat, artifact["scaler_x"],
                                              artifact["scaler_y"])
            if len(Xte) < 50:
                return False
            rng = np.random.default_rng(C.RANDOM_STATE)
            bg = Xte[rng.choice(len(Xte), min(C.SHAP_N["dl_background"], len(Xte)),
                                replace=False)]
            sm = Xte[rng.choice(len(Xte), min(C.SHAP_N["dl_sample"], len(Xte)),
                                replace=False)]
            model = artifact["model"]
            # cudnn RNN은 eval 모드 backward 불가 → SHAP 동안 cudnn 비활성
            with torch.backends.cudnn.flags(enabled=False):
                explainer = shap.GradientExplainer(
                    model, torch.from_numpy(bg).to(models_dl.DEVICE))
                sv = explainer.shap_values(torch.from_numpy(sm).to(models_dl.DEVICE))
            # (n, w, F, 2) 또는 리스트 → 시간축·타깃 |SHAP| 합산 → (n, F)
            sv = np.asarray(sv)
            if sv.ndim == 5:  # (2, n, w, F, 1)류 방어
                sv = sv.squeeze()
            if sv.ndim == 4:
                agg = np.abs(sv).mean(axis=-1).sum(axis=1) if sv.shape[-1] == 2 \
                    else np.abs(sv).mean(axis=0).sum(axis=1)
            else:
                agg = np.abs(sv).sum(axis=1)
            imp = agg.mean(axis=0)
            order = np.argsort(imp)
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(feat) + 2))
            ax.barh(np.array(labels)[order], imp[order], color=C.MODEL_COLORS[best_name])
            ax.set_title(f"cluster{cid} {best_name} SHAP 중요도 (시간축 합산)")
            ax.grid(True, alpha=0.2, axis="x")
            fig.tight_layout()
            fig.savefig(C.IMAGES_DIR / f"cluster{cid}_SHAP_bar.png", dpi=130)
            plt.close(fig)
        return True
    except Exception as e:  # SHAP 실패는 치명적이지 않음 — 경고 후 계속
        print(f"  [경고] SHAP 실패 ({best_name}): {e}")
        return False


def plot_model_compare(results: pd.DataFrame, cid: int) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (col1, col2, title, hline) in zip(axes, [
        ("침전지1_r2", "침전지2_r2", "R² (기준 0.9)", C.PASS["r2_min"]),
        ("침전지1_smape", "침전지2_smape", "SMAPE % (기준 5%)", C.PASS["smape_max"]),
    ]):
        ax.bar(x - 0.2, results[col1], 0.4, label="침전지1", color="#4878b0")
        ax.bar(x + 0.2, results[col2], 0.4, label="침전지2", color="#dd8452")
        ax.axhline(hline, color="red", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n(k={k})" for m, k in
                            zip(results["모델"], results["피처수"])], fontsize=8)
        ax.set_title(f"cluster{cid} {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"cluster{cid}_모델비교.png", dpi=130)
    plt.close(fig)


def plot_pred(best_name: str, artifact: dict, common, cid: int,
              metrics_row: pd.Series) -> None:
    """best 모델: 실제vs예측 시계열(침전지1/2) + 산점도."""
    import matplotlib.pyplot as plt

    yt, yp = artifact["common_true"], artifact["common_pred"]
    for i, basin in enumerate(["침전지1", "침전지2"]):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(common, yt[:, i], color="#999999", lw=0.7, label="실제")
        ax.plot(common, yp[:, i], color=C.MODEL_COLORS[best_name], lw=0.7,
                label=f"{best_name} 예측")
        r2 = metrics_row[f"{basin}_r2"]
        sm = metrics_row[f"{basin}_smape"]
        ax.set_title(f"cluster{cid} {basin} 탁도(1h평균) 실제 vs 예측 (test) — "
                     f"R²={r2:.3f}, SMAPE={sm:.2f}%")
        ax.set_ylabel("탁도 (NTU)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(C.IMAGES_DIR / f"cluster{cid}_실제vs예측_{basin}.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, (ax, basin) in enumerate(zip(axes, ["침전지1", "침전지2"])):
        ax.scatter(yt[:, i], yp[:, i], s=4, alpha=0.3,
                   color=C.MODEL_COLORS[best_name])
        lim = [min(yt[:, i].min(), yp[:, i].min()), max(yt[:, i].max(), yp[:, i].max())]
        ax.plot(lim, lim, color="black", lw=1, ls="--")
        ax.set_xlabel("실제 (NTU)")
        ax.set_ylabel("예측 (NTU)")
        ax.set_title(f"{basin} (R²={metrics_row[f'{basin}_r2']:.3f})")
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"cluster{cid} {best_name} 실제 vs 예측 산점도")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"cluster{cid}_실제vs예측_산점도.png", dpi=130)
    plt.close(fig)


def plot_overall(all_results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric, hline, title in zip(
            axes, ["최소_r2", "평균_smape"],
            [C.PASS["r2_min"], C.PASS["smape_max"]],
            ["최소 R² (두 타깃 중 낮은 값)", "평균 SMAPE %"]):
        for name in C.MODEL_NAMES:
            sub = all_results[all_results["모델"] == name]
            if len(sub):
                ax.plot(sub["cluster"], sub[metric], "o-", label=name,
                        color=C.MODEL_COLORS[name])
        ax.axhline(hline, color="red", ls="--", lw=1)
        ax.set_xticks(C.CLUSTERS)
        ax.set_xticklabels([f"cluster{c}" for c in C.CLUSTERS])
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle("군집 × 모델 test 성능 비교")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "전체_모델성능비교.png", dpi=130)
    plt.close(fig)
