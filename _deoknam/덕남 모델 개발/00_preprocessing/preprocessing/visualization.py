# -*- coding: utf-8 -*-
"""전처리 전/후 비교 시각화 및 요약 리포트 그림."""
import pandas as pd

from . import config as C

RAW_COLOR = "#c8c8c8"
CLEAN_COLOR = "#2f5c9e"
REMOVED_COLOR = "#c44e52"


def _safe_name(col: str) -> str:
    return col.replace(".", "_")


def plot_before_after(raw: pd.Series, clean: pd.Series, removed: pd.Series,
                      col: str, out_path) -> None:
    """원본(10분 max, 회색) vs 전처리 후(10분 mean, 파랑) + 제거지점(빨강 산점)."""
    import matplotlib.pyplot as plt

    env_raw = raw.resample("10min").max()
    env_clean = clean.resample("10min").mean()
    n_rm = int(removed.sum())
    pct = n_rm / len(raw) * 100

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(env_raw.index, env_raw, color=RAW_COLOR, lw=0.6, label="전처리 전(10분 max)", zorder=1)
    ax.plot(env_clean.index, env_clean, color=CLEAN_COLOR, lw=0.8, label="전처리 후(10분 mean)", zorder=3)
    if n_rm:
        pts = raw[removed]
        ax.scatter(pts.index, pts.values, s=6, color=REMOVED_COLOR, alpha=0.5,
                   label=f"제거 지점 ({n_rm:,}행)", zorder=2)
    ax.set_title(f"{col} — 전처리 전/후 비교 (제거 {n_rm:,}행, {pct:.3f}%)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_all(df_raw: pd.DataFrame, df_clean: pd.DataFrame, masks: dict) -> None:
    """원본 40컬럼 + 분할 신규 4컬럼 전체 비교 이미지 생성."""
    none_mask = pd.Series(False, index=df_raw.index)
    for col in df_raw.columns:
        path = C.COMPARE_DIR / f"{_safe_name(col)}.png"
        plot_before_after(df_raw[col], df_clean[col], masks.get(col, none_mask), col, path)
        print(f"저장: {path.name}")
    # 분할 컬럼: 부모 원본을 같은 비율로 스케일해 '전' 곡선으로 사용
    for parent, children in C.CHLORINE_SPLIT.items():
        for child, ratio in zip(children, (C.PRE_RATIO, C.MID_RATIO)):
            path = C.COMPARE_DIR / f"{_safe_name(child)}.png"
            plot_before_after(df_raw[parent] * ratio, df_clean[child],
                              masks.get(parent, none_mask), child, path)
            print(f"저장: {path.name}")


def plot_rate_heatmap(report: pd.DataFrame, n0: int, out_path) -> None:
    """변수 × 방법 이상치 탐지율(%) 히트맵. n0 = 전체 행 수."""
    import matplotlib.pyplot as plt

    methods = ["물리범위밖", "Hampel", "IQR", "ISO채택", "최종제거"]
    mat = report.set_index("변수")[methods].div(n0 / 100)
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(mat) + 2))
    im = ax.imshow(mat.values, aspect="auto", cmap="OrRd",
                   vmin=0, vmax=max(0.5, min(3, float(mat.values.max()))))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([v.split(".")[-1] for v in mat.index], fontsize=8)
    for i in range(len(mat)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("변수 × 방법 이상치 탐지율 (%)")
    fig.colorbar(im, ax=ax, label="탐지율 %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
