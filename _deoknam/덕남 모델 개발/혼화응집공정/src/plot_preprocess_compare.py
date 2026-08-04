# -*- coding: utf-8 -*-
"""전처리 전/후 시계열 비교 플롯.

원본(옅은 회색)과 전처리(*_clean, 진한 파랑)를 겹쳐 그려 스파이크 제거·평활 효과를 확인.
실행: python -m src.plot_preprocess_compare
"""
import pandas as pd

from . import common as C

PRE = C.DATA_DIR / "덕남_통합_전처리.parquet"

PANELS = [
    ("RCS_6.AI.착수정_TB", "원수 탁도 (NTU)"),
    ("RCS_6.AI.착수정_PH", "원수 pH"),
    ("RCS_6.AI.착수정_AL", "알칼리도"),
    ("RCS_6.AI.착수정_전기전도도", "전기전도도"),
    ("RCS_6.AI.착수정_온도", "수온 (°C)"),
    (C.TARGET_TB[1], "침전지1 탁도 (NTU)"),
    (C.TARGET_TB[2], "침전지2 탁도 (NTU)"),
    ("RCS_6.AI.여과지_탁도", "여과지 탁도 (NTU)"),
    ("RCS_6.AI.정수지_탁도", "정수지 탁도 (NTU)"),
    ("RCS_6.AI.정수지_잔류염소", "정수지 잔류염소 (mg/L)"),
]

RAW_C = "#c8c8c8"
CLEAN_C = "#2f5c9e"


def main():
    C.setup_output()
    import matplotlib.pyplot as plt

    df = pd.read_parquet(PRE).sort_index()
    # 10분 다운샘플(min–max 엔벨로프 대신 대표값으로 가볍게)
    cols = [c for c, _ in PANELS] + [f"{c}_clean" for c, _ in PANELS]
    env = df[cols].resample("10min").median()

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15, 2.1 * len(PANELS)), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        ax.plot(env.index, env[col], color=RAW_C, lw=0.7, label="원본", zorder=1)
        ax.plot(env.index, env[f"{col}_clean"], color=CLEAN_C, lw=0.6, label="전처리", zorder=2)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(
        f"덕남 통합 데이터 전처리 전/후 비교 (10분 다운샘플, {df.index.min().date()} ~ {df.index.max().date()})"
        " — 회색: 원본, 파랑: 전처리(스파이크 제거+평활)"
    )
    fig.tight_layout()
    path = C.PLOTS_DIR / "preprocess_compare.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
