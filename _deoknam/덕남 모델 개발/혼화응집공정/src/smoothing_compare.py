# -*- coding: utf-8 -*-
"""보정(평활) 방법 비교: 칼만 필터 vs 중앙값 평활.

칼만은 이상치 '탐지기'가 아니라 상태추정·'보정' 필터다. 탐지(룰베이스/z-score/
IQR/IsoForest)로 이상치를 제거한 뒤의 평활 단계에서, 칼만 보정과 중앙값 평활을 비교.

실행: python -m src.smoothing_compare
"""
import numpy as np
import pandas as pd

from . import common as C

INTEGRATED = C.DATA_DIR / "덕남_응집제공정_소독공정_통합.parquet"

# 확대 비교 구간(평활 효과가 잘 보이는 스파이크 밀집 구간)과 대상 변수
FOCUS = [
    ("RCS_6.AI.침전지1_탁도", "침전지1 탁도 (NTU)"),
    ("RCS_6.AI.침전지2_탁도", "침전지2 탁도 (NTU)"),
    ("RCS_6.AI.정수지_탁도", "정수지 탁도 (NTU)"),
]
ZOOM = ("2025-09-01", "2025-09-15")


def kalman_smooth(s: pd.Series, r_scale: float = 5.0, q_scale: float = 1e-3):
    """1D 랜덤워크 칼만 보정. r_scale↑ → 더 매끄럽게(관측 불신).

    반환: 보정된 추정값 시리즈(xhat). 스파이크는 상태에 천천히 반영돼 완화됨.
    """
    x = s.to_numpy(dtype=float)
    n = len(x)
    finite = x[np.isfinite(x)]
    r = np.var(np.diff(finite)) * r_scale if len(finite) > 1 else 1.0
    q = r * q_scale
    xhat = np.empty(n)
    xh = finite[0] if len(finite) else 0.0
    P = 1.0
    for i in range(n):
        Pm = P + q
        xi = x[i]
        if np.isfinite(xi):
            K = Pm / (Pm + r)
            xh = xh + K * (xi - xh)
            P = (1 - K) * Pm
        else:
            P = Pm
        xhat[i] = xh
    return pd.Series(xhat, index=s.index)


def main():
    C.setup_output()
    import matplotlib.pyplot as plt

    df = pd.read_parquet(INTEGRATED).sort_index()
    z = df.loc[ZOOM[0]:ZOOM[1]]

    fig, axes = plt.subplots(len(FOCUS), 1, figsize=(15, 3.6 * len(FOCUS)), sharex=True)
    for ax, (col, label) in zip(axes, FOCUS):
        raw = z[col]
        med = raw.rolling(15, center=True, min_periods=1).median()
        kal = kalman_smooth(raw)
        ax.plot(raw.index, raw, color="#c8c8c8", lw=0.6, label="원본", zorder=1)
        ax.plot(med.index, med, color="#dd8452", lw=1.1, label="중앙값 평활(15분)", zorder=2)
        ax.plot(kal.index, kal, color="#4878b0", lw=1.1, label="칼만 보정", zorder=3)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"보정(평활) 비교: 칼만 필터 vs 중앙값 평활 ({ZOOM[0]} ~ {ZOOM[1]} 확대)")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = C.PLOTS_DIR / "smoothing_compare.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"저장: {path}")

    # 정량 비교: 원본 대비 잔차 표준편차(평활 강도)와 스파이크 완화율
    print("\n===== 평활 강도 비교 (확대 구간) =====")
    for col, label in FOCUS:
        raw = z[col].dropna()
        med = raw.rolling(15, center=True, min_periods=1).median()
        kal = kalman_smooth(raw)
        print(
            f"{label}: 원본 std={raw.std():.3f} | "
            f"중앙값평활 잔차std={ (raw-med).std():.3f} | 칼만보정 잔차std={(raw-kal).std():.3f} | "
            f"원본 최대={raw.max():.2f} → 중앙값={med.max():.2f}, 칼만={kal.max():.2f}"
        )


if __name__ == "__main__":
    main()
