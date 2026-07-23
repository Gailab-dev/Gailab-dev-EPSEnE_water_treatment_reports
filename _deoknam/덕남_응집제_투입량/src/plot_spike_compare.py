# -*- coding: utf-8 -*-
"""원수 탁도 스파이크 vs 침전지 탁도 스파이크 대조 진단.

목적: 스파이크가 진짜 수질 이벤트(원수→체류시간→침전지 전파)인지,
      센서 오류(한쪽만 튐)인지 판별해 전처리 규칙의 근거를 만든다.

세 가지 대표 유형을 원수/침전지1/침전지2 오버레이로 시각화:
  (A) 진짜 강우 이벤트  : 원수 급등 → 3~4h 후 침전지 상승
  (B) 원수 센서 오류    : 원수 50(상한) 순간 급등 → 침전지 무반응
  (C) 원인 불명 침전지 탁도 급등: 침전지 3.0(상한) 급등 → 선행 원수 잔잔

실행: python -m src.plot_spike_compare
"""
import pandas as pd

from . import common as C

INTEGRATED = C.DATA_DIR / "덕남_응집제공정_소독공정_통합.parquet"
RAW = "RCS_6.AI.착수정_TB"
S1 = C.TARGET_TB[1]
S2 = C.TARGET_TB[2]

# (제목, 중심시각, 유형색). 앞선 분석에서 확인한 대표 사례.
CASES = [
    ("(A) 실제 수질 이벤트(강우 추정)\n원수 급등 → 체류시간 후 침전지 상승", "2025-09-04 20:01", "#55a868"),
    ("(B) 원수 탁도계 센서 오류\n원수 50(상한) 순간 급등 → 침전지 무반응", "2025-11-25 00:53", "#c44e52"),
    ("(C) 원인 불명 침전지 탁도 급등\n선행 원수 급등 없이 침전지만 3.0(상한) 급등", "2025-02-09 13:17", "#8172b2"),
]
HALF = pd.Timedelta("8h")


def main():
    C.setup_output()
    import matplotlib.pyplot as plt

    df = pd.read_parquet(INTEGRATED).sort_index()
    fig, axes = plt.subplots(len(CASES), 1, figsize=(14, 4.2 * len(CASES)))

    for ax, (title, center, color) in zip(axes, CASES):
        t = pd.Timestamp(center)
        win = df.loc[t - HALF : t + HALF]
        ax.plot(win.index, win[RAW], color="#4878b0", lw=1.2, label="원수 탁도")
        ax.axvline(t, color=color, ls="--", lw=1.2)
        ax.axhline(50, color="#4878b0", ls=":", lw=0.7, alpha=0.6)
        ax.set_ylabel("원수 탁도 (NTU)", color="#4878b0", fontsize=9)
        ax.tick_params(axis="y", labelcolor="#4878b0")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(True, alpha=0.2)

        ax2 = ax.twinx()
        ax2.plot(win.index, win[S1], color="#dd8452", lw=1.2, label="침전지1 탁도")
        ax2.plot(win.index, win[S2], color="#937860", lw=1.2, label="침전지2 탁도")
        ax2.axhline(3.0, color="#dd8452", ls=":", lw=0.7, alpha=0.6)
        ax2.set_ylabel("침전지 탁도 (NTU)", color="#dd8452", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#dd8452")

        lines = ax.get_lines()[:1] + ax2.get_lines()[:2]
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)

    fig.suptitle(
        "원수 탁도 스파이크 vs 침전지 탁도 스파이크 대조 (중심시각 ±8시간, 점선: 계측 상한)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = C.PLOTS_DIR / "spike_compare.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
