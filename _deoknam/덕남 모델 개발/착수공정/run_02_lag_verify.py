# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 02: 시차상관·DTW 이벤트 검증 (§4.5-3).

실행: python 착수공정/run_02_lag_verify.py
산출: output/시차검증_이벤트별.csv, 시차검증_요약.csv, images/시차_히스토그램_vs_후보.png
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import io_utils as IO
from intake_coag import lag_verify as LV


def main():
    IO.setup_output()
    df = IO.load_master()
    # 10분 리샘플 (평균) — 원수탁도·침전지·주입유량 proxy만
    cols = [C.TURBIDITY_COL] + C.SED_TB + C.DOSE_FLOW
    df10 = df[cols].resample(f"{C.LAGV['step_min']}min").mean()
    df10.columns = [C.TURBIDITY_COL, *C.SED_TB, *[c.split(".")[-1] for c in C.DOSE_FLOW]]

    events = LV.find_events(df10)
    print(f"이벤트 후보: {len(events)}건 (착수정_TB 60분 변화 z>{C.LAGV['event_z']})")

    detail = LV.event_lags(df10, events)
    IO.save_csv(detail, "시차검증_이벤트별.csv")

    # 요약: 침전지별 지지 lag 분포 + 후보 대조
    rows = []
    for b in (1, 2, 0):
        sub = detail[detail["침전지"] == b]
        vals = pd.concat([sub["DTW_lag(분)"].dropna(),
                          sub["xcorr_lag(분)"].dropna()]).astype(float)
        if not len(vals):
            continue
        med = float(vals.median())
        lo, hi = float(vals.quantile(0.25)), float(vals.quantile(0.75))
        supported = [nm for nm, v in C.CENTERS_STATIC.items() if lo - 30 <= v <= hi + 30]
        rows.append({"침전지": {0: "dose→sed1(참고)"}.get(b, f"침전지{b}"),
                     "유효건수": int(len(vals)), "중앙(분)": round(med, 1),
                     "IQR(분)": f"{lo:.0f}~{hi:.0f}",
                     "DTW중앙(분)": round(float(sub["DTW_lag(분)"].dropna().median()), 1)
                     if sub["DTW_lag(분)"].notna().any() else None,
                     "xcorr중앙(분)": round(float(sub["xcorr_lag(분)"].dropna().median()), 1)
                     if sub["xcorr_lag(분)"].notna().any() else None,
                     "지지후보(±30분·IQR)": ",".join(supported) or "-"})
    summary = pd.DataFrame(rows)
    IO.save_csv(summary, "시차검증_요약.csv")
    print(summary.to_string(index=False))

    # 검증: 유효 이벤트 수·DTW vs xcorr 괴리
    for b in (1, 2):
        sub = detail[detail["침전지"] == b]
        n = int(sub[["DTW_lag(분)", "xcorr_lag(분)"]].notna().any(axis=1).sum())
        if n < C.LAGV["min_events"]:
            print(f"경고: 침전지{b} 유효 이벤트 {n} < {C.LAGV['min_events']} — 임계 완화 검토")
        both = sub.dropna(subset=["DTW_lag(분)", "xcorr_lag(분)"])
        if len(both):
            gap = float((both["DTW_lag(분)"] - both["xcorr_lag(분)"]).median())
            tag = "" if abs(gap) < 60 else " (경고: 60분 초과)"
            print(f"침전지{b}: DTW-xcorr 중앙 괴리 {gap:+.0f}분{tag}")

    # 히스토그램
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, b in zip(axes, (1, 2)):
        sub = detail[detail["침전지"] == b]
        vals = pd.concat([sub["DTW_lag(분)"].dropna(),
                          sub["xcorr_lag(분)"].dropna()]).astype(float)
        ax.hist(vals, bins=np.arange(90, 550, 30), color="#4878b0", alpha=0.8)
        for nm, v in C.CENTERS_STATIC.items():
            ax.axvline(v, ls="--", lw=1, color="r")
            ax.text(v, ax.get_ylim()[1] * 0.95, nm, rotation=90, fontsize=8, color="r")
        ax.set_title(f"침전지{b} 이벤트 lag 분포"), ax.set_xlabel("lag(분)")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "시차_히스토그램_vs_후보.png", dpi=130)
    plt.close(fig)
    print("저장: 시차검증_이벤트별.csv, 시차검증_요약.csv, images/시차_히스토그램_vs_후보.png")


if __name__ == "__main__":
    main()
