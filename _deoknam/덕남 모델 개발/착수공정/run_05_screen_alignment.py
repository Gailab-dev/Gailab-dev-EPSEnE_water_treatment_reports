# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 05: 정렬조합 스크리닝 (§4.5-4·§7.4 준수).

84 정렬조합 × XGBoost(AR포함) → **검증구간** sMAPE로 침전지별 top-3 선정.
시험구간(≥ts2)은 어떤 학습·선택에도 사용하지 않는다(assert).
persistence(현재 1h 평활 유지) 검증 sMAPE 병기, run_02 물리지지 범위 표기(§4.5-6).

실행: python 착수공정/run_05_screen_alignment.py  (run_02·run_03 산출 필요)
산출: output/정렬스크리닝_검증성능.csv, 선정_정렬조합.json, images/스크리닝_중심별_sMAPE.png
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import datasets as D
from intake_coag import evaluate as E
from intake_coag import io_utils as IO
from intake_coag import models as M


def y_columns() -> list[str]:
    cols = []
    for cname in list(C.CENTERS_STATIC) + C.CENTERS_DYNAMIC:
        for b in (1, 2):
            cols.append(f"y{b}__{cname}__pt")
            for d in C.DELTAS:
                cols += [f"y{b}__{cname}__med{d}", f"y{b}__{cname}__max{d}"]
    return cols


def main():
    IO.setup_output()
    table = pd.read_parquet(C.OUTPUT_DIR / "정렬학습테이블_10min.parquet")
    bounds = IO.load_json("분할경계.json")
    ts1, ts2 = pd.Timestamp(bounds["ts1"]), pd.Timestamp(bounds["ts2"])
    fdef = pd.read_csv(C.OUTPUT_DIR / "피처정의.csv", encoding="utf-8-sig")
    base = fdef[fdef["그룹"] == "base"]["컬럼"].tolist()
    sed_ar = fdef[fdef["그룹"] == "sed_ar"]["컬럼"].tolist()
    feats = base + sed_ar                      # 스크리닝은 AR포함 세트(§2-3)

    # run_02 지지범위 (±30분·IQR) — §4.5-6 물리지지 표기
    支 = {}
    try:
        s = pd.read_csv(C.OUTPUT_DIR / "시차검증_요약.csv", encoding="utf-8-sig")
        for _, r in s.iterrows():
            if str(r["침전지"]).startswith("침전지"):
                b = int(str(r["침전지"])[-1])
                lo, hi = str(r["IQR(분)"]).split("~")
                支[b] = (float(lo) - 30, float(hi) + 30)
    except FileNotFoundError:
        pass

    rows = []
    ycols = y_columns()
    t0 = time.time()
    for i, ycol in enumerate(ycols, 1):
        b = int(ycol[1])
        cname, mode = ycol.split("__")[1], ycol.split("__")[2]
        pack = D.make_pack(table, feats, ycol, f"rapid__{cname}", ts1, ts2)
        if pack is None:
            continue
        # §7.4: 시험구간 미접촉 assert
        assert pack["idx_va"].max() < ts2 and (pack["idx_va"] >= ts1).all()
        mdl = M.train_tabular(C.SCREEN_MODEL, pack["Xtr"], pack["ytr"],
                              pack["Xva"], pack["yva"])
        pv = M.predict_tabular(mdl, pack["Xva"])
        mt = E.metrics(pack["yva"], pv)
        pers = E.metrics(pack["yva"], E.persistence_smoothed(pack["aux_va"], b))
        center_min = C.CENTERS_STATIC.get(cname)
        in_support = None
        if b in 支 and center_min is not None:
            in_support = 支[b][0] <= center_min <= 支[b][1]
        rows.append({"침전지": b, "중심": cname, "모드": mode, "y": ycol,
                     "검증_sMAPE": round(mt["sMAPE"], 3),
                     "검증_RMSE": round(mt["RMSE"], 4), "검증_MAE": round(mt["MAE"], 4),
                     "검증_R2": round(mt["R2"], 3) if np.isfinite(mt["R2"]) else np.nan,
                     "persistence_sMAPE": round(pers["sMAPE"], 3),
                     "개선폭(%p)": round(pers["sMAPE"] - mt["sMAPE"], 3),
                     "n_train": len(pack["ytr"]), "n_val": mt["n"],
                     "run02_지지범위내": in_support})
        if i % 12 == 0:
            print(f"  {i}/{len(ycols)} ({time.time() - t0:.0f}s) — "
                  f"{ycol}: 검증 sMAPE {mt['sMAPE']:.2f}%")

    res = pd.DataFrame(rows).sort_values(["침전지", "검증_sMAPE"])
    IO.save_csv(res, "정렬스크리닝_검증성능.csv")

    # 침전지별 top-K + persistence 개선 확인
    sel = {}
    for b in (1, 2):
        sub = res[res["침전지"] == b]
        top = sub.head(C.TOP_K_ALIGN)
        sel[str(b)] = top["y"].tolist()
        best = sub.iloc[0]
        beat = sub[sub["개선폭(%p)"] > 0]
        print(f"침전지{b} top-{C.TOP_K_ALIGN}: {sel[str(b)]}")
        print(f"  최저 검증 sMAPE {best['검증_sMAPE']:.2f}% "
              f"(persistence {best['persistence_sMAPE']:.2f}%), "
              f"persistence 개선 조합 {len(beat)}/{len(sub)}")
        if not len(beat):
            print(f"  경고: 침전지{b} — XGB가 persistence를 이기는 정렬조합 없음")
    IO.save_json({"top_k": C.TOP_K_ALIGN, "선정": sel,
                  "스크리닝모델": C.SCREEN_MODEL, "피처셋": "AR포함"},
                 "선정_정렬조합.json")

    # 중심별 최저 sMAPE 곡선
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    order = list(C.CENTERS_STATIC) + C.CENTERS_DYNAMIC
    for ax, b in zip(axes, (1, 2)):
        sub = res[res["침전지"] == b]
        best_by_center = sub.groupby("중심")["검증_sMAPE"].min().reindex(order)
        pers = sub.groupby("중심")["persistence_sMAPE"].median().reindex(order)
        ax.plot(order, best_by_center, "o-", label="XGB 최저 sMAPE")
        ax.plot(order, pers, "s--", color="gray", label="persistence")
        ax.set_title(f"침전지{b}: 중심별 검증 sMAPE"), ax.set_ylabel("sMAPE(%)")
        ax.tick_params(axis="x", rotation=30), ax.legend()
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "스크리닝_중심별_sMAPE.png", dpi=130)
    plt.close(fig)
    print(f"저장: 정렬스크리닝_검증성능.csv, 선정_정렬조합.json "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
