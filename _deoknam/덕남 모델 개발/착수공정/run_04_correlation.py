# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 04: 관계분석 보고서 (§6). 학습구간만 사용.

실행: python 착수공정/run_04_correlation.py  (run_03 산출 필요)
산출: output/상관분석_방법별.csv, VIF.csv, 선형회귀_피처.json,
      수질구간_표본수.csv, images/상관_히트맵.png
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import correlate as CR
from intake_coag import datasets as D
from intake_coag import io_utils as IO


def main():
    IO.setup_output()
    table = pd.read_parquet(C.OUTPUT_DIR / "정렬학습테이블_10min.parquet")
    bounds = IO.load_json("분할경계.json")
    ts1 = pd.Timestamp(bounds["ts1"])
    fdef = pd.read_csv(C.OUTPUT_DIR / "피처정의.csv", encoding="utf-8-sig")
    base_feats = fdef[fdef["그룹"] == "base"]["컬럼"].tolist()

    train = table[(table.index < ts1) & D.core_valid(table)]
    print(f"학습구간 {len(train):,}행 (< {ts1})")

    cname, mode = C.REP_ALIGN
    ycol = f"y1__{cname}__{mode}"

    # ── 정렬 유효성: 무정렬 vs 정렬 상관 (착수정_TB 기준) ─────────────
    tb_tag = C.TURBIDITY_COL.split(".")[-1]
    c_unaligned = float(train[tb_tag].corr(train["sed1_tb"], method="spearman"))
    c_aligned = float(train[tb_tag].corr(train[ycol], method="spearman"))
    print(f"착수정_TB↔침전지1: 무정렬 ρ={c_unaligned:.3f} vs 정렬({cname}/{mode}) "
          f"ρ={c_aligned:.3f} {'(상승 — 정렬 유효)' if c_aligned > c_unaligned else '(경고: 미상승)'}")

    # ── 방법별 상관 (대표 정렬 y + 현재 침전지 탁도) ──────────────────
    ct = CR.corr_table(train, base_feats, ycol)
    ct.insert(1, "타깃", ycol)
    IO.save_csv(ct, "상관분석_방법별.csv")

    # ── VIF 축약 (다변수회귀 입력, §6 VIF 점검) ───────────────────────
    lin_feats, vif_tab = CR.vif_reduce(train, base_feats)
    IO.save_csv(vif_tab, "VIF.csv")
    IO.save_json({"대표정렬": list(C.REP_ALIGN), "피처": lin_feats},
                 "선형회귀_피처.json")
    print(f"VIF≤{C.VIF_MAX} 축약: {len(base_feats)} → {len(lin_feats)}개")

    # ── 수질구간 표본수 ───────────────────────────────────────────────
    IO.save_csv(CR.bin_counts(train), "수질구간_표본수.csv")

    # ── 히트맵 (|Spearman| 상위 25 피처) ──────────────────────────────
    import matplotlib.pyplot as plt

    top = ct.head(25)["피처"].tolist()
    sub = train[top + [ycol]].astype(float)
    R = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(R.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(R))), ax.set_xticklabels(R.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(R))), ax.set_yticklabels(R.columns, fontsize=7)
    fig.colorbar(im, shrink=0.7)
    ax.set_title(f"Spearman 상관 (상위 25 피처 + {ycol})")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "상관_히트맵.png", dpi=130)
    plt.close(fig)
    print("저장: 상관분석_방법별.csv, VIF.csv, 선형회귀_피처.json, "
          "수질구간_표본수.csv, images/상관_히트맵.png")


if __name__ == "__main__":
    main()
