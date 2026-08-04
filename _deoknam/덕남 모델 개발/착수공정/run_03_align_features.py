# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 03: 시간정렬(84 y) + 피처 생성 → 학습테이블·분할경계 (§5·§6).

실행: python 착수공정/run_03_align_features.py
산출: output/정렬학습테이블_10min.parquet (태뷸러·시퀀스 공용),
      피처정의.csv, 제외행_집계.csv, 정렬_커버리지.csv, 분할경계.json
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import align as A
from intake_coag import config as C
from intake_coag import datasets as D
from intake_coag import features as F
from intake_coag import io_utils as IO


def main():
    IO.setup_output()
    df = IO.load_master()
    flags = IO.load_flags()
    print(f"로드: {len(df):,}행 1분 정규격자 ({df.index.min()} ~ {df.index.max()})")

    # ── 시간정렬 타깃 84종 + 급변플래그 (1분 격자) ────────────────────
    targets, tstats = A.build_targets(df)
    A.verify_alignment(df, targets)           # 누설 방지 표본 대조 assert
    print(f"타깃 생성: y {sum(c.startswith('y') for c in targets.columns)}종, "
          f"급변플래그 {sum(c.startswith('rapid') for c in targets.columns)}종, "
          f"침전지2 운휴 {tstats['idle2_rows']:,}행 마스킹")

    # ── 피처 (1분 계산 → 10분 스트라이드 저장) ────────────────────────
    stride_idx = df.index[df.index.minute % C.STRIDE_MIN == 0]
    feats, base_feats, sed_ar = F.build_features(df, flags, stride_idx)
    print(f"피처: base {len(base_feats)}개 + sed-AR {len(sed_ar)}개 "
          f"(스트라이드 {C.STRIDE_MIN}분 → {len(feats):,}행)")

    table = feats.join(targets.reindex(stride_idx))
    # rapid 플래그는 bool 유지
    for c in table.columns:
        if c.startswith("rapid__"):
            table[c] = table[c].astype(bool)

    # ── 분할경계 (핵심 유효행 기준, 전 과제 공용 §7.3) ────────────────
    valid = D.core_valid(table)
    ts1, ts2 = D.split_boundaries(table.index[valid])
    IO.save_json({"ts1": str(ts1), "ts2": str(ts2),
                  "n_valid": int(valid.sum()), "n_total": len(table),
                  "train": C.SPLIT["train"], "val": C.SPLIT["val"],
                  "test": C.SPLIT["test"]}, "분할경계.json")
    print(f"분할경계: train<{ts1} ≤val< {ts2} ≤test (유효 {valid.sum():,}행)")

    # ── 저장 ──────────────────────────────────────────────────────────
    out_path = C.OUTPUT_DIR / "정렬학습테이블_10min.parquet"
    table.to_parquet(out_path)
    print(f"저장: {out_path.name} — {table.shape[0]:,}행 × {table.shape[1]}열")

    fdef = pd.DataFrame(
        [(f, "base") for f in base_feats] + [(f, "sed_ar") for f in sed_ar]
        + [(c, "y") for c in table.columns if c.startswith("y")]
        + [(c, "rapid_flag") for c in table.columns if c.startswith("rapid")],
        columns=["컬럼", "그룹"])
    IO.save_csv(fdef, "피처정의.csv")

    # 제외행 집계 (§5.3 규칙별)
    ph_tag = C.PH_COL.split(".")[-1]
    excl = pd.DataFrame([
        {"사유": "핵심(원수5+유량) 결측", "행수": int((~table[D.CORE_TAGS].notna().all(axis=1)).sum())},
        {"사유": "pH=0 무효", "행수": int((table[ph_tag] == 0).sum())},
        *[{"사유": f"급변행({c.split('__')[1]})", "행수": int(table[c].sum())}
          for c in table.columns if c.startswith("rapid__")],
        {"사유": "침전지2 운휴(1분 격자)", "행수": tstats["idle2_rows"]},
        {"사유": "전체(스트라이드)", "행수": len(table)},
    ])
    IO.save_csv(excl, "제외행_집계.csv")

    # 정렬조합별 y 커버리지 (481분 말단 손실 등 §5.2)
    cov = pd.DataFrame([{"y": k, "비결측률": v} for k, v in tstats["coverage"].items()])
    IO.save_csv(cov, "정렬_커버리지.csv")
    lo = cov["비결측률"].min()
    print(f"y 커버리지: 최저 {lo:.3f} ({cov.loc[cov['비결측률'].idxmin(), 'y']}) "
          f"/ 최고 {cov['비결측률'].max():.3f}")
    print("저장: 피처정의.csv, 제외행_집계.csv, 정렬_커버리지.csv, 분할경계.json")


if __name__ == "__main__":
    main()
