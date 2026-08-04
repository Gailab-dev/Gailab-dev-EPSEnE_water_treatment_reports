# -*- coding: utf-8 -*-
"""정제 통합 데이터의 *_clean 값을 원래 클러스터에 주입해 정제 클러스터 생성.

군집 배정(어느 시점이 어느 클러스터)은 원래 그대로 두고, 피처/타깃 값만
전처리된 값으로 교체한다 → 정제 효과만 순수하게 측정 가능.

산출: dataset/덕남_cluster{0,1,2}_clean.parquet
실행: python -m src.build_clean_clusters
"""
import pandas as pd

from . import common as C

PRE = C.DATA_DIR / "덕남_통합_전처리.parquet"

# 원래 컬럼 → 정제 컬럼 (전처리에서 _clean 생성한 변수 전부)
CLEAN_COLS = [
    "RCS_6.AI.착수정_PH", "RCS_6.AI.착수정_TB", "RCS_6.AI.착수정_AL",
    "RCS_6.AI.착수정_온도", "RCS_6.AI.착수정_전기전도도",
    C.COL_DOSE, C.TARGET_TB[1], C.TARGET_TB[2],
]


def main():
    C.setup_output()
    pre = pd.read_parquet(PRE).sort_index()
    clean = pre[[f"{c}_clean" for c in CLEAN_COLS]]

    for cid in range(3):
        orig = C.load_cluster(cid)
        merged = orig.merge(clean, left_on=C.COL_DT, right_index=True, how="left")
        matched = merged[f"{CLEAN_COLS[0]}_clean"].notna().mean()
        for col in CLEAN_COLS:
            merged[col] = merged[f"{col}_clean"]
        merged = merged.drop(columns=[f"{c}_clean" for c in CLEAN_COLS])
        out = C.DATA_DIR / f"덕남_cluster{cid}_clean.parquet"
        merged.to_parquet(out)
        # 정제 후 피처/타깃 결측률(전처리로 마스킹된 행)
        na = merged[CLEAN_COLS].isna().any(axis=1).mean()
        print(f"cluster{cid}: {len(merged):,}행, 시점매칭 {matched:.1%}, 정제후 결측행 {na:.2%} → {out.name}")


if __name__ == "__main__":
    main()
