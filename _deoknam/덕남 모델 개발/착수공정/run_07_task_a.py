# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 07: 과제 A(설정값 모사, §7.1) + ε-sMAPE 게이트.

dose_compare 패키지 함수 직접 재사용(원 코드 무수정) — 신규 스펙 적용:
전처리완료 전체 데이터(cluster_label=0 상수) + 공통 분할경계 + ε 고정 sMAPE 게이트.
학습행은 과제 B와 동일한 10분 스트라이드(§7.3 동일 규칙).
선택 프로토콜: 모델별 (피처셋×창) 검증 sMAPE 선택 → 재학습 → 시험 1회(TestGuard).
persistence(직전 설정 유지)는 계단형 설정값에 매우 강함 — §7.1 모사 성능≠최적 주입률.

실행: python 착수공정/run_07_task_a.py  (run_03 산출 필요)
산출: output/과제A_검증성능_전조합.csv, 과제A_시험성능표.csv, 과제A_게이트판정.csv,
      images/과제A_비교_sMAPE.png
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import evaluate as E
from intake_coag import io_utils as IO
from intake_coag import models as M

IO.add_coag_path()
from dose_compare import config as DC          # noqa: E402
from dose_compare import datasets as DD        # noqa: E402
from dose_compare import features as DF        # noqa: E402

GUARD = E.TestGuard()

TARGETS = {
    "PAC목표주입량": ("PAC.AI.PAC_목표주입량", 0.1, 250.0),
    "PAC주입률SV": ("PAC.AI.PAC_주입율제어_SV", 10.0, 40.0),
    "PACS2목표주입량": ("PAC.AI.PACS2_목표주입량", 0.1, 250.0),
    "PACS2주입률SV": ("PAC.AI.PACS2_주입율제어_SV", 10.0, 40.0),
}


def build_target_tables(df, flags, tcol, tmin, tmax):
    """dose_compare 피처·품질필터 → (10분 스트라이드 태뷸러, 전체 1분 테이블, 피처셋)."""
    table, feat_ar, feat_noar = DF.build_features(df, flags, target_col=tcol)
    table_q, reasons = DF.quality_filter(table, tmin, tmax)
    stride = table_q[table_q.index.minute % C.STRIDE_MIN == 0]
    return stride, table, {"AR포함": feat_ar, "AR제외": feat_noar}, reasons


def tab_pack(stride, feats, ts1, ts2):
    packs = DD.tabular_by_cluster(stride, feats, ts1, ts2)
    return packs.get(0)


def seq_pack(table, sfeats, w, ts1, ts2):
    rs = DD.resample_for_seq(table, sfeats)
    packs = DD.sequences_by_cluster(rs, sfeats, w, ts1, ts2)
    return packs.get(0)


def main():
    IO.setup_output()
    bounds = IO.load_json("분할경계.json")
    ts1, ts2 = pd.Timestamp(bounds["ts1"]), pd.Timestamp(bounds["ts2"])
    df = pd.read_parquet(C.DATA_PARQUET).sort_index()
    df[DC.LABEL_COL] = 0                        # 전체 데이터 단일 군집(§7.3 공통 분할)
    flags = IO.load_flags()
    if flags is not None:
        flags = flags.reindex(df.index)
    print(f"로드: {len(df):,}행 / 분할 {ts1} · {ts2}")

    val_rows, test_rows, qual = [], [], []
    t0 = time.time()
    for tname, (tcol, tmin, tmax) in TARGETS.items():
        stride, full, fsets, reasons = build_target_tables(df, flags, tcol, tmin, tmax)
        qual.append({"타깃": tname, **reasons})
        seq_feats = {fs: [f for f in fsets["AR제외"]
                          if not any(s in f for s in ("__lag", "__d", "__ma"))]
                     + (["SV_prev"] if fs == "AR포함" else [])
                     for fs in C.FEATURE_SETS}

        # ── ① 전 조합 검증 성능 ──────────────────────────────────────
        for fs in C.FEATURE_SETS:
            p = tab_pack(stride, fsets[fs], ts1, ts2)
            if p is None:
                continue
            for name in C.MODELS_TABULAR:
                mdl = M.train_tabular(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                mt = E.metrics(p["yva"], M.predict_tabular(mdl, p["Xva"]))
                val_rows.append({"타깃": tname, "모델": name, "피처셋": fs, "창": "-",
                                 "검증_sMAPE": mt["sMAPE"], "검증_RMSE": mt["RMSE"],
                                 "검증_MAE": mt["MAE"], "검증_R2": mt["R2"]})
                del mdl
            if M._TORCH:
                for w in C.SEQ_WINDOWS:
                    sp = seq_pack(full, seq_feats[fs], w, ts1, ts2)
                    if sp is None:
                        continue
                    for name in C.MODELS_SEQ:
                        mdl = M.train_seq(name, sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"])
                        mt = E.metrics(sp["yva"], M.predict_seq(mdl, sp["Xva"]))
                        val_rows.append({"타깃": tname, "모델": name, "피처셋": fs, "창": w,
                                         "검증_sMAPE": mt["sMAPE"], "검증_RMSE": mt["RMSE"],
                                         "검증_MAE": mt["MAE"], "검증_R2": mt["R2"]})
                        del mdl
        print(f"  [{tname}] 검증 스윕 완료 ({time.time() - t0:.0f}s)")

        # ── ② 선택 고정 → 재학습 → 시험 1회 ──────────────────────────
        vsub = pd.DataFrame([r for r in val_rows if r["타깃"] == tname])
        for name in C.MODELS_TABULAR + (C.MODELS_SEQ if M._TORCH else []):
            rows_m = vsub[vsub["모델"] == name]
            if not len(rows_m):
                continue
            cfg = rows_m.sort_values(["검증_sMAPE", "검증_RMSE", "검증_MAE"]).iloc[0]
            fs, w = cfg["피처셋"], cfg["창"]
            GUARD.check(("과제A", tname, name))
            if name in C.MODELS_TABULAR:
                p = tab_pack(stride, fsets[fs], ts1, ts2)
                mdl = M.train_tabular(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                pt = M.predict_tabular(mdl, p["Xte"])
                yte = p["yte"]
            else:
                sp = seq_pack(full, seq_feats[fs], int(w), ts1, ts2)
                mdl = M.train_seq(name, sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"])
                pt = M.predict_seq(mdl, sp["Xte"])
                yte = sp["yte"]
            mt = E.metrics(yte, pt)
            test_rows.append({"목표변수": tname, "모델": name, "피처셋": fs, "창": w,
                              "RMSE": mt["RMSE"], "MAE": mt["MAE"], "sMAPE": mt["sMAPE"],
                              "R2": mt["R2"], "n": mt["n"],
                              "게이트(≤5%)": "통과" if E.gate_pass(mt["sMAPE"]) else "미통과"})
            print(f"  [{tname}] {name}: 시험 sMAPE {mt['sMAPE']:.2f}% "
                  f"→ {'통과' if E.gate_pass(mt['sMAPE']) else '미통과'}")
            del mdl
        # persistence(직전 설정 유지) — §7.1 계단형에 강함 명시
        p = tab_pack(stride, fsets["AR포함"], ts1, ts2)
        if p is not None and p.get("sv_prev_te") is not None:
            GUARD.check(("과제A", tname, "persistence"))
            mt = E.metrics(p["yte"], p["sv_prev_te"])
            test_rows.append({"목표변수": tname, "모델": "persistence(직전 유지)",
                              "피처셋": "-", "창": "-", "RMSE": mt["RMSE"], "MAE": mt["MAE"],
                              "sMAPE": mt["sMAPE"], "R2": mt["R2"], "n": mt["n"],
                              "게이트(≤5%)": "통과" if E.gate_pass(mt["sMAPE"]) else "미통과"})

    IO.save_csv(pd.DataFrame(val_rows).round(4), "과제A_검증성능_전조합.csv")
    test_df = pd.DataFrame(test_rows).round(4)
    IO.save_csv(test_df, "과제A_시험성능표.csv")
    gate = test_df[["목표변수", "모델", "피처셋", "창", "sMAPE", "게이트(≤5%)"]].copy()
    gate["기준"] = (f"시험 sMAPE ≤ {C.GATE_SMAPE}% (원단위, §7.3) — "
                   "모사 성능이며 최적 주입률 아님(§7.1)")
    IO.save_csv(gate, "과제A_게이트판정.csv")
    IO.save_csv(pd.DataFrame(qual), "과제A_품질필터_집계.csv")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    piv = test_df.pivot_table(index="목표변수", columns="모델", values="sMAPE")
    piv.plot.bar(ax=ax, color=[C.MODEL_COLORS.get(c, "#777")
                               for c in piv.columns])
    ax.axhline(C.GATE_SMAPE, color="k", ls="--", label=f"게이트 {C.GATE_SMAPE}%")
    ax.set_ylabel("시험 sMAPE(%)"), ax.set_title("과제A 설정값 모사 — 시험 성능")
    ax.legend(ncols=4, fontsize=8), plt.xticks(rotation=10), fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "과제A_비교_sMAPE.png", dpi=130)
    plt.close(fig)
    print(f"\n저장: 과제A_시험성능표.csv 등 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
