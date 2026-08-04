# -*- coding: utf-8 -*-
"""덕남 응집제 주입률(SV) 직접예측 — 군집별 5모델 비교 실행 (설계문서 1단계).

실행: python 02_coag/run_dose_compare.py
산출: output/dose_compare/모델비교표.csv, 구간별오차.csv, 품질필터_집계.csv, images/*.png
노트북(dose_compare.ipynb)에서 run()을 import해 동일 결과·시각화 재현.
"""
import sys
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from dose_compare import config as C
from dose_compare import features as F
from dose_compare import datasets as D
from dose_compare import models as M
from dose_compare import evaluate as E
from dose_compare import visualization as V


def _seed():
    random.seed(C.RANDOM_STATE)
    np.random.seed(C.RANDOM_STATE)


def _seq_base_feats(feat_noar):
    return [f for f in feat_noar
            if not any(s in f for s in ("__lag", "__d", "__ma"))]


def _run_one_target(df, flags, tname, tcol, tmin, tmax,
                    tabular, seq, windows, feature_sets, verbose):
    """단일 타깃에 대한 군집×모델 비교. 반환 rows·seg·preds·imp·quality."""
    table, feat_ar, feat_noar = F.build_features(df, flags, target_col=tcol)
    table_q, reasons = F.quality_filter(table, tmin, tmax)
    reasons = {"타깃": tname, **reasons}
    if verbose:
        print(f"[{tname}] 품질필터: 유효 {reasons['최종유효']:,}/{reasons['전체']:,}")
    ts1, ts2 = D.split_times(table_q)
    feat_map = {"AR포함": feat_ar, "AR제외": feat_noar}
    rows, seg_rows, preds_store, imp_store = [], [], {}, {}

    # persistence 베이스라인 (직전값 유지)
    for cid in C.CLUSTERS:
        sub = table_q[(table_q[C.LABEL_COL] == cid) & (table_q.index >= ts2)]
        sub = sub.dropna(subset=["__target__", "SV_prev"])
        if len(sub) >= 50:
            mt = E.metrics(sub["__target__"], sub["SV_prev"])
            rows.append(dict(타깃=tname, 군집=cid, 모델="persistence", 피처셋="-", 창="-", **mt))

    if tabular:
        for fs in feature_sets:
            packs = D.tabular_by_cluster(table_q, feat_map[fs], ts1, ts2)
            for cid in C.CLUSTERS:
                p = packs[cid]
                if p is None:
                    continue
                for name in C.MODELS_TABULAR:
                    mdl = M.train_tabular(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                    yp = M.predict_tabular(mdl, p["Xte"])
                    mt = E.metrics(p["yte"], yp)
                    cpe = E.change_point_error(p["aux_te"], yp)
                    rows.append(dict(타깃=tname, 군집=cid, 모델=name, 피처셋=fs, 창="-",
                                     변경직후MAE=cpe, **mt))
                    if verbose:
                        print(f"  [{tname}·태뷸러] c{cid} {name} {fs}: "
                              f"MAE={mt['MAE']:.3f} R2={mt['R2']}")
                    if fs == "AR제외":
                        preds_store.setdefault(cid, {"idx": p["idx_te"], "y": p["yte"], "preds": {}})
                        preds_store[cid]["preds"][name] = yp
                        sg = E.segment_errors(p["aux_te"], yp)
                        sg.insert(0, "모델", name); sg.insert(0, "군집", cid); sg.insert(0, "타깃", tname)
                        seg_rows.append(sg)
                        if hasattr(mdl, "feature_importances_"):
                            imp_store[(cid, name)] = pd.Series(
                                mdl.feature_importances_, index=p["feat_cols"])

    if seq and M._TORCH:
        base = _seq_base_feats(feat_noar)
        for fs in feature_sets:
            sfeats = base + (["SV_prev"] if fs == "AR포함" else [])
            rs = D.resample_for_seq(table, sfeats)
            for w in windows:
                packs = D.sequences_by_cluster(rs, sfeats, w, ts1, ts2)
                for cid in C.CLUSTERS:
                    p = packs[cid]
                    if p is None:
                        continue
                    for name in C.MODELS_SEQ:
                        mdl = M.train_seq(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                        yp = M.predict_seq(mdl, p["Xte"])
                        mt = E.metrics(p["yte"], yp)
                        rows.append(dict(타깃=tname, 군집=cid, 모델=name, 피처셋=fs, 창=w, **mt))
                        if verbose:
                            print(f"  [{tname}·시퀀스] c{cid} {name} {fs} 창{w}: "
                                  f"MAE={mt['MAE']:.3f} R2={mt['R2']}")

    return rows, seg_rows, preds_store, imp_store, reasons


def run(targets=None, tabular=True, seq=True, windows=None, feature_sets=None,
        save=True, verbose=True):
    """전체 비교 실행(다중 타깃). 반환: dict(summary, seg, quality, preds, importances)."""
    _seed()
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    windows = windows or C.SEQ_WINDOWS
    feature_sets = feature_sets or C.FEATURE_SETS
    targets = targets or C.TARGETS
    if seq and not M._TORCH:
        print("  torch 미설치 — 시퀀스(GRU/LSTM) 건너뜀")

    df, flags = F.load_labeled()
    if verbose:
        print(f"로드: {C.LABELED_PARQUET.name} — {len(df):,}행 / 타깃 {list(targets)}")

    all_rows, all_seg, all_q, preds_all, imp_all = [], [], [], {}, {}
    for tname, (tcol, tmin, tmax) in targets.items():
        rows, seg_rows, preds, imp, reasons = _run_one_target(
            df, flags, tname, tcol, tmin, tmax, tabular, seq, windows, feature_sets, verbose)
        all_rows += rows
        all_seg += seg_rows
        all_q.append(reasons)
        preds_all[tname] = preds
        for k, v in imp.items():
            imp_all[(tname, *k)] = v

    summary = pd.DataFrame(all_rows)
    seg = pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame()
    quality = pd.DataFrame(all_q)

    if save:
        summary.to_csv(C.OUTPUT_DIR / "모델비교표.csv", index=False, encoding="utf-8-sig")
        seg.to_csv(C.OUTPUT_DIR / "구간별오차.csv", index=False, encoding="utf-8-sig")
        quality.to_csv(C.OUTPUT_DIR / "품질필터_집계.csv", index=False, encoding="utf-8-sig")
        for tname in targets:
            sub = summary[summary["타깃"] == tname]
            for metric in ("MAE", "R2"):
                try:
                    V.plot_metric_bars(sub, metric, C.IMAGES_DIR / f"{tname}_비교_{metric}.png")
                except Exception as e:
                    print(f"  plot {tname}/{metric} 스킵: {e}")
            for cid, store in preds_all[tname].items():
                V.plot_pred_vs_actual(store["idx"], store["y"], store["preds"], cid,
                                      C.IMAGES_DIR / f"{tname}_cluster{cid}_실제vs예측.png")
        for (tname, cid, name), imp in imp_all.items():
            V.plot_importance_series(imp, cid, name,
                                     C.IMAGES_DIR / f"{tname}_cluster{cid}_{name}_중요도.png")
        print(f"\n저장: {C.OUTPUT_DIR}")

    return dict(summary=summary, seg=seg, quality=quality,
                preds=preds_all, importances=imp_all)


if __name__ == "__main__":
    res = run()
    print("\n=== 모델비교표 (요약) ===")
    s = res["summary"]
    cols = [c for c in ["타깃", "군집", "모델", "피처셋", "창", "MAE", "RMSE", "R2", "MAPE", "n"] if c in s.columns]
    print(s[cols].to_string(index=False))
