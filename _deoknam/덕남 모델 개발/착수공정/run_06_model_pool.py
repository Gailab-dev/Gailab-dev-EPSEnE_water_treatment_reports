# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 06: 과제 B(공정반응) 5모델 풀비교 (§7).

프로토콜(§7.4): ① top-3 정렬 × 5모델 × 피처셋 × (DL 창) 전 조합을 학습하고
**검증 sMAPE만 기록**(모델 폐기) → ② 모델별 최적 조합을 검증 기준으로 고정 →
동일 시드 재학습 → **시험구간 1회 평가**(TestGuard로 재평가 차단).
주입 입력은 proxy(§3.3 — 실투입 라벨 미확보).

실행: python 착수공정/run_06_model_pool.py  (run_03·04·05 산출 필요)
산출: output/과제B_검증성능_전조합.csv, 과제B_시험성능표.csv, 과제B_구간별성능.csv,
      과제B_게이트판정.csv, models/과제B_*, images/과제B_*.png
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib
import numpy as np
import pandas as pd

from intake_coag import config as C
from intake_coag import correlate as CR
from intake_coag import datasets as D
from intake_coag import evaluate as E
from intake_coag import features as F
from intake_coag import io_utils as IO
from intake_coag import models as M

GUARD = E.TestGuard()


def load_inputs():
    table = pd.read_parquet(C.OUTPUT_DIR / "정렬학습테이블_10min.parquet")
    bounds = IO.load_json("분할경계.json")
    ts1, ts2 = pd.Timestamp(bounds["ts1"]), pd.Timestamp(bounds["ts2"])
    fdef = pd.read_csv(C.OUTPUT_DIR / "피처정의.csv", encoding="utf-8-sig")
    base = fdef[fdef["그룹"] == "base"]["컬럼"].tolist()
    sed_ar = fdef[fdef["그룹"] == "sed_ar"]["컬럼"].tolist()
    sel = IO.load_json("선정_정렬조합.json")["선정"]
    lin = IO.load_json("선형회귀_피처.json")["피처"]
    return table, ts1, ts2, base, sed_ar, sel, lin


def feature_sets(table, ts1, base, sed_ar, lin):
    """모델종류×피처셋 → 피처 리스트. LR은 VIF 축약(§6), AR포함은 sed-AR 추가."""
    train = table[table.index < ts1]
    lin_ar, _ = CR.vif_reduce(train, lin + sed_ar)          # LR AR포함도 VIF 축약
    tab = {"AR포함": base + sed_ar, "AR제외": base}
    linf = {"AR포함": lin_ar, "AR제외": lin}
    seq_base = F.seq_base_feats(base)
    seq_ar = [f for f in sed_ar if f.endswith(("_tb", "_sm"))]
    seq = {"AR포함": seq_base + seq_ar, "AR제외": seq_base}
    return tab, linf, seq


def main():
    IO.setup_output()
    table, ts1, ts2, base, sed_ar, sel, lin = load_inputs()
    tabf, linf, seqf = feature_sets(table, ts1, base, sed_ar, lin)
    print(f"피처셋: 태뷸러 AR포함 {len(tabf['AR포함'])} / AR제외 {len(tabf['AR제외'])} / "
          f"LR {len(linf['AR포함'])}·{len(linf['AR제외'])} / 시퀀스 {len(seqf['AR포함'])}·{len(seqf['AR제외'])}")

    # ── ① 전 조합 검증 성능 ──────────────────────────────────────────
    val_rows = []
    t0 = time.time()
    for b in (1, 2):
        for ycol in sel[str(b)]:
            cname = ycol.split("__")[1]
            rapid = f"rapid__{cname}"
            for fs in C.FEATURE_SETS:
                pack = D.make_pack(table, tabf[fs], ycol, rapid, ts1, ts2)
                lpack = D.make_pack(table, linf[fs], ycol, rapid, ts1, ts2)
                if pack is None:
                    continue
                for name in C.MODELS_TABULAR:
                    p = lpack if name == "LinearRegression" else pack
                    tt = time.time()
                    mdl = M.train_tabular(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                    fit_s = time.time() - tt
                    mt = E.metrics(p["yva"], M.predict_tabular(mdl, p["Xva"]))
                    val_rows.append({"침전지": b, "y": ycol, "모델": name, "피처셋": fs,
                                     "창": "-", "검증_sMAPE": mt["sMAPE"],
                                     "검증_RMSE": mt["RMSE"], "검증_MAE": mt["MAE"],
                                     "검증_R2": mt["R2"], "학습초": round(fit_s, 1)})
                    del mdl
                if M._TORCH:
                    for w in C.SEQ_WINDOWS:
                        sp = D.make_sequences(table, seqf[fs], ycol, rapid, w, ts1, ts2)
                        if sp is None:
                            continue
                        for name in C.MODELS_SEQ:
                            tt = time.time()
                            mdl = M.train_seq(name, sp["Xtr"], sp["ytr"],
                                              sp["Xva"], sp["yva"])
                            fit_s = time.time() - tt
                            mt = E.metrics(sp["yva"], M.predict_seq(mdl, sp["Xva"]))
                            val_rows.append({"침전지": b, "y": ycol, "모델": name,
                                             "피처셋": fs, "창": w,
                                             "검증_sMAPE": mt["sMAPE"],
                                             "검증_RMSE": mt["RMSE"], "검증_MAE": mt["MAE"],
                                             "검증_R2": mt["R2"], "학습초": round(fit_s, 1)})
                            del mdl
            print(f"  침전지{b} {ycol} 완료 ({time.time() - t0:.0f}s)")
    val_df = pd.DataFrame(val_rows)
    IO.save_csv(val_df.round(4), "과제B_검증성능_전조합.csv")

    # ── ② 모델별 최적 조합 고정(검증 sMAPE, 동률 시 RMSE·MAE — §4.5-5) → 재학습 → 시험 1회 ──
    test_rows, seg_frames, best_pred = [], [], {}
    for b in (1, 2):
        sub = val_df[val_df["침전지"] == b]
        for name in C.MODELS_TABULAR + (C.MODELS_SEQ if M._TORCH else []):
            cfg = sub[sub["모델"] == name].sort_values(
                ["검증_sMAPE", "검증_RMSE", "검증_MAE"]).iloc[0]
            ycol, fs, w = cfg["y"], cfg["피처셋"], cfg["창"]
            rapid = f"rapid__{ycol.split('__')[1]}"
            GUARD.check(("과제B", f"침전지{b}", name))
            if name in C.MODELS_TABULAR:
                fset = linf[fs] if name == "LinearRegression" else tabf[fs]
                p = D.make_pack(table, fset, ycol, rapid, ts1, ts2)
                mdl = M.train_tabular(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                pv, pt = M.predict_tabular(mdl, p["Xva"]), M.predict_tabular(mdl, p["Xte"])
                tt = time.time(); M.predict_tabular(mdl, p["Xte"]); inf_s = time.time() - tt
                aux_te, yte, yva = p["aux_te"], p["yte"], p["yva"]
                joblib.dump({"model": mdl, "feat_cols": p["feat_cols"], "fill": p["fill"],
                             "mu": p["mu"], "sd": p["sd"],
                             "meta": {"y": ycol, "피처셋": fs, "모델": name}},
                            C.MODELS_DIR / f"과제B_침전지{b}_{name}.joblib")
            else:
                fset = seqf[fs]
                p = D.make_sequences(table, fset, ycol, rapid, int(w), ts1, ts2)
                mdl = M.train_seq(name, p["Xtr"], p["ytr"], p["Xva"], p["yva"])
                pv, pt = M.predict_seq(mdl, p["Xva"]), M.predict_seq(mdl, p["Xte"])
                tt = time.time(); M.predict_seq(mdl, p["Xte"]); inf_s = time.time() - tt
                yte, yva = p["yte"], p["yva"]
                aux_te = table.reindex(p["idx_te"])          # 구간별 분석용 원본행
                import torch
                torch.save(mdl, C.MODELS_DIR / f"과제B_침전지{b}_{name}.pt")
            mt = E.metrics(yte, pt)
            iv = E.interval_stats(yva, pv, yte, pt)
            excl = (p.get("n_excluded", np.nan) / p.get("n_total", np.nan)
                    if name in C.MODELS_TABULAR else np.nan)
            test_rows.append({"목표변수": f"침전지{b}", "정렬": ycol, "모델": name,
                              "피처셋": fs, "창": w,
                              "RMSE": mt["RMSE"], "MAE": mt["MAE"],
                              "sMAPE": mt["sMAPE"], "R2": mt["R2"], "n": mt["n"],
                              "포함률95": iv["포함률95"], "구간폭": iv["구간폭"],
                              "추론초": round(inf_s, 2),
                              "제외율": round(float(excl), 4) if np.isfinite(excl) else np.nan,
                              "게이트(≤5%)": "통과" if E.gate_pass(mt["sMAPE"]) else "미통과"})
            sg = E.segment_metrics(aux_te, yte, pt)
            sg.insert(0, "모델", name); sg.insert(0, "목표변수", f"침전지{b}")
            seg_frames.append(sg)
            best_pred[(b, name)] = dict(idx=p["idx_te"], y=yte, p=pt, ycol=ycol)
            print(f"침전지{b} {name}: 시험 sMAPE {mt['sMAPE']:.2f}% "
                  f"({ycol}/{fs}/창{w}) → {'통과' if E.gate_pass(mt['sMAPE']) else '미통과'}")
        # persistence 기준선(시험) — 침전지별 best 모델의 정렬 y 기준
        best_name = sub.sort_values("검증_sMAPE").iloc[0]["모델"]
        st = best_pred[(b, best_name)]
        aux = table.reindex(st["idx"])
        GUARD.check(("과제B", f"침전지{b}", "persistence"))
        pers = E.metrics(st["y"], E.persistence_smoothed(aux, b))
        test_rows.append({"목표변수": f"침전지{b}", "정렬": st["ycol"], "모델": "persistence",
                          "피처셋": "-", "창": "-", "RMSE": pers["RMSE"], "MAE": pers["MAE"],
                          "sMAPE": pers["sMAPE"], "R2": pers["R2"], "n": pers["n"],
                          "게이트(≤5%)": "통과" if E.gate_pass(pers["sMAPE"]) else "미통과"})

    # ── TB_outlet = max(ŷ1, ŷ2) 보수 지표 (§5.2) ─────────────────────
    for name in C.MODELS_TABULAR + (C.MODELS_SEQ if M._TORCH else []):
        s1, s2 = best_pred.get((1, name)), best_pred.get((2, name))
        if s1 is None or s2 is None:
            continue
        common = s1["idx"].intersection(s2["idx"])
        if len(common) < C.MIN_ROWS:
            continue
        i1 = pd.Series(range(len(s1["idx"])), index=s1["idx"]).reindex(common).to_numpy(int)
        i2 = pd.Series(range(len(s2["idx"])), index=s2["idx"]).reindex(common).to_numpy(int)
        y_out = np.maximum(s1["y"][i1], s2["y"][i2])
        p_out = np.maximum(s1["p"][i1], s2["p"][i2])
        GUARD.check(("과제B", "TB_outlet", name))
        mt = E.metrics(y_out, p_out)
        test_rows.append({"목표변수": "TB_outlet(max)", "정렬": f"{s1['ycol']}|{s2['ycol']}",
                          "모델": name, "피처셋": "-", "창": "-",
                          "RMSE": mt["RMSE"], "MAE": mt["MAE"], "sMAPE": mt["sMAPE"],
                          "R2": mt["R2"], "n": mt["n"],
                          "게이트(≤5%)": "통과" if E.gate_pass(mt["sMAPE"]) else "미통과"})

    test_df = pd.DataFrame(test_rows).round(4)
    IO.save_csv(test_df, "과제B_시험성능표.csv")
    seg = pd.concat(seg_frames, ignore_index=True).round(4)
    IO.save_csv(seg, "과제B_구간별성능.csv")
    gate = test_df[["목표변수", "정렬", "모델", "피처셋", "창", "sMAPE", "게이트(≤5%)"]].copy()
    gate["기준"] = f"시험 sMAPE ≤ {C.GATE_SMAPE}% (ε={C.SMAPE_EPS} NTU 고정, §7.4)"
    IO.save_csv(gate, "과제B_게이트판정.csv")

    # ── 그림 ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    md = test_df[test_df["목표변수"].str.startswith("침전지")]
    for i, b in enumerate((1, 2)):
        sub = md[md["목표변수"] == f"침전지{b}"].set_index("모델")["sMAPE"]
        xs = np.arange(len(sub)) + i * 0.35
        ax.bar(xs, sub.to_numpy(), 0.33,
               label=f"침전지{b}", color=["#4878b0", "#c44e52"][i])
        if i == 0:
            ax.set_xticks(np.arange(len(sub)) + 0.17)
            ax.set_xticklabels(sub.index, rotation=20)
    ax.axhline(C.GATE_SMAPE, color="k", ls="--", label=f"게이트 {C.GATE_SMAPE}%")
    ax.set_ylabel("시험 sMAPE(%)"), ax.set_title("과제B 시험 성능 (모델별 최적 정렬조합)")
    ax.legend(), fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "과제B_모델비교_sMAPE.png", dpi=130)
    plt.close(fig)

    for b in (1, 2):
        names = [n for (bb, n) in best_pred if bb == b]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(13, 4.5))
        st0 = best_pred[(b, names[0])]
        ax.plot(st0["idx"], st0["y"], color="k", lw=0.8, label="실측")
        for n in names:
            st = best_pred[(b, n)]
            ax.plot(st["idx"], st["p"], lw=0.7, alpha=0.8,
                    color=C.MODEL_COLORS.get(n), label=n)
        ax.set_title(f"침전지{b} 시험구간 실측 vs 예측"), ax.set_ylabel("탁도(NTU)")
        ax.legend(ncols=6, fontsize=8), fig.tight_layout()
        fig.savefig(C.IMAGES_DIR / f"과제B_침전지{b}_실측vs예측.png", dpi=130)
        plt.close(fig)

    print(f"\n저장: 과제B_시험성능표.csv 등 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
