# -*- coding: utf-8 -*-
"""Task 5: 피처 보강 + 6개 모델 후보군 최종 비교 (+ 통합 vs 군집 대조).

- 데이터: 정제 통합 데이터 10분 리샘플 (_clean 컬럼)
- 입력피처(원시 8): 원수성상 5 + PAC SV + 원수유량 FT101 + 해당 침전지 탁도(AR)
- 파생피처(9): 이동평균·변화율 — 전부 t 이전 창만 사용(누수 없음)
- 모델 6종: ExtraTrees / RandomForest / XGBoost / LightGBM / GradientBoosting / Ridge
- 평가: 확장 윈도우 롤링 재학습 백테스트(초기 50%, 재학습 1주 + 고정), R²·SMAPE

실행:
  python -m src.task5_final_models          # 통합 6모델 비교
  python -m src.task5_final_models cluster  # 통합 vs 군집 대조(최고 모델)
"""
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import common as C

PRE = C.DATA_DIR / "덕남_통합_전처리.parquet"
LEAD = 150  # task4에서 확인한 최적 체류시간(분)
SCHEDULES = {"1주": 1, "고정": None}
INIT_FRAC = 0.5
RS = C.RANDOM_STATE

# 군집 대조용: 클러스터별 기존 최적 lead (task3)
CLUSTER_LEADS = {0: 180, 1: 150, 2: 210}

# 입력피처(원시 계측값). AR(침전지 탁도)은 basin별로 추가.
BASE_INPUTS = C.RAW_FEATURES + [C.COL_DOSE, C.COL_FLOW]


def get_models():
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, n_jobs=-1, random_state=RS),
        "RandomForest": RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=RS),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=RS,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RS, verbose=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS
        ),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def add_derived(df: pd.DataFrame, basin: int) -> tuple[pd.DataFrame, list, list]:
    """파생피처 추가. rolling은 과거 창(트레일링)만 사용 → 누수 없음.

    반환: (df, 입력피처 목록, 파생피처 목록)
    """
    tb = df["RCS_6.AI.착수정_TB"]
    sed = df[C.TARGET_TB[basin]]
    d = {
        "TB_MA1h": tb.rolling(6, min_periods=1).mean(),
        "TB_MA3h": tb.rolling(18, min_periods=1).mean(),
        "TB_slope1h": tb - tb.shift(6),
        "PH_MA1h": df["RCS_6.AI.착수정_PH"].rolling(6, min_periods=1).mean(),
        "AL_MA1h": df["RCS_6.AI.착수정_AL"].rolling(6, min_periods=1).mean(),
        "EC_MA1h": df["RCS_6.AI.착수정_전기전도도"].rolling(6, min_periods=1).mean(),
        "온도_MA3h": df["RCS_6.AI.착수정_온도"].rolling(18, min_periods=1).mean(),
        "SED_MA1h": sed.rolling(6, min_periods=1).mean(),
        "SED_slope1h": sed - sed.shift(6),
    }
    for k, v in d.items():
        df[k] = v
    inputs = BASE_INPUTS + [C.TARGET_TB[basin]]
    derived = list(d.keys())
    return df, inputs, derived


def load_unified() -> pd.DataFrame:
    df = pd.read_parquet(PRE).sort_index()
    cols = BASE_INPUTS + [C.TARGET_TB[1], C.TARGET_TB[2]]
    have_clean = [c for c in cols if f"{c}_clean" in df.columns]
    raw_only = [c for c in cols if c not in have_clean]  # FT101 등 공정값은 원본 사용
    sub = df[[f"{c}_clean" for c in have_clean] + raw_only].resample("10min").median()
    sub.columns = have_clean + raw_only
    return sub[cols].reset_index()


def build_dataset(base: pd.DataFrame, basin: int, lead: int):
    df = base.copy()
    df, inputs, derived = add_derived(df, basin)
    df = C.shift_target_by_time(df, C.TARGET_TB[basin], lead)
    feats = inputs + derived
    df = df.dropna(subset=feats + ["y_future"]).sort_values(C.COL_DT).reset_index(drop=True)
    return df, feats, inputs, derived


def run_backtest(df, feats, lead, model_factory):
    cut = int(len(df) * INIT_FRAC)
    T0 = df[C.COL_DT].iloc[cut - 1]
    bt = df.iloc[cut:].reset_index(drop=True)
    week = np.floor((bt[C.COL_DT] - T0) / pd.Timedelta(days=7)).astype(int).to_numpy()
    cutoff_idx = {s: ((week // w) * w if w else np.zeros(len(bt), dtype=int)) for s, w in SCHEDULES.items()}
    needed = sorted(set(np.concatenate(list(cutoff_idx.values())).tolist()))

    preds = {}
    for j in needed:
        Tj = T0 + pd.Timedelta(days=7 * j)
        train = df[df[C.COL_DT] <= Tj - pd.Timedelta(minutes=lead)]  # 누수 방지
        assert len(train) == 0 or train[C.COL_DT].max() <= Tj - pd.Timedelta(minutes=lead)
        if len(train) < 100:
            continue
        m = model_factory()
        m.fit(train[feats], train["y_future"])
        p = np.full(len(bt), np.nan)
        mask = (bt[C.COL_DT] >= Tj).to_numpy()
        p[mask] = m.predict(bt.loc[mask, feats])
        preds[j] = p

    y_true = bt["y_future"].to_numpy()
    out = {}
    for s in SCHEDULES:
        idx = cutoff_idx[s]
        if any(j not in preds for j in set(idx.tolist())):
            continue
        y_pred = np.array([preds[j][i] for i, j in enumerate(idx)])
        ok = ~np.isnan(y_pred)
        out[s] = (C.compute_metrics(y_true[ok], y_pred[ok]), bt.loc[ok, C.COL_DT], y_true[ok], y_pred[ok])
    return out, len(bt)


def run_model_comparison():
    base = load_unified()
    print(f"통합 데이터: {len(base):,}행 ({base[C.COL_DT].min().date()} ~ {base[C.COL_DT].max().date()})\n")
    rows, best_pred = [], {}
    for basin in (1, 2):
        df, feats, inputs, derived = build_dataset(base, basin, LEAD)
        print(f"침전지{basin}: n={len(df):,}, 입력 {len(inputs)} + 파생 {len(derived)} = {len(feats)}피처")
        for mname in get_models():
            t = time.time()
            res, _ = run_backtest(df, feats, LEAD, lambda mn=mname: get_models()[mn])
            for sched, (m, ts, yt, yp) in res.items():
                rows.append({
                    "basin": basin, "lead": LEAD, "model": mname, "schedule": sched,
                    **{k: round(v, 4) for k, v in m.items()}, "passed": C.passes(m),
                })
                if sched == "1주":
                    key = basin
                    if key not in best_pred or m["r2"] > best_pred[key]["r2"]:
                        best_pred[key] = {"model": mname, "r2": m["r2"], "smape": m["smape"],
                                          "ts": ts, "y_true": yt, "y_pred": yp}
            print(f"  {mname}: {time.time()-t:.0f}초")
    res = pd.DataFrame(rows)
    C.save_csv(res, C.RESULTS_DIR / "task5_model_results.csv")
    plot_model_comparison(res)
    plot_best_predictions(best_pred)

    print("\n===== 6모델 비교 (재학습 1주) =====")
    one = res[res.schedule == "1주"].sort_values(["basin", "r2"], ascending=[True, False])
    print(one[["basin", "model", "r2", "smape", "rmse", "passed"]].to_string(index=False))
    print(f"\n통과: {int(res.passed.sum())}건")


def run_cluster_contrast():
    """통합 최고 모델을 동일 피처로 군집별에 적용해 대조."""
    res5 = pd.read_csv(C.RESULTS_DIR / "task5_model_results.csv")
    one = res5[res5.schedule == "1주"]
    rows = []
    for basin in (1, 2):
        best_model = one[one.basin == basin].sort_values("r2").iloc[-1]["model"]
        # 통합 결과 재사용
        b = one[(one.basin == basin) & (one.model == best_model)].iloc[0]
        rows.append({"setting": "통합", "basin": basin, "model": best_model,
                     "lead": LEAD, "r2": b.r2, "smape": b.smape})
        # 군집별
        for cid in range(3):
            df = C.load_cluster(cid, clean=True)
            df = df.set_index(C.COL_DT).sort_index().reset_index()
            df, feats, _, _ = (lambda d: build_dataset(d, basin, CLUSTER_LEADS[cid]))(df)
            if len(df) < 2000:
                continue
            t = time.time()
            res, _ = run_backtest(df, feats, CLUSTER_LEADS[cid], lambda: get_models()[best_model])
            if "1주" in res:
                m = res["1주"][0]
                rows.append({"setting": f"cluster{cid}", "basin": basin, "model": best_model,
                             "lead": CLUSTER_LEADS[cid], "r2": round(m["r2"], 4), "smape": round(m["smape"], 4)})
            print(f"침전지{basin} cluster{cid} {best_model}: {time.time()-t:.0f}초")
    out = pd.DataFrame(rows)
    C.save_csv(out, C.RESULTS_DIR / "task5_unified_vs_cluster.csv")
    plot_unified_vs_cluster(out)
    print("\n===== 통합 vs 군집 (재학습 1주, R²/SMAPE) =====")
    print(out.to_string(index=False))


def plot_model_comparison(res):
    import matplotlib.pyplot as plt

    one = res[res.schedule == "1주"]
    models = list(get_models().keys())
    x = np.arange(len(models))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, basin in enumerate((1, 2)):
        d = one[one.basin == basin].set_index("model").reindex(models)
        axes[0].bar(x + (i - 0.5) * w, d.r2, w, label=f"침전지{basin}",
                    color=["#2f5c9e", "#dd8452"][i])
        axes[1].bar(x + (i - 0.5) * w, d.smape, w, label=f"침전지{basin}",
                    color=["#2f5c9e", "#dd8452"][i])
    axes[0].axhline(0.9, color="red", ls=":", lw=1, label="기준 R²=0.9")
    axes[1].axhline(10, color="red", ls=":", lw=1, label="기준 SMAPE=10%")
    for ax, ylab, title in [(axes[0], "R²", "6모델 R² (재학습 1주)"), (axes[1], "SMAPE (%)", "6모델 SMAPE")]:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"피처 보강 후 6개 모델 후보군 비교 (통합 학습, lead {LEAD}분)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.PLOTS_DIR / "task5_model_comparison.png", dpi=120)
    plt.close(fig)


def plot_best_predictions(best_pred):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for i, basin in enumerate((1, 2)):
        b = best_pred[basin]
        ax = axes[i][0]
        ax.scatter(b["y_true"], b["y_pred"], s=3, alpha=0.3, color="#2f5c9e")
        lim = [min(b["y_true"].min(), b["y_pred"].min()), max(b["y_true"].max(), b["y_pred"].max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("실측 (NTU)")
        ax.set_ylabel("예측 (NTU)")
        ax.set_title(f"침전지{basin} {b['model']} — R²={b['r2']:.3f}, SMAPE={b['smape']:.1f}%")
        ax2 = axes[i][1]
        n = min(len(b["ts"]), 2000)
        ax2.plot(b["ts"].iloc[:n], b["y_true"][:n], lw=0.7, label="실측", color="#555555")
        ax2.plot(b["ts"].iloc[:n], b["y_pred"][:n], lw=0.7, label="예측", color="#dd8452")
        ax2.set_title(f"침전지{basin} 백테스트 구간 (앞 {n}점)")
        ax2.legend(fontsize=8)
    fig.suptitle("최고 모델 예측-실측 (통합 학습, 재학습 1주)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(C.PLOTS_DIR / "task5_best_predictions.png", dpi=120)
    plt.close(fig)


def plot_unified_vs_cluster(out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylab, crit in [(axes[0], "r2", "R²", 0.9), (axes[1], "smape", "SMAPE (%)", 10)]:
        for i, basin in enumerate((1, 2)):
            d = out[out.basin == basin]
            settings = d.setting.tolist()
            x = np.arange(len(settings))
            ax.bar(x + i * 0.38, d[metric], 0.38, label=f"침전지{basin}",
                   color=["#2f5c9e", "#dd8452"][i])
            ax.set_xticks(x + 0.19)
            ax.set_xticklabels(settings, fontsize=9)
        ax.axhline(crit, color="red", ls=":", lw=1)
        ax.set_ylabel(ylab)
        ax.set_title(f"통합 vs 군집별 {ylab}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("통합학습 vs 군집분류 학습 (동일 피처·모델, 재학습 1주)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(C.PLOTS_DIR / "task5_unified_vs_cluster.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    C.setup_output()
    if "cluster" in sys.argv[1:]:
        run_cluster_contrast()
    else:
        run_model_comparison()
