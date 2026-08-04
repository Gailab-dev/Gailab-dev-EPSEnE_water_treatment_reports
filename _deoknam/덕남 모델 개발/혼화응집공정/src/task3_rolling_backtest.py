# -*- coding: utf-8 -*-
"""Task 3: 확장 윈도우 롤링 재학습 백테스트.

재학습 주기 {1주, 2주, 4주, 12주, 고정} 별로 6그룹(클러스터×침전지)의
실전 예측 성능을 측정한다. 모든 주기가 7일 배수이므로 주 단위 컷오프별로
모델을 1회만 학습하고 스케줄 간 예측을 공유한다.

누수 방지: 컷오프 T_j에서 학습 가능한 행은 타깃이 이미 관측된 행뿐
(t + lead <= T_j). 예측은 t > T_j 인 행만.

실행: python -m src.task3_rolling_backtest [cid ...]
"""
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from . import common as C

SCHEDULES = {"1주": 1, "2주": 2, "4주": 4, "12주": 12, "고정": None}
MIN_EVAL_N = 50
INIT_FRAC = 0.5

# 자기회귀 피처: t 시점의 해당 침전지 탁도(직전 관측값)를 피처에 추가.
# 실행 인자 'ar'로 활성화: python -m src.task3_rolling_backtest ar [cid ...]
INCLUDE_AR = False

# 슬라이딩 윈도우 재학습: 실행 인자 'win'으로 활성화 → 확장/180일/90일 3종 비교.
# 윈도우 내 데이터가 MIN_TRAIN 미만이면 컷오프 이전 최근 MIN_TRAIN행으로 보정.
INCLUDE_SLIDING = False
WINDOWS_SLIDING = {"확장": None, "180일": pd.Timedelta(days=180), "90일": pd.Timedelta(days=90)}
MIN_TRAIN = 500

# 전처리 정제 클러스터 사용: 실행 인자 'clean'으로 활성화.
USE_CLEAN = False

RS = C.RANDOM_STATE


def _et(n):
    return lambda: ExtraTreesRegressor(n_estimators=n, n_jobs=-1, random_state=RS)


BEST_CONFIG = {
    (0, 1): {"lead": 180, "flow": False, "name": "ExtraTrees300", "factory": _et(300)},
    (0, 2): {"lead": 180, "flow": False, "name": "ExtraTrees300", "factory": _et(300)},
    (1, 1): {"lead": 150, "flow": True, "name": "ExtraTrees600", "factory": _et(600)},
    (1, 2): {
        "lead": 210, "flow": True, "name": "XGBoost1000d8",
        "factory": lambda: XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=RS,
        ),
    },
    (2, 1): {"lead": 260, "flow": True, "name": "ExtraTrees600", "factory": _et(600)},
    (2, 2): {
        "lead": 240, "flow": False, "name": "LightGBM500",
        "factory": lambda: LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RS, verbose=-1,
        ),
    },
}

# 무작위 분할 상한 R² (task1_candidates_random.csv / task1_round2_random.csv)
REF_RANDOM = {(0, 1): 0.947, (0, 2): 0.933, (1, 1): 0.873, (1, 2): 0.906, (2, 1): 0.878, (2, 2): 0.923}


def build_dataset(cid: int, basin: int, lead: int, include_flow: bool):
    df = C.load_cluster(cid, clean=USE_CLEAN)
    df = C.shift_target_by_time(df, C.TARGET_TB[basin], lead)
    features = C.RAW_FEATURES + [C.COL_DOSE] + ([C.COL_FLOW] if include_flow else [])
    if INCLUDE_AR:
        features = features + [C.TARGET_TB[basin]]
    df = df.dropna(subset=features + ["y_future"]).sort_values(C.COL_DT).reset_index(drop=True)
    assert df[C.COL_DT].is_unique
    return df, features


def safe_metrics(y_true, y_pred) -> dict:
    if len(y_true) < MIN_EVAL_N:
        return {k: np.nan for k in ("r2", "smape", "mape", "rmse", "mae")}
    m = C.compute_metrics(y_true, y_pred)
    if np.var(np.asarray(y_true, dtype=float)) < 1e-12:
        m["r2"] = np.nan
    return m


def run_group(cid: int, basin: int, cfg: dict, window_name: str = "확장", window=None):
    t_start = time.time()
    df, features = build_dataset(cid, basin, cfg["lead"], cfg["flow"])
    cut = int(len(df) * INIT_FRAC)
    T0 = df[C.COL_DT].iloc[cut - 1]
    bt = df.iloc[cut:].reset_index(drop=True)
    week = np.floor((bt[C.COL_DT] - T0) / pd.Timedelta(days=7)).astype(int).to_numpy()

    cutoff_idx = {}
    for sched, w in SCHEDULES.items():
        cutoff_idx[sched] = (week // w) * w if w else np.zeros(len(bt), dtype=int)
    needed = sorted(set(np.concatenate(list(cutoff_idx.values())).tolist()))

    preds = {}
    for j in needed:
        Tj = T0 + pd.Timedelta(days=7 * j)
        train_cut = Tj - pd.Timedelta(minutes=cfg["lead"])
        train = df[df[C.COL_DT] <= train_cut]
        assert len(train) and train[C.COL_DT].max() <= train_cut  # 누수 방지
        if window is not None:
            recent = train[train[C.COL_DT] > train_cut - window]
            # 희소 구간 안전장치: 윈도우 내 데이터 부족 시 최근 MIN_TRAIN행 사용
            train = recent if len(recent) >= MIN_TRAIN else train.tail(MIN_TRAIN)
        model = cfg["factory"]()
        model.fit(train[features], train["y_future"])
        p = np.full(len(bt), np.nan)
        # t == Tj 경계 행 포함(>=): 타깃은 t+lead 미래라 누수 아님
        mask = (bt[C.COL_DT] >= Tj).to_numpy()
        p[mask] = model.predict(bt.loc[mask, features])
        preds[j] = p

    y_true = bt["y_future"].to_numpy()
    months = bt[C.COL_DT].dt.strftime("%Y-%m").to_numpy()
    summary_rows, timeline_rows, sched_preds = [], [], {}
    for sched in SCHEDULES:
        y_pred = np.array([preds[j][i] for i, j in enumerate(cutoff_idx[sched])])
        assert not np.isnan(y_pred).any(), "모든 백테스트 행이 예측 커버되어야 함"
        sched_preds[sched] = y_pred
        m = C.compute_metrics(y_true, y_pred)
        summary_rows.append({
            "cluster": cid, "basin": basin, "model": cfg["name"], "lead": cfg["lead"],
            "include_flow": cfg["flow"], "window": window_name, "schedule": sched,
            "n_fits": len(needed), "n_pred": len(bt),
            **{k: round(v, 4) for k, v in m.items()},
            "passed": C.passes(m), "r2_random_ref": REF_RANDOM[(cid, basin)],
        })
        for mon in np.unique(months):
            sel = months == mon
            mm = safe_metrics(y_true[sel], y_pred[sel])
            timeline_rows.append({
                "cluster": cid, "basin": basin, "window": window_name, "schedule": sched,
                "month": mon, "n": int(sel.sum()),
                **{k: round(v, 4) if np.isfinite(v) else np.nan for k, v in mm.items()},
            })

    r2_fixed = next(r["r2"] for r in summary_rows if r["schedule"] == "고정")
    ref = REF_RANDOM[(cid, basin)]
    for r in summary_rows:
        denom = ref - r2_fixed
        r["gap_closure_r2"] = round((r["r2"] - r2_fixed) / denom, 4) if abs(denom) > 1e-9 else np.nan
        if r["schedule"] != "고정" and r["r2"] < r2_fixed - 0.05:
            print(f"[경고] c{cid} b{basin} {window_name} {r['schedule']}: 고정보다 R² {r2_fixed - r['r2']:.3f} 낮음")

    print(
        f"cluster{cid} 침전지{basin} [{window_name}] 완료: fit {len(needed)}회, 백테스트 {len(bt)}행 "
        f"({bt[C.COL_DT].min().date()} ~ {bt[C.COL_DT].max().date()}), {time.time() - t_start:.0f}초"
    )
    return summary_rows, timeline_rows, {"bt": bt, "y_true": y_true, "sched_preds": sched_preds}


def plot_group(cid, basin, summary_df, timeline_df):
    import matplotlib.pyplot as plt

    g = summary_df[(summary_df["cluster"] == cid) & (summary_df["basin"] == basin)]
    order = ["고정", "12주", "4주", "2주", "1주"]
    g = g.set_index("schedule").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(order, g["r2"], color="#4878b0")
    axes[0].axhline(REF_RANDOM[(cid, basin)], color="green", ls="--", lw=1, label="무작위 분할 상한")
    axes[0].axhline(0.9, color="red", ls=":", lw=1, label="기준 R²=0.9")
    for bar, (_, row) in zip(bars, g.iterrows()):
        axes[0].annotate(
            f"SMAPE\n{row['smape']:.1f}%", (bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0)),
            ha="center", va="bottom", fontsize=8,
        )
    axes[0].set_ylabel("백테스트 R²")
    ar_tag = " (+직전탁도)" if INCLUDE_AR else ""
    axes[0].set_title(f"cluster{cid} 침전지{basin}: 재학습 주기별 성능{ar_tag}")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(min(0, g["r2"].min() - 0.1), 1.05)

    t = timeline_df[(timeline_df["cluster"] == cid) & (timeline_df["basin"] == basin)]
    for sched, style in [("고정", "r--"), ("4주", "y-"), ("1주", "g-")]:
        s = t[t["schedule"] == sched].dropna(subset=["r2"]).sort_values("month")
        axes[1].plot(s["month"], s["r2"], style, marker="o", ms=3, lw=1, label=sched)
    axes[1].axhline(0.9, color="red", ls=":", lw=0.8)
    axes[1].set_title("월별 R² 추이")
    axes[1].set_ylabel("월별 R²")
    axes[1].tick_params(axis="x", rotation=60, labelsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    ar_sfx = "_ar" if INCLUDE_AR else ""
    fig.savefig(C.PLOTS_DIR / f"task3_c{cid}_basin{basin}{ar_sfx}.png", dpi=120)
    plt.close(fig)


def plot_group_windows(cid, basin, summary_df):
    import matplotlib.pyplot as plt

    order = ["고정", "12주", "4주", "2주", "1주"]
    g = summary_df[(summary_df["cluster"] == cid) & (summary_df["basin"] == basin)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for wname, style in [("확장", "o-"), ("180일", "s-"), ("90일", "^-")]:
        gw = g[g["window"] == wname].set_index("schedule").reindex(order)
        axes[0].plot(order, gw["r2"], style, label=wname)
        axes[1].plot(order, gw["smape"], style, label=wname)
    axes[0].axhline(0.9, color="red", ls=":", lw=0.8, label="기준 R²=0.9")
    axes[0].set_ylabel("백테스트 R²")
    axes[0].set_title(f"cluster{cid} 침전지{basin}: 학습 윈도우별 성능 (+직전탁도)")
    axes[1].axhline(10, color="red", ls=":", lw=0.8, label="기준 SMAPE=10%")
    axes[1].set_ylabel("백테스트 SMAPE (%)")
    axes[1].set_title("SMAPE")
    for ax in axes:
        ax.set_xlabel("재학습 주기")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / f"task3_c{cid}_basin{basin}_sliding.png", dpi=120)
    plt.close(fig)


def main():
    C.setup_output()
    args = sys.argv[1:]
    cids = [int(a) for a in args if a.isdigit()] or [0, 1, 2]
    windows = WINDOWS_SLIDING if INCLUDE_SLIDING else {"확장": None}
    all_summary, all_timeline = [], []
    for cid in cids:
        for basin in (1, 2):
            for wname, wdelta in windows.items():
                s, t, _ = run_group(cid, basin, BEST_CONFIG[(cid, basin)], wname, wdelta)
                all_summary += s
                all_timeline += t

    summary = pd.DataFrame(all_summary)
    timeline = pd.DataFrame(all_timeline)
    suffix = (
        ("_ar" if INCLUDE_AR else "")
        + ("_sliding" if INCLUDE_SLIDING else "")
        + ("_clean" if USE_CLEAN else "")
        + ("" if cids == [0, 1, 2] else "_partial")
    )
    C.save_csv(summary, C.RESULTS_DIR / f"task3_backtest_results{suffix}.csv")
    C.save_csv(timeline, C.RESULTS_DIR / f"task3_backtest_timeline{suffix}.csv")
    for cid in cids:
        for basin in (1, 2):
            if INCLUDE_SLIDING:
                plot_group_windows(cid, basin, summary)
            else:
                plot_group(cid, basin, summary, timeline)

    print("\n===== Task 3 요약: 재학습 주기별 백테스트 성능 =====")
    order = ["고정", "12주", "4주", "2주", "1주"]
    for (cid, basin), g in summary.groupby(["cluster", "basin"]):
        print(f"\n--- cluster{cid} 침전지{basin} (무작위 상한 R²={REF_RANDOM[(cid, basin)]}) ---")
        for wname in windows:
            gw = g[g["window"] == wname].set_index("schedule").loc[order]
            tag = f" [윈도우 {wname}]" if len(windows) > 1 else ""
            print(f"{tag}")
            print(gw[["r2", "smape", "passed", "gap_closure_r2"]].to_string())
    print("\n--- (윈도우, 주기)별 기준 통과 그룹 수 ---")
    print(summary.groupby(["window", "schedule"])["passed"].sum().astype(int).to_string())


if __name__ == "__main__":
    if "ar" in sys.argv[1:]:
        INCLUDE_AR = True
    if "win" in sys.argv[1:]:
        INCLUDE_SLIDING = True
    if "clean" in sys.argv[1:]:
        USE_CLEAN = True
    main()
