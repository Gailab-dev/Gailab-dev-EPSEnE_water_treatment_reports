# -*- coding: utf-8 -*-
"""Task 1: 침전지 탁도 예측 모델 후보군 선정.

클러스터{0,1,2} × 수조{침전지1,2} × 후보시간 4개 × 모델 10종 = 240 fit.
피처: 현재 시점 원수성상 5개 + PAC 주입율 SV. 타깃: t+lead 침전지 탁도.
통과 기준: 테스트 R² >= 0.9 AND SMAPE <= 10%.
"""
import sys

import numpy as np
import pandas as pd

from . import common as C

INCLUDE_FLOW = False  # 2라운드: FT101 피처 추가 시 True

# 분할 모드: "time"(시간순, 기본) 또는 "random". 실행 인자로 지정: python -m src.task1_... random
SPLIT_MODE = "time"

# 2라운드 튜닝 여지: Ridge/Lasso alpha {1e-4..10}, 트리 n_estimators {300,600,1000}, max_depth


def get_models():
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from xgboost import XGBRegressor

    rs = C.RANDOM_STATE
    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=1e-3, max_iter=10000)),
        "ElasticNet": make_pipeline(
            StandardScaler(), ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10000)
        ),
        "Poly2_Ridge": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=rs
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=rs,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=rs, verbose=-1,
        ),
    }


def build_dataset(cid: int, basin: int, lead_used: int):
    df = C.load_cluster(cid)
    df = C.shift_target_by_time(df, C.TARGET_TB[basin], lead_used)
    features = C.RAW_FEATURES + [C.COL_DOSE]
    if INCLUDE_FLOW:
        features = features + [C.COL_FLOW]
    match_rate = df["y_future"].notna().mean()
    df = df.dropna(subset=features + ["y_future"])
    return df, features, match_rate


def plot_best(cid, basin, lead_used, model_name, test_df, y_pred, metrics):
    import matplotlib.pyplot as plt

    y_true = test_df["y_future"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("실측 침전지 탁도 (NTU)")
    axes[0].set_ylabel("예측 침전지 탁도 (NTU)")
    axes[0].set_title(
        f"cluster{cid} 침전지{basin} +{lead_used}분 {model_name}\n"
        f"R²={metrics['r2']:.3f}, SMAPE={metrics['smape']:.2f}%"
    )
    n = min(len(test_df), 1000)
    axes[1].plot(test_df[C.COL_DT].iloc[:n], y_true[:n], lw=0.8, label="실측")
    axes[1].plot(test_df[C.COL_DT].iloc[:n], y_pred[:n], lw=0.8, label="예측")
    axes[1].set_title("테스트 구간 시계열 (앞 1000점)")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = C.PLOTS_DIR / f"task1_c{cid}_basin{basin}_best.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)


def split_dataset(df):
    if SPLIT_MODE == "random":
        from sklearn.model_selection import train_test_split

        return train_test_split(df, test_size=0.2, random_state=C.RANDOM_STATE)
    return C.time_ordered_split(df)


def main():
    C.setup_output()
    suffix = "" if SPLIT_MODE == "time" else f"_{SPLIT_MODE}"
    rows = []
    best = {}  # (cid, basin) -> dict for plotting
    for cid in range(3):
        for basin in (1, 2):
            for lead_req, lead_used in C.LEAD_CANDIDATES[cid]:
                df, features, match_rate = build_dataset(cid, basin, lead_used)
                train, test = split_dataset(df)
                X_tr, y_tr = train[features], train["y_future"]
                X_te, y_te = test[features], test["y_future"]
                for name, model in get_models().items():
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    m = C.compute_metrics(y_te, y_pred)
                    rows.append({
                        "cluster": cid, "basin": basin,
                        "lead_requested": lead_req, "lead_used": lead_used,
                        "lead_note": "" if lead_req == lead_used else f"{lead_req}분→{lead_used}분 반올림",
                        "model": name, "n_train": len(train), "n_test": len(test),
                        "match_rate": round(match_rate, 4), **{k: round(v, 4) for k, v in m.items()},
                        "passed": C.passes(m),
                    })
                    key = (cid, basin)
                    if key not in best or m["r2"] > best[key]["metrics"]["r2"]:
                        best[key] = {
                            "lead_used": lead_used, "model": name, "metrics": m,
                            "test": test, "y_pred": y_pred,
                        }
                print(f"cluster{cid} 침전지{basin} +{lead_used}분 완료 (n={len(df)}, 매칭률 {match_rate:.1%})")

    res = pd.DataFrame(rows)
    res.insert(0, "split", SPLIT_MODE)
    C.save_csv(res, C.RESULTS_DIR / f"task1_all_results{suffix}.csv")
    cand = res[res["passed"]].sort_values(["cluster", "basin", "r2"], ascending=[True, True, False])
    C.save_csv(cand, C.RESULTS_DIR / f"task1_candidates{suffix}.csv")

    if SPLIT_MODE == "time":
        for (cid, basin), b in best.items():
            plot_best(cid, basin, b["lead_used"], b["model"], b["test"], b["y_pred"], b["metrics"])

    print(f"\n===== Task 1 요약 (split={SPLIT_MODE}): 기준 통과 (R²≥0.9 & SMAPE≤10%) =====")
    if cand.empty:
        print("통과한 조합이 없습니다.")
    else:
        summary = cand.groupby(["cluster", "basin"]).size().rename("통과 조합 수")
        print(summary.to_string())
        cols = ["cluster", "basin", "lead_requested", "lead_used", "model", "r2", "smape", "rmse"]
        print("\n" + cand[cols].to_string(index=False))
    print(
        f"\n전체 결과 {len(res)}행 → results/task1_all_results{suffix}.csv, "
        f"통과 {len(cand)}행 → task1_candidates{suffix}.csv"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "random":
        SPLIT_MODE = "random"
    main()
