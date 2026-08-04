# -*- coding: utf-8 -*-
"""Task 2: 응집제 주입률(PAC SV) 예측 — 선형 계열 모델 후보군 선정.

스코프{cluster0,1,2, pooled} × 변환 4종 × 선형모델 6종 = 96 fit.
피처: 현재 시점 원수성상 5개. 타깃: PAC.AI.PAC_주입율제어_SV.
로그-선형 가설에 따라 log(y)~log(X) 등 변환 비교. 지표는 원 스케일에서 계산.
통과 기준: 테스트 R² >= 0.9 AND SMAPE <= 10%.
"""
import numpy as np
import pandas as pd

from . import common as C

ADD_INTERACTIONS = False  # 2라운드: log1p(TB)×온도 등 상호작용 항 추가 시 True

TRANSFORMS = ["logy_logX", "logy_X", "y_logX", "y_X"]


def get_models():
    from sklearn.linear_model import (
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        Ridge,
        TheilSenRegressor,
    )

    return {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10000),
        "Huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
        "TheilSen": TheilSenRegressor(random_state=C.RANDOM_STATE, max_subpopulation=5000),
    }


def make_estimator(model, log_y: bool):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), model)
    if log_y:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def build_features(df: pd.DataFrame, log_X: bool) -> pd.DataFrame:
    X = df[C.RAW_FEATURES].copy()
    if log_X:
        assert (X.min() > -1).all(), "log1p 적용 불가 컬럼 존재"
        X = np.log1p(X)
        X.columns = [f"log1p({c})" for c in C.RAW_FEATURES]
    if ADD_INTERACTIONS:
        tb, temp, al, ph = (df[C.RAW_FEATURES[i]] for i in (1, 3, 2, 0))
        X["log1p(TB)x온도"] = np.log1p(tb) * temp
        X["log1p(TB)xlog1p(AL)"] = np.log1p(tb) * np.log1p(al)
        X["PHxlog1p(TB)"] = ph * np.log1p(tb)
    return X


def load_scope(scope: str) -> pd.DataFrame:
    if scope == "pooled":
        df = C.load_pooled()
    else:
        df = C.load_cluster(int(scope[-1]))
    return df.dropna(subset=C.RAW_FEATURES + [C.COL_DOSE])


def extract_coefs(est, log_y: bool, feat_names):
    pipe = est.regressor_ if log_y else est
    lin = pipe[-1]
    if not hasattr(lin, "coef_"):
        return None
    return dict(zip(feat_names, np.ravel(lin.coef_))), float(np.ravel([lin.intercept_])[0])


def correlation_diagnostics(scopes):
    rows = []
    for scope in scopes:
        df = load_scope(scope)
        log_y = np.log(df[C.COL_DOSE])
        for c in C.RAW_FEATURES:
            rows.append({
                "scope": scope,
                "feature": c,
                "corr(log_SV, x)": round(np.corrcoef(df[c], log_y)[0, 1], 4),
                "corr(log_SV, log1p(x))": round(np.corrcoef(np.log1p(df[c]), log_y)[0, 1], 4),
                "n_unique_SV": df[C.COL_DOSE].nunique(),
            })
    return pd.DataFrame(rows)


def side_diagnostics():
    """PACS2−PAC 일치율, 주입_유량 분포 (저유량 SV 모드 관련) 진단."""
    rows = []
    for cid in range(3):
        df = C.load_cluster(cid)
        p1, p2 = df[C.COL_DOSE], df[C.COL_DOSE_S2]
        both = p1.notna() & p2.notna()
        diff = (p2 - p1)[both]
        row = {
            "cluster": cid,
            "PACS2_PAC_일치율(%)": round((diff == 0).mean() * 100, 2),
            "PACS2-PAC_평균": round(diff.mean(), 4),
            "PACS2-PAC_최소": round(diff.min(), 2),
            "PACS2-PAC_최대": round(diff.max(), 2),
        }
        for fc, label in [("PAC.AI.주입_유량1", "유량1"), ("PAC.AI.주입_유량2", "유량2")]:
            s = df[fc]
            row[f"{label}_중앙값"] = round(s.median(), 1)
            row[f"{label}_100미만(%)"] = round((s < 100).mean() * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_best(scope, transform, model_name, y_true, y_pred, metrics):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("실측 PAC 주입률 SV (ppm)")
    ax.set_ylabel("예측 PAC 주입률 SV (ppm)")
    ax.set_title(
        f"{scope} {transform} {model_name}\nR²={metrics['r2']:.3f}, SMAPE={metrics['smape']:.2f}%"
    )
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / f"task2_{scope}_best.png", dpi=120)
    plt.close(fig)


def main():
    C.setup_output()
    scopes = ["cluster0", "cluster1", "cluster2", "pooled"]
    rows, coef_rows = [], []
    best = {}
    for scope in scopes:
        df = load_scope(scope)
        train, test = C.time_ordered_split(df)
        for transform in TRANSFORMS:
            log_y = transform.startswith("logy")
            log_X = transform.endswith("logX")
            X_tr = build_features(train, log_X)
            X_te = build_features(test, log_X)
            y_tr, y_te = train[C.COL_DOSE], test[C.COL_DOSE]
            for name, model in get_models().items():
                est = make_estimator(model, log_y)
                est.fit(X_tr, y_tr)
                y_pred = est.predict(X_te)  # 원 스케일 (TransformedTargetRegressor가 역변환)
                m = C.compute_metrics(y_te, y_pred)
                rows.append({
                    "scope": scope, "transform": transform, "model": name,
                    "n_train": len(train), "n_test": len(test),
                    "n_unique_y": df[C.COL_DOSE].nunique(),
                    **{k: round(v, 4) for k, v in m.items()},
                    "passed": C.passes(m),
                })
                coefs = extract_coefs(est, log_y, list(X_tr.columns))
                if coefs is not None:
                    coef_rows.append({
                        "scope": scope, "transform": transform, "model": name,
                        "intercept": round(coefs[1], 4),
                        **{k: round(v, 4) for k, v in coefs[0].items()},
                    })
                key = scope
                if key not in best or m["r2"] > best[key]["metrics"]["r2"]:
                    best[key] = {
                        "transform": transform, "model": name, "metrics": m,
                        "y_true": y_te.to_numpy(), "y_pred": y_pred,
                    }
        print(f"{scope} 완료 (n={len(df)})")

    res = pd.DataFrame(rows)
    C.save_csv(res, C.RESULTS_DIR / "task2_all_results.csv")
    cand = res[res["passed"]].sort_values(["scope", "r2"], ascending=[True, False])
    C.save_csv(cand, C.RESULTS_DIR / "task2_candidates.csv")
    C.save_csv(pd.DataFrame(coef_rows), C.RESULTS_DIR / "task2_coefficients.csv")
    C.save_csv(correlation_diagnostics(scopes), C.RESULTS_DIR / "task2_correlations.csv")
    C.save_csv(side_diagnostics(), C.RESULTS_DIR / "task2_side_diagnostics.csv")

    for scope, b in best.items():
        plot_best(scope, b["transform"], b["model"], b["y_true"], b["y_pred"], b["metrics"])

    print("\n===== Task 2 요약: 기준 통과 (R²≥0.9 & SMAPE≤10%) =====")
    if cand.empty:
        print("통과한 조합이 없습니다.")
    else:
        print(cand[["scope", "transform", "model", "r2", "smape", "rmse"]].to_string(index=False))

    print("\n----- 스코프별 최고 성능 -----")
    for scope, b in best.items():
        m = b["metrics"]
        print(f"{scope}: {b['transform']} {b['model']} → R²={m['r2']:.3f}, SMAPE={m['smape']:.2f}%")

    per_cluster_pass = cand[cand["scope"] != "pooled"]
    if per_cluster_pass.empty:
        print(
            "\n[진단] 클러스터별로 기준(R²≥0.9)을 통과한 선형모델이 없습니다.\n"
            "원인: PAC 주입률 SV가 준이산 운영 설정값(클러스터당 고유값 5~22개)이고,\n"
            "클러스터 내부에서는 원수성상과의 상관이 약합니다(log(SV) 기준 최대 ~0.17).\n"
            "→ 클러스터 간 변동이 포함된 pooled 결과 및 task2_correlations.csv, "
            "task2_coefficients.csv를 함께 검토하세요."
        )
    print(f"\n전체 결과 {len(res)}행 → results/task2_all_results.csv, 통과 {len(cand)}행 → task2_candidates.csv")


if __name__ == "__main__":
    main()
