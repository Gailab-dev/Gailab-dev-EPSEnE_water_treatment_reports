# -*- coding: utf-8 -*-
"""평가 지표(원단위), 합격 판정, best 모델 선정."""
import numpy as np
import pandas as pd

from . import config as C


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """r2/smape/mape/rmse/mae — 덕남_응집제_투입량/src/common.py 이식."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    smape = 100.0 * np.mean(2.0 * np.abs(y_true - y_pred)
                            / (np.abs(y_true) + np.abs(y_pred)))
    mape = 100.0 * np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape),
        "mape": float(mape),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def metrics_two_targets(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """타깃(침전지1/2)별 지표 + 평균/최소 요약. 입력은 원단위 (N,2)."""
    m1 = compute_metrics(Y_true[:, 0], Y_pred[:, 0])
    m2 = compute_metrics(Y_true[:, 1], Y_pred[:, 1])
    return {
        "침전지1_r2": m1["r2"], "침전지1_smape": m1["smape"], "침전지1_mape": m1["mape"],
        "침전지1_rmse": m1["rmse"], "침전지1_mae": m1["mae"],
        "침전지2_r2": m2["r2"], "침전지2_smape": m2["smape"], "침전지2_mape": m2["mape"],
        "침전지2_rmse": m2["rmse"], "침전지2_mae": m2["mae"],
        "평균_smape": (m1["smape"] + m2["smape"]) / 2,
        "최소_r2": min(m1["r2"], m2["r2"]),
    }


def passes(m: dict) -> bool:
    """두 타깃 모두 SMAPE < 5% AND R² ≥ 0.9."""
    return (m["침전지1_smape"] < C.PASS["smape_max"]
            and m["침전지2_smape"] < C.PASS["smape_max"]
            and m["침전지1_r2"] >= C.PASS["r2_min"]
            and m["침전지2_r2"] >= C.PASS["r2_min"])


def n_pass_targets(m: dict) -> int:
    n = 0
    if m["침전지1_smape"] < C.PASS["smape_max"] and m["침전지1_r2"] >= C.PASS["r2_min"]:
        n += 1
    if m["침전지2_smape"] < C.PASS["smape_max"] and m["침전지2_r2"] >= C.PASS["r2_min"]:
        n += 1
    return n


def select_best(results: pd.DataFrame) -> pd.Series:
    """best 선정: 합격군 중 최소 피처수 → 평균 SMAPE.

    합격 없으면 (합격 타깃수 내림차순, 평균 SMAPE 오름차순) 최선 + 미달 플래그.
    """
    res = results.copy()
    res["합격"] = res.apply(lambda r: passes(r.to_dict()), axis=1)
    res["합격타깃수"] = res.apply(lambda r: n_pass_targets(r.to_dict()), axis=1)
    ok = res[res["합격"]]
    if len(ok):
        best = ok.sort_values(["피처수", "평균_smape"]).iloc[0]
    else:
        best = res.sort_values(["합격타깃수", "평균_smape"],
                               ascending=[False, True]).iloc[0]
    return best
