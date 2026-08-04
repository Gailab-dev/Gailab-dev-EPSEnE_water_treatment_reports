# -*- coding: utf-8 -*-
"""피처 선택: XGB gain 랭킹 → top-k 중첩 부분집합 val 평가 → 모델별 최소 k."""
import numpy as np
import pandas as pd

from . import config as C
from . import datasets, evaluate, models_dl, models_ml


def rank_features(df_tr: pd.DataFrame, df_val: pd.DataFrame) -> list:
    """후보 전체로 XGB 학습 → gain 중요도(2출력 평균) 내림차순 랭킹."""
    scaler_x, scaler_y = datasets.fit_scalers(df_tr, C.CAND_FEATURES)
    Xtr, Ytr, _ = datasets.to_tabular(df_tr, C.CAND_FEATURES, scaler_x, scaler_y)
    Xv, Yv, _ = datasets.to_tabular(df_val, C.CAND_FEATURES, scaler_x, scaler_y)
    model = models_ml.train_ml("XGBoost", Xtr, Ytr, Xv, Yv)
    imp = model.feature_importances_  # 멀티아웃풋 평균 gain
    order = np.argsort(imp)[::-1]
    ranked = [C.CAND_FEATURES[i] for i in order]
    return ranked


def _eval_val(name: str, feat_cols: list, df_tr, df_val) -> dict:
    """(모델, 피처셋)을 train으로 학습해 val 원단위 지표 반환."""
    scaler_x, scaler_y = datasets.fit_scalers(df_tr, feat_cols)
    if name in ("XGBoost", "RandomForest"):
        Xtr, Ytr, _ = datasets.to_tabular(df_tr, feat_cols, scaler_x, scaler_y)
        Xv, Yv, idx = datasets.to_tabular(df_val, feat_cols, scaler_x, scaler_y)
        model = models_ml.train_ml(name, Xtr, Ytr,
                                   Xv if name == "XGBoost" else None,
                                   Yv if name == "XGBoost" else None)
        pred_s = model.predict(Xv)
    else:
        Xtr, Ytr, _ = datasets.to_sequences(df_tr, feat_cols, scaler_x, scaler_y)
        Xv, Yv, idx = datasets.to_sequences(df_val, feat_cols, scaler_x, scaler_y)
        if len(Xtr) < 100 or len(Xv) < 50:
            return None
        # DL 조기종료용 내부 val = train 마지막 10%
        cut = int(len(Xtr) * (1 - C.SEQ["inner_val_frac"]))
        model = models_dl.train_dl(name, Xtr[:cut], Ytr[:cut], Xtr[cut:], Ytr[cut:])
        pred_s = models_dl.predict_dl(model, Xv)
    diff_pred = scaler_y.inverse_transform(np.asarray(pred_s))
    Y_pred = datasets.reconstruct_abs(df_val, idx, diff_pred)
    Y_true = datasets.abs_true(df_val, idx)
    return evaluate.metrics_two_targets(Y_true, Y_pred)


def run_selection(cid: int, df_tr, df_val, ranked: list,
                  model_names: list) -> pd.DataFrame:
    """k 그리드 × 모델 val 평가 결과 테이블."""
    rows = []
    for k in C.K_GRID:
        feat_cols = ranked[:k]
        for name in model_names:
            m = _eval_val(name, feat_cols, df_tr, df_val)
            if m is None:
                continue
            rows.append({"cluster": cid, "모델": name, "피처수": k, **m})
            print(f"  [선택] {name} k={k}: val 평균SMAPE {m['평균_smape']:.2f}%, "
                  f"최소R² {m['최소_r2']:.3f}")
    return pd.DataFrame(rows)


def pick_k(sel_df: pd.DataFrame, name: str) -> int:
    """val 최고 성능 대비 SMAPE +K_TOL_SMAPE, R² -K_TOL_R2 이내인 최소 k."""
    sub = sel_df[sel_df["모델"] == name].sort_values("피처수")
    if not len(sub):
        return max(C.K_GRID)
    best_smape = sub["평균_smape"].min()
    best_r2 = sub["최소_r2"].max()
    ok = sub[(sub["평균_smape"] <= best_smape + C.K_TOL_SMAPE)
             & (sub["최소_r2"] >= best_r2 - C.K_TOL_R2)]
    return int(ok.iloc[0]["피처수"]) if len(ok) else int(sub.iloc[0]["피처수"])
