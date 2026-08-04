# -*- coding: utf-8 -*-
"""최종 학습(train+val 재학습) 및 test 1회 평가 — 공정 비교(index 교집합)."""
import numpy as np
import pandas as pd

from . import config as C
from . import datasets, evaluate, models_dl, models_ml, selection


def final_eval(cid: int, df_tr, df_val, df_te, ranked: list, sel_df: pd.DataFrame,
               model_names: list):
    """모델별 선정 k로 train+val 재학습 → test 예측.

    반환: (results_df, artifacts dict)
    artifacts[name] = {model, feat_cols, scaler_x, scaler_y, k,
                       idx, Y_pred(원단위), Y_true(원단위)}
    """
    df_trval = pd.concat([df_tr, df_val])
    artifacts = {}
    for name in model_names:
        k = selection.pick_k(sel_df, name)
        feat = ranked[:k]
        scaler_x, scaler_y = datasets.fit_scalers(df_trval, feat)
        if name in ("XGBoost", "RandomForest"):
            Xtr, Ytr, _ = datasets.to_tabular(df_trval, feat, scaler_x, scaler_y)
            model = models_ml.train_ml(name, Xtr, Ytr)  # 재학습은 조기종료 없이 전체
            Xte, Yte, idx = datasets.to_tabular(df_te, feat, scaler_x, scaler_y)
            pred_s = model.predict(Xte)
        else:
            Xtr, Ytr, _ = datasets.to_sequences(df_trval, feat, scaler_x, scaler_y)
            cut = int(len(Xtr) * (1 - C.SEQ["inner_val_frac"]))
            model = models_dl.train_dl(name, Xtr[:cut], Ytr[:cut], Xtr[cut:], Ytr[cut:])
            Xte, Yte, idx = datasets.to_sequences(df_te, feat, scaler_x, scaler_y)
            pred_s = models_dl.predict_dl(model, Xte)
        diff_pred = scaler_y.inverse_transform(np.asarray(pred_s))
        artifacts[name] = {
            "model": model, "feat_cols": feat, "k": k,
            "scaler_x": scaler_x, "scaler_y": scaler_y, "idx": idx,
            "Y_pred": datasets.reconstruct_abs(df_te, idx, diff_pred),
            "Y_true": datasets.abs_true(df_te, idx),
        }
        print(f"  [최종] {name} k={k}: test {len(idx):,}행 예측 완료")

    # 공정 비교: 전 모델 유효 test index 교집합
    common = None
    for a in artifacts.values():
        s = pd.DatetimeIndex(a["idx"])
        common = s if common is None else common.intersection(s)
    print(f"  공통 평가행: {len(common):,}")

    # 참고 기준선: persistence (변화 없음 가정, 예측 = 현재 평활값)
    base_pred = df_te.loc[common, C.BASE_COLS].to_numpy(float)
    base_true = datasets.abs_true(df_te, common)
    mb = evaluate.metrics_two_targets(base_true, base_pred)
    print(f"  [참고] persistence 기준선: 침전지1 R²={mb['침전지1_r2']:.3f}/"
          f"SMAPE={mb['침전지1_smape']:.2f}%, 침전지2 R²={mb['침전지2_r2']:.3f}/"
          f"SMAPE={mb['침전지2_smape']:.2f}%")

    rows = []
    for name, a in artifacts.items():
        pos = pd.DatetimeIndex(a["idx"]).get_indexer(common)
        m = evaluate.metrics_two_targets(a["Y_true"][pos], a["Y_pred"][pos])
        a["common_true"] = a["Y_true"][pos]
        a["common_pred"] = a["Y_pred"][pos]
        rows.append({"cluster": cid, "모델": name, "피처수": a["k"], **m})
    results = pd.DataFrame(rows)
    return results, artifacts, common
