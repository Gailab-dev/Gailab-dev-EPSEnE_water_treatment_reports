# -*- coding: utf-8 -*-
"""추천용 탁도 모델: 주입률 강제 포함 + 단조 제약 XGBoost (타깃별 단일출력).

타깃은 02_coag와 동일한 델타(평활탁도(t+lag) − 평활탁도(t)) 원단위.
트리 모델이므로 스케일링 없이 학습한다. 용연은 단일 침전지 타깃(N=1).
"""
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import config as R
from coag import config as CC
from coag import datasets, dtw_lag, evaluate, features, io_utils

N_TARGETS = len(R.TARGET_COLS)


def build_dataset(cid: int):
    """02_coag와 동일한 파이프라인으로 군집 학습 테이블 생성."""
    df10 = features.add_segment_id(features.resample_10min(io_utils.load_cluster(cid)))
    df10 = features.add_dose_feature(df10)
    df10, _ = features.mask_idle_targets(df10)
    df10 = features.add_trend_features(df10)
    df10 = features.add_smoothed_targets(df10)
    lags, _ = dtw_lag.estimate_cluster_lags(df10, cid)
    dyn = CC.LAG_MODE.get(cid) == "dynamic"
    table = features.build_table(features.align_targets(df10, lags, dynamic=dyn))
    return table, lags


def train_dose_models(table: pd.DataFrame):
    """train+val로 타깃별 단조제약 XGB 학습. 반환 {i: model} (i=1..N)."""
    df_tr, df_val, _ = datasets.chrono_split(table)
    df_trval = pd.concat([df_tr, df_val])
    X = df_trval[R.DOSE_FEATURES].to_numpy(float)
    models = {}
    for i in range(1, N_TARGETS + 1):
        y = df_trval[f"y{i}_diff"].to_numpy(float)
        m = XGBRegressor(monotone_constraints=R.MONOTONE, **R.XGB_DOSE)
        m.fit(X, y, verbose=False)
        models[i] = m
    return models


def predict_abs(models: dict, df_rows: pd.DataFrame) -> np.ndarray:
    """행별 절대 탁도 예측 (N, n_targets): 현재 평활값 + 예측 델타."""
    X = df_rows[R.DOSE_FEATURES].to_numpy(float)
    base = df_rows[CC.BASE_COLS].to_numpy(float)
    out = np.empty((len(df_rows), N_TARGETS))
    for i in range(1, N_TARGETS + 1):
        out[:, i - 1] = base[:, i - 1] + models[i].predict(X)
    return out


def eval_models(models: dict, table: pd.DataFrame) -> dict:
    """test 구간 성능 (원단위 절대 탁도 기준)."""
    _, _, df_te = datasets.chrono_split(table)
    pred = predict_abs(models, df_te)
    true = datasets.abs_true(df_te, df_te.index)
    return evaluate.metrics_targets(true, pred)


def save_models(cid: int, models: dict, lags: dict, metrics: dict) -> None:
    d = R.MODELS_DIR / f"cluster{cid}"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "feature_cols": R.DOSE_FEATURES,
                 "monotone": R.MONOTONE, "lags": lags,
                 "test_metrics": metrics}, d / "dose_xgb_monotone.pkl")
    print(f"추천용 모델 저장: {d}\\dose_xgb_monotone.pkl")


def load_models(cid: int) -> dict:
    return joblib.load(R.MODELS_DIR / f"cluster{cid}" / "dose_xgb_monotone.pkl")
