# -*- coding: utf-8 -*-
"""트리 계열 모델: XGBoost(GPU) / RandomForest."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from . import config as C


def make_xgb(early_stopping: bool = True) -> XGBRegressor:
    kw = dict(C.XGB)
    if early_stopping:
        kw["early_stopping_rounds"] = C.XGB_EARLY
    return XGBRegressor(**kw)


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(**C.RF)


def train_ml(name: str, Xtr: np.ndarray, Ytr: np.ndarray,
             Xval: np.ndarray = None, Yval: np.ndarray = None):
    """XGBoost는 val이 있으면 early stopping, RandomForest는 그대로 학습."""
    if name == "XGBoost":
        if Xval is not None and len(Xval):
            model = make_xgb(early_stopping=True)
            model.fit(Xtr, Ytr, eval_set=[(Xval, Yval)], verbose=False)
        else:
            model = make_xgb(early_stopping=False)
            model.fit(Xtr, Ytr, verbose=False)
    elif name == "RandomForest":
        model = make_rf()
        model.fit(Xtr, Ytr)
    else:
        raise ValueError(name)
    return model
