# -*- coding: utf-8 -*-
"""모사 분류기 학습·평가: XGBoost 5클래스 (AR 포함/미포함 2종) + persistence 대비."""
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import config as C
from . import data


def _xy(day: pd.DataFrame, line: int, use_ar: bool):
    feats = list(data.FEATURES) + ([f"SV{line}_어제"] if use_ar else [])
    X = day[feats].to_numpy(float)
    y = day[f"SV{line}"].astype(int).to_numpy()
    y_idx = np.searchsorted(C.CLASSES, y)
    return X, y_idx, feats


def train(day_tr: pd.DataFrame, line: int, use_ar: bool) -> dict:
    X, y, feats = _xy(day_tr, line, use_ar)
    present = np.unique(y)                      # 학습창에 존재하는 클래스만 압축 매핑
    y_compact = np.searchsorted(present, y)
    clf = XGBClassifier(objective="multi:softprob", **C.XGB)
    clf.fit(X, y_compact, verbose=False)
    return {"clf": clf, "feats": feats, "use_ar": use_ar, "line": line,
            "classes": np.array(C.CLASSES)[present]}


def predict(bundle: dict, day: pd.DataFrame) -> np.ndarray:
    X = day[bundle["feats"]].to_numpy(float)
    return bundle["classes"][bundle["clf"].predict(X)]


def evaluate(bundle: dict, day_te: pd.DataFrame) -> dict:
    line = bundle["line"]
    y_true = day_te[f"SV{line}"].astype(int).to_numpy()
    y_prev = day_te[f"SV{line}_어제"].astype(int).to_numpy()
    y_pred = predict(bundle, day_te)

    changed = y_true != y_prev                      # 실제 변경일
    pred_change = y_pred != y_prev                  # 모델이 변경이라 한 날
    res = {
        "정확도(%)": round((y_pred == y_true).mean() * 100, 1),
        "persistence(%)": round((y_prev == y_true).mean() * 100, 1),
        "MAE(ppm)": round(np.abs(y_pred - y_true).mean(), 3),
        "변경일수": int(changed.sum()),
        "변경감지 재현율(%)": round((pred_change & changed).sum() / max(changed.sum(), 1) * 100, 1),
        "오경보(변경 아님에 변경 예측)": int((pred_change & ~changed).sum()),
    }
    return res


def change_event_analysis(day: pd.DataFrame, line: int) -> pd.DataFrame:
    """전 기간 변경 이벤트의 당일 조건 정리 (규칙화 근거)."""
    sv = day[f"SV{line}"]
    chg = day[sv.diff().fillna(0) != 0].copy()
    out = chg[["TB", "TB_7일최대", "수온", "AL", "정수량", "침전지1_전일", "침전지2_전일"]].copy()
    out.insert(0, "변경", [f"{int(a)}→{int(b)}" for a, b in
                          zip(sv.shift(1).loc[chg.index], sv.loc[chg.index])])
    out.insert(1, "방향", np.where(sv.diff().loc[chg.index] > 0, "증량", "감량"))
    return out.round(2)


def save(bundles: dict, path=None) -> None:
    path = C.MODELS_DIR / "dose_mimic_xgb.pkl" if path is None else path
    joblib.dump(bundles, path)
    print(f"모사 모델 저장: {path}")
