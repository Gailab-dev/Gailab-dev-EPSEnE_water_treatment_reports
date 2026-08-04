# -*- coding: utf-8 -*-
"""KMeans 군집 모델: 학습, 탁도 기준 재라벨, 예측, 중심표, 저장/온라인 헬퍼."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import feature_columns, window_features_single


def fit_kmeans(feats: pd.DataFrame, k: int = C.K,
               random_state: int = C.RANDOM_STATE) -> Pipeline:
    """NaN 없는 유효행 전체로 StandardScaler + KMeans 학습."""
    valid = feats.dropna()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init=C.N_INIT)),
    ])
    pipe.fit(valid.to_numpy())
    print(f"KMeans 학습 완료: 유효 {len(valid):,}행 / 전체 {len(feats):,}행")
    return pipe


def _centers_original(pipe: Pipeline) -> np.ndarray:
    """스케일 역변환된 원단위 군집 중심 (k × 2F)."""
    return pipe.named_steps["scaler"].inverse_transform(
        pipe.named_steps["kmeans"].cluster_centers_)


def relabel_by_turbidity(pipe: Pipeline) -> np.ndarray:
    """착수정_TB__mean 오름차순으로 라벨 재배열 → cluster0 = 저탁도.

    반환: label_map — label_map[KMeans 원라벨] = 새 라벨.
    """
    tb_idx = feature_columns().index(f"{C.TURBIDITY_COL}__mean")
    tb_means = _centers_original(pipe)[:, tb_idx]
    order = np.argsort(tb_means)              # order[new] = raw
    label_map = np.empty_like(order)
    label_map[order] = np.arange(len(order))  # label_map[raw] = new
    return label_map


def predict_labels(pipe: Pipeline, label_map: np.ndarray,
                   feats: pd.DataFrame) -> pd.Series:
    """전 시점 라벨 (Int64). NaN 특징행(선두 9행·결측 창)은 NA."""
    labels = pd.Series(pd.NA, index=feats.index, dtype="Int64", name=C.LABEL_COL)
    valid = feats.dropna()
    if len(valid):
        raw = pipe.predict(valid.to_numpy())
        labels.loc[valid.index] = label_map[raw]
    return labels


def compute_centers(pipe: Pipeline, label_map: np.ndarray) -> pd.DataFrame:
    """새 라벨 순서로 정렬된 원단위 중심표."""
    centers = _centers_original(pipe)
    order = np.argsort(label_map)             # order[new] = raw
    out = pd.DataFrame(centers[order], columns=feature_columns())
    out.insert(0, C.LABEL_COL, np.arange(len(order)))
    return out


def silhouette_sampled(pipe: Pipeline, feats: pd.DataFrame, labels: pd.Series,
                       n: int = C.SIL_SAMPLE,
                       random_state: int = C.RANDOM_STATE) -> float:
    """스케일 공간 silhouette (표본, 참고용)."""
    valid = feats.dropna()
    lab = labels.loc[valid.index].astype(int)
    if len(valid) > n:
        idx = valid.sample(n, random_state=random_state).index
        valid, lab = valid.loc[idx], lab.loc[idx]
    scaled = pipe.named_steps["scaler"].transform(valid.to_numpy())
    return float(silhouette_score(scaled, lab.to_numpy()))


def save_model(pipe: Pipeline, label_map: np.ndarray, path=C.MODEL_PATH) -> None:
    """온라인 사용까지 자기완결인 번들 저장."""
    joblib.dump({
        "pipeline": pipe,
        "label_map": label_map,
        "feature_cols": feature_columns(),
        "raw_cols": C.RAW_FEATURES,
        "window": C.WINDOW,
    }, path)
    print(f"모델 저장: {path}")


def load_model(path=C.MODEL_PATH) -> dict:
    return joblib.load(path)


def assign_label(window_df: pd.DataFrame, bundle: dict):
    """온라인 헬퍼: 최근 10행(1분×10, 시간 오름차순) → 라벨 1개.

    배치(run_clustering) 결과는 이 함수를 매 분 반복한 것과 동일하다.
    윈도우에 NaN이 있으면 None 반환.
    """
    feat = window_features_single(window_df, cols=bundle["raw_cols"],
                                  window=bundle["window"])
    if not np.isfinite(feat).all():
        return None
    raw = bundle["pipeline"].predict(feat)[0]
    return int(bundle["label_map"][raw])


def build_report(labels: pd.Series, silhouette: float) -> pd.DataFrame:
    """cluster_report.csv: 라벨별 건수/비율 + 유효/제외 행수 + silhouette."""
    n_total = len(labels)
    n_valid = int(labels.notna().sum())
    rows = []
    for cid in range(C.K):
        n = int((labels == cid).sum())
        rows.append({"군집": f"cluster{cid}", "행수": n,
                     "비율(%)": round(n / n_valid * 100, 2)})
    rows.append({"군집": "라벨없음(NA)", "행수": n_total - n_valid,
                 "비율(%)": round((n_total - n_valid) / n_total * 100, 2)})
    rep = pd.DataFrame(rows)
    rep["전체행"] = n_total
    rep["silhouette(20k표본)"] = round(silhouette, 4)
    return rep
