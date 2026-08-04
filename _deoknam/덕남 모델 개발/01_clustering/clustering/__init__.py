# -*- coding: utf-8 -*-
"""덕남 원수성상 슬라이딩 윈도우 군집화 패키지.

사용 예 (배치):
    from clustering import config, io_utils, features, model
    io_utils.setup_output()
    df = io_utils.load_preprocessed()
    feats = features.make_window_features(df)
    pipe = model.fit_kmeans(feats)

사용 예 (온라인, 매 1분):
    bundle = model.load_model()
    label = model.assign_label(recent_10min_df, bundle)
"""
from . import config
from .features import feature_columns, make_window_features, window_features_single
from .io_utils import load_preprocessed, save_csv, save_labeled, setup_output
from .model import (assign_label, build_report, compute_centers, fit_kmeans,
                    load_model, predict_labels, relabel_by_turbidity,
                    save_model, silhouette_sampled)

__all__ = [
    "config", "setup_output", "load_preprocessed", "save_csv", "save_labeled",
    "feature_columns", "make_window_features", "window_features_single",
    "fit_kmeans", "relabel_by_turbidity", "predict_labels", "compute_centers",
    "silhouette_sampled", "save_model", "load_model", "assign_label", "build_report",
]
