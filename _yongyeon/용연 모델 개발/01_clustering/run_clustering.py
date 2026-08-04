# -*- coding: utf-8 -*-
"""용연 원수성상 슬라이딩 윈도우 군집화 실행 스크립트.

실행: python run_clustering.py   (01_clustering 디렉터리에서)
산출:
  dataset/clustering_data/용연_군집라벨.parquet   (원본 44 + 피처 10 + cluster_label)
  dataset/clustering_data/용연_cluster{0,1,2}.parquet
  output/model/용연_kmeans_k3.joblib
  output/cluster_report.csv, cluster_centers.csv
  output/plots/*.png
"""
import pandas as pd

from clustering import config as C
from clustering import features, io_utils, model, visualization


def main():
    io_utils.setup_output()
    df = io_utils.load_preprocessed()
    print(f"로드: {C.IN_PARQUET.name} — {len(df):,}행 ({df.index.min()} ~ {df.index.max()})")

    feats = features.make_window_features(df)
    print(f"윈도우 특징 생성: {feats.shape[1]}차원, 유효 {feats.dropna().shape[0]:,}행")

    pipe = model.fit_kmeans(feats)
    label_map = model.relabel_by_turbidity(pipe)
    labels = model.predict_labels(pipe, label_map, feats)
    centers = model.compute_centers(pipe, label_map)
    sil = model.silhouette_sampled(pipe, feats, labels)
    print(f"silhouette (표본 {C.SIL_SAMPLE:,}): {sil:.4f}")

    model.save_model(pipe, label_map)

    df_out = pd.concat([df, feats], axis=1)
    df_out[C.LABEL_COL] = labels
    io_utils.save_labeled(df_out)

    report = model.build_report(labels, sil)
    io_utils.save_csv(report, C.OUTPUT_DIR / "cluster_report.csv")
    io_utils.save_csv(centers, C.OUTPUT_DIR / "cluster_centers.csv")
    print(f"\n리포트 저장: cluster_report.csv, cluster_centers.csv")
    print(report.to_string(index=False))
    print("\n원단위 군집 중심:")
    print(centers.to_string(index=False))

    visualization.plot_all(df, feats, labels, pipe)
    print("\n군집화 완료.")


if __name__ == "__main__":
    main()

