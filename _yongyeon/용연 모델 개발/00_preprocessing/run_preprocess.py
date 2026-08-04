# -*- coding: utf-8 -*-
"""용연 raw_data 전 컬럼 전처리 실행 스크립트 (덕남 이식본).

실행: python run_preprocess.py   (00_preprocessing 디렉터리에서)
산출:
  dataset/preprocessing_data/용연_전처리완료.parquet  (측정 36컬럼 정제값)
  output/preprocess_report.csv                       (컬럼별 방법별 이상치 건수)
  output/compare/<컬럼>.png                          (전처리 전/후 비교)
"""
from preprocessing import config as C
from preprocessing import io_utils, pipeline, visualization


def main():
    io_utils.setup_output()
    io_utils.copy_raw_if_missing()
    df = io_utils.load_raw()
    print(f"로드: {C.RAW_PARQUET.name} — {len(df):,}행 × {df.shape[1]}컬럼 "
          f"({df.index.min()} ~ {df.index.max()})")

    df_clean, report, masks, fill_flags = pipeline.run_pipeline(df)

    io_utils.save_csv(report, C.OUTPUT_DIR / "preprocess_report.csv")
    print(f"\n리포트 저장: {C.OUTPUT_DIR / 'preprocess_report.csv'}")
    print(report.to_string(index=False))

    df_clean.to_parquet(C.OUT_PARQUET)
    print(f"\n전처리 데이터 저장: {C.OUT_PARQUET} — {len(df_clean):,}행 × {df_clean.shape[1]}컬럼")

    fill_flags.to_parquet(C.OUT_FLAGS)
    print(f"채움 플래그 저장: {C.OUT_FLAGS} — 총 채움 {int(fill_flags.to_numpy().sum()):,}칸 "
          f"(내부결측 {C.MAX_FILL_GAP_MIN}분 초과는 NaN 유지)")

    visualization.plot_all(df, df_clean, masks)
    visualization.plot_rate_heatmap(report, len(df), C.OUTPUT_DIR / "outlier_rate_heatmap.png")
    print("\n전처리 완료.")


if __name__ == "__main__":
    main()
