# -*- coding: utf-8 -*-
"""응집제 투입량 추천: 단조제약 모델 재학습 → 백테스트 → 리포트.

실행: python run_recommend.py   (03_dose_recommend 디렉터리에서)
"""
import numpy as np
import pandas as pd

from recommend import config as R
from recommend import backtest, dose_model, recommender

from coag import datasets, io_utils


def main():
    io_utils.setup_output()
    R.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    R.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for cid in R.CLUSTERS:
        print(f"\n===== cluster{cid} =====")
        table, lags = dose_model.build_dataset(cid)
        models = dose_model.train_dose_models(table)
        metrics = dose_model.eval_models(models, table)
        print(f"추천용 모델 test: b1 R²={metrics['침전지1_r2']:.3f}/"
              f"SMAPE={metrics['침전지1_smape']:.2f}% | "
              f"b2 R²={metrics['침전지2_r2']:.3f}/SMAPE={metrics['침전지2_smape']:.2f}%")
        # 단조성 확인: 스윕 예측이 후보축으로 비증가인지
        _, _, df_te = datasets.chrono_split(table)
        pred = recommender.sweep_predict(models, df_te.iloc[:200])
        assert (np.diff(pred, axis=1) <= 1e-9).all(), "단조성 위반"
        print("단조성 검증 OK (주입률↑ → 예측탁도 비증가)")

        dose_model.save_models(cid, models, lags, metrics)
        rec = backtest.run_backtest(cid, models, table)
        summary.append(backtest.summarize(cid, rec, metrics))
        backtest.plot_backtest(cid, rec)
        backtest.plot_response_curves(cid, models, table)
        rec.reset_index().to_csv(R.OUTPUT_DIR / f"cluster{cid}_추천이력.csv",
                                 index=False, encoding="utf-8-sig")

    summ = pd.DataFrame(summary)
    summ.to_csv(R.OUTPUT_DIR / "백테스트_요약.csv", index=False, encoding="utf-8-sig")
    print("\n===== 백테스트 요약 =====")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
