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
        if len(table) < 1000:
            print(f"cluster{cid}: 학습 테이블 {len(table)}행 — 데이터 부족으로 제외")
            summary.append({"cluster": cid, "test행": 0, "비고": "데이터 부족 제외"})
            continue
        models = dose_model.train_dose_models(table)
        metrics = dose_model.eval_models(models, table)
        print("추천용 모델 test: " + " | ".join(
            f"{n} R²={metrics[f'{n}_r2']:.3f}/SMAPE={metrics[f'{n}_smape']:.2f}%"
            for n in R.TARGET_NAMES))
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
