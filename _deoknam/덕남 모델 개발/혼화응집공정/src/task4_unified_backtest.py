# -*- coding: utf-8 -*-
"""Task 4: 군집 미분할 통합 학습 — 정제 통합 데이터 전체로 침전지 탁도 예측.

클러스터로 나누지 않고 정제 통합 데이터(10분 리샘플, gap 없는 연속 시계열)
전체를 하나로 학습. 체류시간(lead)을 통합 후보로 그리드 탐색.
AR 피처(직전 침전지 탁도) 포함, 확장 윈도우 롤링 재학습 백테스트.

실행: python -m src.task4_unified_backtest
"""
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from . import common as C

PRE = C.DATA_DIR / "덕남_통합_전처리.parquet"
LEADS = [150, 180, 210, 240, 270]  # 통합 체류시간 후보(분)
SCHEDULES = {"1주": 1, "4주": 4, "고정": None}
INIT_FRAC = 0.5
MIN_EVAL_N = 50
RS = C.RANDOM_STATE


def get_models():
    return {
        "ExtraTrees600": ExtraTreesRegressor(n_estimators=600, n_jobs=-1, random_state=RS),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RS, verbose=-1,
        ),
    }


def load_unified():
    df = pd.read_parquet(PRE).sort_index()
    cols = C.RAW_FEATURES + [C.COL_DOSE, C.TARGET_TB[1], C.TARGET_TB[2]]
    sub = df[[f"{c}_clean" for c in cols]].resample("10min").median()
    sub.columns = cols
    return sub.reset_index()


def build_dataset(base, basin, lead):
    df = C.shift_target_by_time(base, C.TARGET_TB[basin], lead)
    feats = C.RAW_FEATURES + [C.COL_DOSE, C.TARGET_TB[basin]]  # AR: 현재 침전지 탁도
    df = df.dropna(subset=feats + ["y_future"]).sort_values(C.COL_DT).reset_index(drop=True)
    return df, feats


def run_backtest(df, feats, lead, model_factory):
    cut = int(len(df) * INIT_FRAC)
    T0 = df[C.COL_DT].iloc[cut - 1]
    bt = df.iloc[cut:].reset_index(drop=True)
    week = np.floor((bt[C.COL_DT] - T0) / pd.Timedelta(days=7)).astype(int).to_numpy()

    cutoff_idx = {s: ((week // w) * w if w else np.zeros(len(bt), dtype=int)) for s, w in SCHEDULES.items()}
    needed = sorted(set(np.concatenate(list(cutoff_idx.values())).tolist()))

    preds = {}
    for j in needed:
        Tj = T0 + pd.Timedelta(days=7 * j)
        train = df[df[C.COL_DT] <= Tj - pd.Timedelta(minutes=lead)]  # 누수 방지
        if len(train) < 100:
            continue
        m = model_factory()
        m.fit(train[feats], train["y_future"])
        p = np.full(len(bt), np.nan)
        mask = (bt[C.COL_DT] >= Tj).to_numpy()
        p[mask] = m.predict(bt.loc[mask, feats])
        preds[j] = p

    y_true = bt["y_future"].to_numpy()
    out = {}
    for s in SCHEDULES:
        idx = cutoff_idx[s]
        if any(j not in preds for j in set(idx.tolist())):
            continue
        y_pred = np.array([preds[j][i] for i, j in enumerate(idx)])
        ok = ~np.isnan(y_pred)
        out[s] = C.compute_metrics(y_true[ok], y_pred[ok])
    return out, len(bt)


def main():
    C.setup_output()
    base = load_unified()
    print(f"통합 데이터: {len(base):,}행 ({base[C.COL_DT].min().date()} ~ {base[C.COL_DT].max().date()})\n")

    rows = []
    for basin in (1, 2):
        for lead in LEADS:
            df, feats = build_dataset(base, basin, lead)
            for mname in get_models():
                t = time.time()
                res, n_bt = run_backtest(df, feats, lead, lambda mn=mname: get_models()[mn])
                for sched, m in res.items():
                    rows.append({
                        "basin": basin, "lead": lead, "model": mname, "schedule": sched,
                        "n_bt": n_bt, **{k: round(v, 4) for k, v in m.items()},
                        "passed": C.passes(m),
                    })
                print(f"침전지{basin} lead{lead} {mname}: {time.time()-t:.0f}초")

    res = pd.DataFrame(rows)
    C.save_csv(res, C.RESULTS_DIR / "task4_unified_results.csv")

    print("\n===== Task 4 통합 학습 요약 (재학습 1주 기준) =====")
    one = res[res.schedule == "1주"].sort_values(["basin", "r2"], ascending=[True, False])
    print(one[["basin", "lead", "model", "r2", "smape", "rmse", "passed"]].to_string(index=False))

    print("\n--- 침전지별 최고 R² (전 스케줄) ---")
    for basin in (1, 2):
        g = res[res.basin == basin]
        b = g.loc[g.r2.idxmax()]
        print(f"침전지{basin}: lead{int(b.lead)} {b.model} {b.schedule} → R²={b.r2:.3f}, SMAPE={b.smape:.1f}%")

    cand = res[res.passed]
    print(f"\n기준(R²≥0.9 & SMAPE≤10%) 통과: {len(cand)}건")
    if len(cand):
        print(cand[["basin", "lead", "model", "schedule", "r2", "smape"]].to_string(index=False))


if __name__ == "__main__":
    main()
