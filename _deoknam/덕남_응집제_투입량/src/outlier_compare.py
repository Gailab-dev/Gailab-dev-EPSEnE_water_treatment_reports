# -*- coding: utf-8 -*-
"""수질 지표 이상치 탐지 4기법 비교 + 룰베이스 + 탁도 후향 규칙.

공정 컬럼(PAC 주입/유량/레벨/염소주입 제어)은 전처리 제외.
수질 측정값(원수·침전지·여과지·정수지의 pH/탁도/온도/전도도/알칼리도/잔류염소)만 대상.

방법:
  1) z-score (롤링)   2) IQR (롤링)   3) IsolationForest   4) 칼만 필터(1D)
  + 룰베이스(물리 유효범위)
  + 탁도 후향 규칙: 원수 탁도 급등이 침전지에 전파됐는지로 센서오류 판별

실행: python -m src.outlier_compare
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import common as C

INTEGRATED = C.DATA_DIR / "덕남_응집제공정_소독공정_통합.parquet"

# 수질 변수: (컬럼, 물리 유효범위 lo, hi). 범위 밖은 룰베이스 이상치.
QUALITY_VARS = {
    "RCS_6.AI.착수정_PH": (5.5, 9.0),
    "RCS_6.AI.착수정_TB": (0.0, 50.0),
    "RCS_6.AI.착수정_AL": (5.0, 100.0),
    "RCS_6.AI.착수정_온도": (0.0, 35.0),
    "RCS_6.AI.착수정_전기전도도": (40.0, 200.0),
    "RCS_6.AI.침전지1_PH": (5.5, 9.0),
    "RCS_6.AI.침전지1_탁도": (0.0, 3.0),
    "RCS_6.AI.침전지2_PH": (5.5, 9.0),
    "RCS_6.AI.침전지2_탁도": (0.0, 3.0),
    "RCS_6.AI.여과지_PH": (5.5, 9.0),
    "RCS_6.AI.여과지_탁도": (0.0, 3.0),
    "RCS_6.AI.정수_수온": (0.0, 35.0),
    "RCS_6.AI.정수지_PH": (5.5, 9.0),
    "RCS_6.AI.정수지_탁도": (0.0, 3.0),
    "RCS_6.AI.침전지1_잔류염소": (0.0, 3.0),
    "RCS_6.AI.침전지2_잔류염소": (0.0, 3.0),
    "RCS_6.AI.정수지_잔류염소": (0.0, 3.0),
    "RCS_3.FCC_19.AI.통합여과수_잔류염소": (0.0, 3.0),
    "주입변경.정수지_잔류염소": (0.0, 3.0),
}

# 비교 그래프에서 상세히 볼 대표 변수
FOCUS = [
    "RCS_6.AI.착수정_TB",
    "RCS_6.AI.침전지1_탁도",
    "RCS_6.AI.정수지_탁도",
    "RCS_6.AI.착수정_전기전도도",
]

ROLL = "6h"
MINP = 30


def z_mask(s, thr=3.0):
    m = s.rolling(ROLL, center=True, min_periods=MINP).mean()
    sd = s.rolling(ROLL, center=True, min_periods=MINP).std()
    return ((s - m).abs() / sd.replace(0, np.nan) > thr).fillna(False)


def iqr_mask(s, k=3.0):
    q1 = s.rolling(ROLL, center=True, min_periods=MINP).quantile(0.25)
    q3 = s.rolling(ROLL, center=True, min_periods=MINP).quantile(0.75)
    iqr = q3 - q1
    return ((s < q1 - k * iqr) | (s > q3 + k * iqr)).fillna(False)


def isoforest_mask(s, contam=0.005):
    feat = pd.DataFrame({
        "x": s,
        "resid": s - s.rolling(11, center=True, min_periods=1).median(),
        "diff": s.diff(),
    }).fillna(0.0)
    iso = IsolationForest(contamination=contam, n_estimators=100, random_state=C.RANDOM_STATE, n_jobs=-1)
    return pd.Series(iso.fit_predict(feat.values) == -1, index=s.index)


def kalman_mask(s, thr=6.0):
    """1D 랜덤워크 칼만: 예측 대비 innovation이 thr·sqrt(S) 초과면 이상치."""
    x = s.to_numpy(dtype=float)
    n = len(x)
    finite = x[np.isfinite(x)]
    r = np.var(np.diff(finite)) * 2.0 if len(finite) > 1 else 1.0  # 관측 노이즈(정상 변동 허용)
    q = r * 1e-3  # 프로세스 노이즈
    flag = np.zeros(n, dtype=bool)
    xh = finite[0] if len(finite) else 0.0
    P = 1.0
    for i in range(n):
        Pm = P + q
        xi = x[i]
        if np.isfinite(xi):
            S = Pm + r
            inno = xi - xh
            if abs(inno) > thr * np.sqrt(S):
                flag[i] = True
                K = 0.0  # 이상치는 상태 갱신 억제
            else:
                K = Pm / S
            xh = xh + K * inno
            P = (1 - K) * Pm
        else:
            P = Pm
    return pd.Series(flag, index=s.index)


METHODS = {"룰베이스": None, "z-score": z_mask, "IQR": iqr_mask, "IsoForest": isoforest_mask, "칼만": kalman_mask}


def raw_turbidity_retro(df):
    """탁도 후향 규칙: 원수 탁도 급등 후 침전지 1·2 둘 다 무반응이면 센서오류."""
    raw = df["RCS_6.AI.착수정_TB"]
    sed1 = df["RCS_6.AI.침전지1_탁도"].rolling(15, center=True, min_periods=1).median()
    sed2 = df["RCS_6.AI.침전지2_탁도"].rolling(15, center=True, min_periods=1).median()
    med = raw.rolling("6h", center=True, min_periods=30).median()
    resid = raw - med
    spike = ((resid > 3.0) & (raw > 5.0)).fillna(False)
    flag = pd.Series(False, index=df.index)
    sp = raw.index[spike]
    if len(sp) == 0:
        return flag, 0, 0
    groups, cur = [], [sp[0]]
    for t in sp[1:]:
        (cur.append(t) if t - cur[-1] <= pd.Timedelta("30min") else (groups.append(cur), cur := [t]))
    groups.append(cur)
    wide = (resid > 1.5).fillna(False)
    n_sensor = 0
    for g in groups:
        t = resid.loc[g[0]:g[-1]].idxmax()
        r1 = sed1.loc[t + pd.Timedelta("2h"):t + pd.Timedelta("5h")].max() - sed1.loc[t - pd.Timedelta("3h"):t].median()
        r2 = sed2.loc[t + pd.Timedelta("2h"):t + pd.Timedelta("5h")].max() - sed2.loc[t - pd.Timedelta("3h"):t].median()
        if (r1 < 0.1 or np.isnan(r1)) and (r2 < 0.1 or np.isnan(r2)):
            seg = wide.loc[t - pd.Timedelta("1h"):t + pd.Timedelta("1h")]
            on = seg[seg].index
            flag.loc[(on.min() if len(on) else t):(on.max() if len(on) else t)] = True
            n_sensor += 1
    return flag, len(groups), n_sensor


def main():
    C.setup_output()
    df = pd.read_parquet(INTEGRATED).sort_index()
    n0 = len(df)

    # 변수 × 방법 이상치 마스크
    masks = {}  # (col, method) -> bool Series
    summary = []
    for col, (lo, hi) in QUALITY_VARS.items():
        valid = df[col].notna().sum()
        row = {"변수": col, "유효값": int(valid)}
        rule = (df[col] < lo) | (df[col] > hi)
        masks[(col, "룰베이스")] = rule.fillna(False)
        row["룰베이스"] = int(rule.sum())
        for mname, fn in METHODS.items():
            if fn is None:
                continue
            m = fn(df[col])
            masks[(col, mname)] = m
            row[mname] = int(m.sum())
            row[f"{mname}(%)"] = round(m.sum() / n0 * 100, 3)
        summary.append(row)
        print(f"완료: {col} (유효 {valid:,})")

    rep = pd.DataFrame(summary)
    C.save_csv(rep, C.RESULTS_DIR / "outlier_methods_summary.csv")
    print("\n===== 방법별 이상치 탐지 수 =====")
    show = ["변수", "룰베이스", "z-score", "IQR", "IsoForest", "칼만"]
    print(rep[show].to_string(index=False))

    # 탁도 후향 규칙
    retro_flag, n_ev, n_sensor = raw_turbidity_retro(df)
    print(f"\n[탁도 후향 규칙] 원수 급등 {n_ev}건 중 침전지 무반응(센서오류) {n_sensor}건 → {int(retro_flag.sum())}행")

    _plot_method_compare(df, masks, retro_flag)
    _plot_summary_heatmap(rep, n0)


def _plot_method_compare(df, masks, retro_flag):
    import matplotlib.pyplot as plt

    methods = ["룰베이스", "z-score", "IQR", "IsoForest", "칼만"]
    colors = {"룰베이스": "#c44e52", "z-score": "#dd8452", "IQR": "#55a868",
              "IsoForest": "#8172b2", "칼만": "#4878b0"}
    env = df[FOCUS].resample("10min").max()  # 스파이크 보존

    fig, axes = plt.subplots(len(FOCUS), 1, figsize=(15, 3.4 * len(FOCUS)), sharex=True)
    for ax, col in zip(axes, FOCUS):
        ax.plot(env.index, env[col], color="#999999", lw=0.5, zorder=1)
        for i, mth in enumerate(methods):
            m = masks.get((col, mth))
            if m is None or not m.any():
                continue
            pts = df[col][m]
            ax.scatter(pts.index, pts.values, s=8, color=colors[mth], label=mth,
                       marker="o", alpha=0.6, zorder=2 + i)
        if col == "RCS_6.AI.착수정_TB" and retro_flag.any():
            pts = df[col][retro_flag]
            ax.scatter(pts.index, pts.values, s=14, facecolors="none", edgecolors="black",
                       label="탁도후향(센서오류)", zorder=10)
        ax.set_ylabel(col.split(".")[-1], fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title("수질 지표 이상치 탐지 4기법 비교 (회색: 원본 10분 max, 점: 각 방법이 지목한 이상치)")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "outlier_method_compare.png", dpi=110)
    plt.close(fig)
    print("저장: results/plots/outlier_method_compare.png")


def _plot_summary_heatmap(rep, n0):
    import matplotlib.pyplot as plt

    methods = ["룰베이스", "z-score", "IQR", "IsoForest", "칼만"]
    mat = rep.set_index("변수")[methods].div(n0 / 100)  # 탐지율 %
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(mat) + 2))
    im = ax.imshow(mat.values, aspect="auto", cmap="OrRd", vmin=0, vmax=min(3, mat.values.max()))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([v.split(".")[-1] for v in mat.index], fontsize=8)
    for i in range(len(mat)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("변수 × 방법 이상치 탐지율 (%)")
    fig.colorbar(im, ax=ax, label="탐지율 %")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "outlier_summary_heatmap.png", dpi=120)
    plt.close(fig)
    print("저장: results/plots/outlier_summary_heatmap.png")


if __name__ == "__main__":
    main()
