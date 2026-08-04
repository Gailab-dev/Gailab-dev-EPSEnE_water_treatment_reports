# -*- coding: utf-8 -*-
"""주입률 운영자 모사 모델: 학습 → 평가 → 변경 이벤트 분석 → 시각화.

실행: python run_mimic.py   (04_dose_mimic 디렉터리에서)
"""
import sys

import numpy as np
import pandas as pd

from mimic import config as C
from mimic import data, model


def setup():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def plot_timeline(day_te, line, y_pred, path):
    import matplotlib.pyplot as plt

    y_true = day_te[f"SV{line}"].astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(day_te.index, y_true, where="post", color="#999999", lw=1.5, label="실제")
    ax.step(day_te.index, y_pred, where="post", color="#4878b0", lw=1.2, label="모사")
    ax.set_ylabel("주입률 SV (ppm)")
    ax.set_yticks(C.CLASSES)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"계열{line} 주입률: 실제 vs 모사 (test, 일 단위)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(bundle, path):
    import matplotlib.pyplot as plt

    imp = bundle["clf"].feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 2))
    ax.barh(np.array(bundle["feats"])[order], imp[order], color="#55a868")
    ax.set_title(f"계열{bundle['line']} 모사 모델 특성중요도 (AR {'포함' if bundle['use_ar'] else '제외'})")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    setup()
    day = data.build_daily_table()
    day_tr, day_te = data.split(day)
    print(f"일 단위 테이블: {len(day)}일 (train {len(day_tr)} / test {len(day_te)})")

    rows, bundles = [], {}
    for line in (1, 2):
        for use_ar, tag in [(True, "AR포함"), (False, "AR제외")]:
            b = model.train(day_tr, line, use_ar)
            m = model.evaluate(b, day_te)
            rows.append({"계열": line, "모델": f"XGB({tag})", **m})
            if use_ar:
                bundles[line] = b
                y_pred = model.predict(b, day_te)
                plot_timeline(day_te, line, y_pred,
                              C.IMAGES_DIR / f"계열{line}_실제vs모사.png")
                plot_importance(b, C.IMAGES_DIR / f"계열{line}_특성중요도.png")

        # 변경 이벤트 조건 분석 (전 기간)
        ev = model.change_event_analysis(day, line)
        ev.reset_index(names="날짜").to_csv(
            C.OUTPUT_DIR / f"계열{line}_변경이벤트_조건.csv",
            index=False, encoding="utf-8-sig")
        inc = ev[ev["방향"] == "증량"]
        dec = ev[ev["방향"] == "감량"]
        print(f"계열{line} 변경 {len(ev)}회 — 증량 {len(inc)}회(TB중앙 {inc['TB'].median():.1f}, "
              f"수온중앙 {inc['수온'].median():.1f}) / 감량 {len(dec)}회(TB중앙 {dec['TB'].median():.1f}, "
              f"수온중앙 {dec['수온'].median():.1f})")

    model.save(bundles)
    res = pd.DataFrame(rows)
    res.to_csv(C.OUTPUT_DIR / "모사모델_평가.csv", index=False, encoding="utf-8-sig")
    print("\n===== 모사 모델 평가 (test) =====")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
