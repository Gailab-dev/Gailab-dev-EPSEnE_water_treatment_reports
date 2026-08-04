# -*- coding: utf-8 -*-
"""교반기 G값·RPM 산정: 검증 → 조견표 → 지별 진단 → 일별 권장 rpm.

실행: python run_mixing_g.py   (05_mixing_gvalue 디렉터리에서)
"""
import sys

import numpy as np

from mixing_g import config as C
from mixing_g import gvalue, tables, visualization


def setup():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def validate():
    """기술진단서 수치 재현 검증 (실패 시 assert)."""
    ref13 = {5: (2.05, 1.46, 0.70), 10: (1.95, 1.39, 0.67), 15: (1.86, 1.32, 0.64),
             20: (1.79, 1.27, 0.61), 25: (1.72, 1.22, 0.59), 30: (1.66, 1.18, 0.57)}
    err = max(abs(float(gvalue.rpm_for_g(C.G_TARGETS[s], t)) - r) / r
              for t, rpms in ref13.items() for s, r in zip((1, 2, 3), rpms))
    assert err < 0.02, f"표 2.3-13 재현 오차 {err:.1%}"
    g_h = float(gvalue.hydraulic_g(20, hrt_s=C.HYD_HRT_S["평균"]))
    assert abs(g_h - 209.9) / 209.9 < 0.02, f"수류혼화 G 오차: {g_h:.1f}"
    print(f"검증 통과: 표준운영조건 표 재현 최대오차 {err:.2%}, "
          f"수류혼화 G(평균유량 20℃) {g_h:.1f}/sec (표 209.9), K0={gvalue.K0:.5f}")


def main():
    setup()
    validate()

    # 1) 수온별 권장 RPM 조견표
    tab = tables.rpm_table()
    tab.to_csv(C.OUTPUT_DIR / "수온별_권장RPM_조견표.csv", index=False,
               encoding="utf-8-sig")
    print(f"조견표 저장: 0~30℃ × 3단 ({len(tab)}행)")

    # 2) 지별 운영 진단 (진단서 실측 rpm, 최악조건 5℃ + 평균 15℃)
    for t, flow, tag in [(5.0, C.FLOW_MAX, "최악조건(5℃·최대유량)"),
                         (15.0, C.FLOW_AVG, "평균조건(15℃·평균유량)")]:
        diag = tables.diagnose_basins(t, flow)
        name = f"지별_운영진단_{tag.split('(')[0]}.csv"
        diag.to_csv(C.OUTPUT_DIR / name, index=False, encoding="utf-8-sig")
        n_bad = (diag["판정"] != "적정").sum()
        print(f"\n[{tag}] 기준 미달/이상 {n_bad}/{len(diag)}지")
        print(diag[["지", "rpm(1/2/3단)", "G 1단", "G 2단", "G 3단", "판정"]]
              .to_string(index=False))
        if t == 5.0:
            visualization.plot_basin_diagnosis(diag, t, C.IMAGES_DIR / "지별_G진단.png")

    # 3) 일별 권장 rpm (데이터 연계)
    daily = tables.daily_recommended_rpm()
    daily.reset_index(names="날짜").to_csv(C.OUTPUT_DIR / "일별_권장RPM.csv",
                                          index=False, encoding="utf-8-sig")
    w = daily[daily["수온"] < 10]["1단 권장rpm"].mean()
    s = daily[daily["수온"] > 20]["1단 권장rpm"].mean()
    print(f"\n일별 권장 rpm {len(daily)}일 — 1단 기준 저수온기(<10℃) 평균 {w:.2f} "
          f"vs 고수온기(>20℃) {s:.2f} (겨울↑ 확인)")
    assert w > s, "겨울 권장 rpm이 여름보다 낮음 — 온도 의존성 오류"

    # 4) 시각화
    visualization.plot_g_rpm_curves(C.IMAGES_DIR / "G-RPM곡선.png")
    visualization.plot_daily_rpm(daily, C.IMAGES_DIR / "일별_권장RPM.png")
    print("시각화 저장: G-RPM곡선, 지별_G진단, 일별_권장RPM")


if __name__ == "__main__":
    main()
