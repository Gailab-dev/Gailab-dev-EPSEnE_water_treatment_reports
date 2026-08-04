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
    """용연 기술진단서 수치 재현 검증 (실패 시 assert)."""
    # 표 2.3-10 (평균운영조건 15℃, 지별 G값) — K0_DIAG 앵커 표
    ref_15 = {"#1": (36.1, 29.3, 16.1), "#2": (36.5, 29.2, 15.4),
              "#3": (37.2, 26.5, 16.4), "#4": (37.5, 29.8, 16.5),
              "#5": (36.3, 28.6, 16.0), "#6": (39.1, 31.1, 19.7),
              "#7": (37.6, 26.4, 19.1)}
    # 표 2.3-11 (최악조건 5℃, 지별 G값) — 수온 스케일링 교차 검증
    ref_5 = {"#1": (31.2, 25.4, 13.9), "#2": (31.6, 25.3, 13.4),
             "#3": (32.2, 23.0, 14.2), "#4": (32.5, 25.8, 14.3),
             "#5": (31.4, 24.8, 13.8), "#6": (33.9, 26.9, 17.1),
             "#7": (32.5, 22.8, 16.6)}
    # 지별 진단(K0_DIAG) 검증: 실측 rpm → G
    errs = []
    for ref, temp in [(ref_15, 15.0), (ref_5, 5.0)]:
        for basin, gs in ref.items():
            calc = gvalue.g_from_rpm(np.array(C.BASIN_RPM[basin]), temp)
            errs += [abs(c - g) / g for c, g in zip(calc, gs)]
    err_diag = max(errs)
    assert err_diag < 0.02, f"표 2.3-10/11 재현 오차 {err_diag:.1%}"

    # 표준운영 조견표(K0_STD) 검증: 목표 G 50/30/10 → 권장 rpm (표 2.3-12)
    ref_312 = {5: (2.19, 1.56, 0.75), 10: (2.08, 1.48, 0.71),
               15: (1.99, 1.42, 0.68), 20: (1.91, 1.36, 0.65),
               25: (1.84, 1.31, 0.63), 30: (1.77, 1.26, 0.61)}
    err_std = max(abs(float(gvalue.rpm_for_g(C.G_TARGETS[s], t)) - r) / r
                  for t, rpms in ref_312.items() for s, r in zip((1, 2, 3), rpms))
    assert err_std < 0.02, f"표 2.3-12 조견표 재현 오차 {err_std:.1%}"

    g_h = float(gvalue.hydraulic_g(20, hrt_s=C.HYD_HRT_S["설계"]))
    assert abs(g_h - 571.1) / 571.1 < 0.02, f"수류혼화 G 오차: {g_h:.1f}"
    v = float(gvalue.tip_speed_cm_s(1.82))
    assert abs(v - 33.43) / 33.43 < 0.01, f"주변속도 오차: {v:.2f}"
    print(f"검증 통과 (이원계수):\n"
          f"  지별 진단 표 2.3-10/11 재현 오차 {err_diag:.2%} (42값, K0_DIAG={gvalue.K0_DIAG:.5f})\n"
          f"  표준운영 조견표 2.3-12 재현 오차 {err_std:.2%} (18값, K0_STD={gvalue.K0_STD:.5f})\n"
          f"  수류혼화 G(설계 20℃, 낙차 0.58m) {g_h:.1f}/sec (표 571.1)\n"
          f"  주변속도 {v:.2f}cm/s@1.82rpm (표 33.43)")


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
