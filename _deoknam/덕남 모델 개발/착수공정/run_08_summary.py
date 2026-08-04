# -*- coding: utf-8 -*-
"""착수공정 1단계 — run 08: 최종 성능비교표·승인게이트 판정표·결과요약 조립 (§14).

실행: python 착수공정/run_08_summary.py  (run_01~07 산출 필요)
산출: output/1단계_성능비교_최종표.csv, 1단계_승인게이트_판정표.csv, 1단계_결과요약.md
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from intake_coag import config as C
from intake_coag import io_utils as IO

CHECKLIST = """§7.4 원인점검 체크리스트 (시험 sMAPE 5% 초과 시 — 시험자료 재튜닝 금지):
1. T_outlet 산정·시간정렬 오차 — run_01 산정표·run_02 시차검증·run_05 스크리닝 대조
2. 실제 사용 약품·실제 주입량 라벨 불확실성 — §3.3 미확보(proxy 사용 중) → 라벨 확보 필요
3. 계절·고탁도·유량급변 구간 표본 부족 — 과제B_구간별성능.csv·수질구간_표본수.csv 확인
4. 피처 누락·센서 드리프트·운전방식 변경 — 상관분석_방법별.csv·제외행_집계.csv 확인
5. 단일 모델로 설명되지 않는 계열별 공정 차이 — 침전지1 vs 2 성능 비교
→ 위 점검 후 새 시간 구간을 확보하여 다시 시험한다(§7.4)."""


def _read(name):
    return pd.read_csv(C.OUTPUT_DIR / name, encoding="utf-8-sig")


def main():
    IO.setup_output()
    b_test = _read("과제B_시험성능표.csv")
    a_test = _read("과제A_시험성능표.csv")
    bounds = IO.load_json("분할경계.json")
    sel = IO.load_json("선정_정렬조합.json")

    # ── 최종 성능비교표 (§14: 목표변수·정렬방식·모델·입력창 × 4지표) ──
    b_test.insert(0, "과제", "B.공정반응(침전수 탁도)")
    a_test.insert(0, "과제", "A.설정값 모사")
    a_test.insert(2, "정렬", "-")
    cols = ["과제", "목표변수", "정렬", "모델", "피처셋", "창",
            "RMSE", "MAE", "sMAPE", "R2", "n", "게이트(≤5%)"]
    final = pd.concat([b_test.reindex(columns=cols + ["포함률95", "구간폭", "제외율"]),
                       a_test.reindex(columns=cols)], ignore_index=True)
    IO.save_csv(final, "1단계_성능비교_최종표.csv")

    # ── 승인게이트 판정표 ─────────────────────────────────────────────
    gate = final[final["모델"].isin(C.MODELS_TABULAR + C.MODELS_SEQ)].copy()
    gate["판정"] = gate["게이트(≤5%)"]
    n_pass = int((gate["판정"] == "통과").sum())
    IO.save_csv(gate[["과제", "목표변수", "정렬", "모델", "피처셋", "창",
                      "sMAPE", "판정"]], "1단계_승인게이트_판정표.csv")

    # ── 결과요약.md ───────────────────────────────────────────────────
    res = _read("체류시간_산정표.csv")
    lagv = _read("시차검증_요약.csv")
    scr = _read("정렬스크리닝_검증성능.csv")

    def tbl(df):
        return df.to_markdown(index=False)

    fail = gate[gate["판정"] == "미통과"]
    md = f"""# 착수공정 1단계 결과요약 — 체류시간·시간정렬·후보 모델풀 비교

- 데이터: {C.DATA_PARQUET.name} (1분), 학습행 10분 스트라이드
- 분할(공통, §7.3): train < {bounds['ts1']} ≤ val < {bounds['ts2']} ≤ test
- sMAPE ε = {C.SMAPE_EPS} NTU 고정(§7.3), 승인게이트 시험 sMAPE ≤ {C.GATE_SMAPE}%(§7.4)
- 주입 입력은 전부 **proxy**(주입_유량N·주입량_AN+BN·SV) — 실투입 라벨(§3.3) 미확보.
  2단계(공정반응 기반 최적화) 승격은 라벨 확보를 전제로 한다.

## 1. 체류시간 산정 (§4)

{tbl(res)}

## 2. 시차상관·DTW 검증 (§4.5)

{tbl(lagv)}

- 침전지 응답은 빠른 첨두 성분(≈120~180분)과 주입 proxy 기준 평균 성분(≈340~530분)이
  공존 — 설계문서 §4.2.3의 '첨두 80분 vs 평균 279분' 이중응답과 정합.

## 3. 정렬조합 스크리닝 (run_05, 검증 sMAPE)

- 침전지1 top-{sel['top_k']}: {', '.join(sel['선정']['1'])}
- 침전지2 top-{sel['top_k']}: {', '.join(sel['선정']['2'])}
- 스크리닝 전량({len(scr)}조합)은 정렬스크리닝_검증성능.csv 참조.

## 4. 최종 성능표 (시험구간 1회 평가)

{tbl(final[cols].round(3))}

## 5. 승인게이트 판정

- 통과 {n_pass} / 대상 {len(gate)} (persistence 제외)
"""
    if len(fail):
        md += f"""
### 미통과 모델 ({len(fail)}건)

{tbl(fail[['과제', '목표변수', '모델', 'sMAPE']].round(3))}

{CHECKLIST}
"""
    md += """
## 6. 비고

- 시험구간이 가을~겨울(저탁도)에 편중 — 고탁도(>5 NTU) 일반화는 여름 장마 포함
  구간 재시험 권고(§7.4). 구간별 성능은 과제B_구간별성능.csv 참조.
- 과제 A 성능은 현행 운전 **모사** 성능이며 최적 주입률로 해석하지 않는다(§7.1).
- 침전수 탁도가 폐루프(주입 보상) 하에 있어 단변수 상관으로는 정렬 차이가 드러나지
  않음(run_04) — 정렬 선택은 다변수 모델 검증 성능(run_05)으로 수행했다.
"""
    with open(C.OUTPUT_DIR / "1단계_결과요약.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"저장: 1단계_성능비교_최종표.csv, 1단계_승인게이트_판정표.csv, 1단계_결과요약.md")
    print(f"게이트: 통과 {n_pass}/{len(gate)}")


if __name__ == "__main__":
    main()
