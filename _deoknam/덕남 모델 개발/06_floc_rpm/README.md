# 06_floc_rpm — 혼화·플록형성 G/RPM 규칙기반 추천엔진

`혼화응집공정/설계문서.md` §5~12의 오프라인 구현(문서 §17의 1~2단계).
RPM 실측·인버터·지별유량 태그가 없는 현재 데이터에서 규칙기반 추천 파이프라인을
전기간(950,286행, 1분) 백테스트하고 침전수 탁도와 시간정렬 사후평가를 수행한다.

## 실행

```bash
cd 06_floc_rpm
python run_floc_rpm.py
```

실행 순서: §16.1 수식 단위시험(25건, 실패 시 assert 중단) → 전기간 추천 산출
→ §16.2 백테스트 통계(가용률·위반 0건 assert) → §12 시간창 비교·사후평가
→ CSV/parquet/PNG 저장.

## 구조

| 파일 | 역할 |
| --- | --- |
| `floc_rpm/config.py` | 설계문서 수치의 단일 출처 (표 2.3-13, 낙차 0.45m, Gt 범위 등) |
| `floc_rpm/mixing_link.py` | `05_mixing_gvalue` 재사용 브리지 (sys.path 조작 격리) |
| `floc_rpm/io_utils.py` | parquet 로드 + 1분 정규 그리드 reindex |
| `floc_rpm/validity.py` | §4.3 HOLD/WARN 마스크 (5분 연속결측 run-length 등) |
| `floc_rpm/flow.py` | §6 유량 정규화·지별 배분·체류시간·상류지연 |
| `floc_rpm/alignment.py` | 가변시차 정렬 (상류 §6.4 / 하류 §12) |
| `floc_rpm/recommend.py` | §10 표준 RPM 룩업보간 + G^(2/3) 보정 + K0 검산 + §8 낙차부 G |
| `floc_rpm/constraints.py` | §10.4 상하한·점감·Gt 검사 |
| `floc_rpm/reasons.py` | §15.3 사유코드 비트마스크 + 우선순위 단일코드 |
| `floc_rpm/evaluate.py` | §12 침전탁도 시간창 비교·사후평가·§13.3 지표 |
| `floc_rpm/checks.py` | §16.1 단위시험 + §16.2 백테스트 통계 |

## 핵심 설계

- **추천 기준 = 표 2.3-13 룩업 선형보간**(§10.1). `05_mixing_gvalue`의
  K0 물리모델(G = K0·√(ρ/μ)·rpm^1.5)은 §10.3 검산 역할 — 편차 >3% 시
  `WARN_MODEL_DIVERGENCE`(§15.3 확장코드).
- **FT101 단위 안전장치**(§6.1): `FLOW_UNIT_CONFIRMED=False`인 동안
  `control_available` 전 행 False. 유량 중앙값이 타당범위(1,000~20,000㎥/h)
  밖이면 실행 자체를 중단.
- **HOLD 행은 추천 RPM NaN 마스킹**(§4.3). 인버터 태그 연결 후
  `reasons.control_available(bits, vfd_tag_present=True)`로 활성화.
- **§12 기본 시간창 = 추적자평균 322±45분**. `시간창후보_비교.csv`의 우세
  후보는 권고만 하고 `config.LAG_DEFAULT` 자동 변경은 하지 않음.
- 완화(45/27/9)·강화(50/35/12) 후보는 사후평가용 저장만(§9.3), 운영 추천은 표준.

## 산출물 (`output/`)

- `추천_전기간_시계열.parquet` — 1분 전기간: 유량·정렬수질·rpm_std/rec/safe/phys·
  Gt·reason_code·bitmask·control_available
- `추천_일별요약.csv`, `사유코드_통계.csv`, `수온별_표준RPM_보간표.csv`(룩업 vs K0),
  `지별_Gt_점검.csv`(가중/균등 시나리오), `시간창후보_비교.csv`,
  `침전탁도_사후평가.csv`(+전기간 parquet), `침전탁도_기준선지표.csv`(24h persistence),
  `검증_요약.csv`
- `images/` — 계절별_추천RPM, 사유코드_타임라인, G_drop_시계열, Gt_분포, 시간창_상관비교

## 백테스트 결과 요약 (2024-05~2026-02)

- §16.1 단위시험 25건 통과 (표 2.3-13 정확 재현, 낙차부 G 209.9 재현 등)
- 가용률: 필수입력 존재행 기준 99.1% (승인기준 ≥95%), 전체행 기준 88.0%
  (착수정 수온 결측 11.3%가 지배)
- 점감·한계 위반 0건, HOLD 행 추천값 누출 0건
- 계절 정합: 1단 추천 RPM 저수온기(<10℃) 1.99 > 고수온기(>20℃) 1.78

## 태그 추가 시 확장 지점

| 추가 태그 (설계문서 §4.4) | 코드 반영 위치 |
| --- | --- |
| 지별 운전/휴지, 지별 유량 | `flow.basin_flows`의 weights 교체 |
| 1·2·3단 RPM 실측 | `recommend` 추천 vs 실측 비교, §13 모델풀 학습 가능 |
| 인버터 Hz·Ready | `reasons.control_available(vfd_tag_present=True)` + §14 인터록 |
| 실제 응집제·주입률(mg/L) | §8.3 Letterman G 산정 해제 (`WARN_NO_ACTIVE_CHEM` 조건) |
