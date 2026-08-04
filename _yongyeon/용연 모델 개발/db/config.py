# -*- coding: utf-8 -*-
"""용연정수장 DB 스키마 매핑.

계측 테이블 WTP_WIDE_MEASR_YY (24컬럼) / 교반 테이블 WTP_WIDE_MXC_YY (43컬럼,
7지×3단). 모두 1분 주기, MSRMT_DT(datetime) 시계열 키.

주의: DB 계측 스키마는 학습에 쓴 parquet과 다르다.
  - DB에만 있음: 착수 망간(INT_MANG), 착수 조류(INT_ALGA)
  - DB에 없음: 원수 알칼리도, 원수 전기전도도  (parquet RAW5에는 존재)
  - 덕남과 달리 침전지 탁도는 단일(SED_TRBD), 염소는 전/중/후 분리(PRCL/MDCL/PSCL).
따라서 PIPELINE_MAP은 겹치는 컬럼만 매핑하며, 누락 원수성상(알칼리도·전기전도도)
처리 방침을 별도로 정해야 파이프라인에 직결할 수 있다.
"""
PLANT = "용연"
MEASR_TABLE = "WTP_WIDE_MEASR_YY"
MXC_TABLE = "WTP_WIDE_MXC_YY"
TIME_COL = "MSRMT_DT"

# 플록형성지: 7지 (#1~7), 각 3단. 교반 컬럼 코드 = {지}{단}, 예: MXC_GVAL_11 = 1지 1단.
N_BASINS = 7
BASIN_CODES = ["1", "2", "3", "4", "5", "6", "7"]
N_STAGES = 3

# DB 계측 컬럼 → 표준 한글명 (rename=True 시 사용)
COLUMN_MAP = {
    "WTK_MMNT_FL": "취수유량", "WTK_ACML_QTY": "취수적산",
    "CGLT_INVT": "응집제재고", "CGLT_ACML_USQTY": "응집제적산사용량",
    "PRCL_USQTY": "전염소사용량", "MDCL_USQTY": "중염소사용량",
    "PSCL_USQTY": "후염소사용량", "CL_RMND_QTY": "염소잔량",
    "INT_TRBD": "착수탁도", "INT_WTMP": "착수수온", "INT_PH": "착수pH",
    "INT_MANG": "착수망간", "INT_ALGA": "착수조류",
    "INT_PAC": "PAC주입률", "INT_PACS2": "PACS2주입률",
    "SED_TRBD": "침전탁도", "SED_RSCL": "침전잔류염소",
    "FLT_TRBD": "여과탁도", "FLT_RSCL": "여과잔류염소",
    "PUR_TRBD": "정수탁도", "PUR_RSCL": "정수잔류염소",
    "PUR_MMNT_FL": "정수유량", "PUR_ACML_QTY": "정수적산",
}

# DB 컬럼 → 기존 파이프라인(parquet) 컬럼명 (겹치는 것만; 파이프라인 직결용)
PIPELINE_MAP = {
    "INT_TRBD": "원수 탁도",
    "INT_WTMP": "원수 온도",
    "INT_PH": "원수 PH",
    "SED_TRBD": "침전지 탁도",
    "FLT_TRBD": "여과지 탁도",
    "PUR_TRBD": "정수지 탁도",
    "WTK_MMNT_FL": "착수정 유입유량(㎥/h)",
}
# parquet RAW5 중 DB에 없는 원수성상 (파이프라인 사용 시 대체·보간 필요)
MISSING_RAW = ["원수 알카리도", "원수 전기전도도"]
