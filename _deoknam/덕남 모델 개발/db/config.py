# -*- coding: utf-8 -*-
"""덕남정수장 DB 스키마 매핑.

계측 테이블 WTP_WIDE_MEASR_DK (29컬럼) / 교반 테이블 WTP_WIDE_MXC_DK (73컬럼,
12지×3단). 모두 1분 주기, MSRMT_DT(datetime) 시계열 키.

주의: DB 계측 스키마는 학습에 쓴 parquet과 다르다.
  - DB에만 있음: 착수 망간(INT_MANG), 착수 조류(INT_ALGA)
  - DB에 없음: 착수 알칼리도, 착수 전기전도도  (parquet RAW5에는 존재)
따라서 PIPELINE_MAP은 겹치는 컬럼만 매핑하며, 파이프라인에 그대로 넣으려면
누락 원수성상(알칼리도·전기전도도) 처리 방침을 별도로 정해야 한다.
"""
PLANT = "덕남"
MEASR_TABLE = "WTP_WIDE_MEASR_DK"
MXC_TABLE = "WTP_WIDE_MXC_DK"
TIME_COL = "MSRMT_DT"

# 플록형성지: 12지 (1계열 #101~106, 2계열 #207~212), 각 3단.
# 교반 컬럼 코드 = {지}{단}, 예: MXC_GVAL_1011 = 101지 1단.
N_BASINS = 12
BASIN_CODES = ["101", "102", "103", "104", "105", "106",
               "207", "208", "209", "210", "211", "212"]
N_STAGES = 3

# DB 계측 컬럼 → 표준 한글명 (rename=True 시 사용)
COLUMN_MAP = {
    "WTK_MMNT_FL": "취수유량", "WTK_ACML_QTY": "취수적산",
    "CGLT_INVT": "응집제재고", "CGLT_ACML_USQTY": "응집제적산사용량",
    "PRMDCL_USQTY": "전중염소사용량", "PSCL_USQTY": "후염소사용량",
    "CL_RMND_QTY": "염소잔량",
    "INT_TRBD": "착수탁도", "INT_WTMP": "착수수온", "INT_PH": "착수pH",
    "INT_MANG": "착수망간", "INT_ALGA": "착수조류",
    "INT_PAC": "PAC주입률", "INT_PACS2": "PACS2주입률",
    "SED_TRBD1": "침전1탁도", "SED_TRBD2": "침전2탁도",
    "SED_RSCL1": "침전1잔류염소", "SED_RSCL2": "침전2잔류염소",
    "FLT_TRBD1": "여과1탁도", "FLT_TRBD2": "여과2탁도",
    "FLT_RSCL1": "여과1잔류염소", "FLT_RSCL2": "여과2잔류염소",
    "PUR_TRBD1": "정수1탁도", "PUR_TRBD2": "정수2탁도",
    "PUR_RSCL1": "정수1잔류염소", "PUR_RSCL2": "정수2잔류염소",
    "PUR_MMNT_FL": "정수유량", "PUR_ACML_QTY": "정수적산",
}

# DB 컬럼 → 기존 파이프라인(parquet) 컬럼명 (겹치는 것만; 파이프라인 직결용)
PIPELINE_MAP = {
    "INT_TRBD": "RCS_6.AI.착수정_TB",
    "INT_WTMP": "RCS_6.AI.착수정_온도",
    "INT_PH": "RCS_6.AI.착수정_PH",
    "INT_PAC": "PAC.AI.PAC_주입율제어_SV",
    "SED_TRBD1": "RCS_6.AI.침전지1_탁도",
    "SED_TRBD2": "RCS_6.AI.침전지2_탁도",
    "FLT_TRBD1": "RCS_6.AI.여과지_탁도",
    "PUR_TRBD1": "RCS_6.AI.정수지_탁도",
    "WTK_MMNT_FL": "RCS_1.AI.FT101",
}
# parquet RAW5 중 DB에 없는 원수성상 (파이프라인 사용 시 대체·보간 필요)
MISSING_RAW = ["RCS_6.AI.착수정_AL", "RCS_6.AI.착수정_전기전도도"]
