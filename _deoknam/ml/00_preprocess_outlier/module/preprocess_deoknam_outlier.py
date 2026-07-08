from pathlib import Path

from outlier_preprocess_common import run_outlier_pipeline


def main():
    # 현재 스크립트 위치:
    # _deoknam/ml/00_preprocess_outlier
    script_dir = Path(__file__).resolve().parent

    # 작업 폴더:
    # _deoknam/ml/00_preprocess_outlier
    work_dir = script_dir.parent

    # 프로젝트 루트: EPSEnE_water_treatment
    project_root = script_dir.parents[3]

    # 원본 parquet 위치: EPSEnE_water_treatment/dataset
    dataset_dir = project_root / "dataset"

    # 출력 위치:
    # _deoknam/ml/00_preprocess_outlier/output/...
    output_root = work_dir / "output"
    output_dataset_dir = output_root / "dataset"
    output_report_dir = output_root / "report"

    summary = run_outlier_pipeline(
        input_path=dataset_dir / "덕남_응집제공정_소독공정_통합.parquet",
        output_path=output_dataset_dir / "덕남_응집제공정_소독공정_통합_이상치제거.parquet",
        report_csv_path=output_report_dir / "덕남_이상치제거_리포트.csv",
        report_md_path=output_report_dir / "덕남_전처리_리포트.md",
        plant_name="덕남",
    )

    save_status = "저장 완료" if summary.get("저장 여부", True) else "저장 생략"
    print(f"덕남 이상치 제거 처리 완료 ({save_status})")
    print(f"원본 행 수: {summary['원본 행 수']:,}")
    print(f"통계 기반 제거 행 수: {summary['통계 기반 제거 행 수']:,}")
    print(f"AI 기반 제거 행 수: {summary['AI 기반 제거 행 수']:,}")
    print(f"전체 제거 행 수: {summary['전체 제거 행 수']:,}")
    print(f"최종 행 수: {summary['최종 행 수']:,}")
    print(f"제거율: {summary['제거율(%)']:.4f}%")
    print(f"실제 탐지 대상 컬럼 수: {summary['실제 탐지 대상 컬럼 수']:,}")
    print(f"탐지 제외 컬럼 수: {summary['탐지 제외 컬럼 수']:,}")


if __name__ == "__main__":
    main()