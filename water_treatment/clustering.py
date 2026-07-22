# -*- coding: utf-8 -*-
"""Contract-driven raw-water clustering for Deoknam and Yongyeon."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "cluster_label"
FIXED_INTERVAL = "10min"
KMEANS_K = 3
RANDOM_STATE = 42


@dataclass(frozen=True)
class OutputPaths:
    labeled_dataset: Path
    cluster_datasets: tuple[Path, Path, Path]
    model: Path
    report: Path


@dataclass(frozen=True)
class PlantConfig:
    key: str
    name: str
    tree: str
    prefix: str
    datetime_col: str
    raw_water_features: dict[str, str]


def _first_valid(series: pd.Series) -> object:
    nonnull = series.dropna()
    return nonnull.iloc[0] if not nonnull.empty else pd.NA


def aggregate_fixed_windows(
    frame: pd.DataFrame,
    datetime_col: str,
    feature_map: Mapping[str, str],
    interval: str = FIXED_INTERVAL,
) -> pd.DataFrame:
    """Aggregate independent, clock-aligned windows without rolling/backfill."""
    required = [datetime_col, *feature_map.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))
    if LABEL_COLUMN in frame.columns:
        raise ValueError(f"input already contains reserved column: {LABEL_COLUMN}")

    working = frame.copy()
    working[datetime_col] = pd.to_datetime(working[datetime_col], errors="coerce")
    invalid_timestamps = int(working[datetime_col].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"datetime contains {invalid_timestamps} invalid value(s)")
    working = working.sort_values(datetime_col, kind="stable")
    window_col = "__fixed_window_start__"
    working[window_col] = working[datetime_col].dt.floor(interval)

    feature_columns = set(feature_map.values())
    for column in feature_columns:
        numeric = pd.to_numeric(working[column], errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf, 0.0], np.nan)
        working[column] = numeric

    grouped = working.groupby(window_col, sort=True, dropna=False)
    result = pd.DataFrame(index=grouped.size().index)
    for column in frame.columns:
        if column == datetime_col:
            result[column] = result.index
        elif column in feature_columns:
            result[column] = grouped[column].mean()
        elif column.endswith("_flag") or column.endswith("_stage1_flag"):
            result[column] = grouped[column].max()
        elif column.endswith("_src"):
            result[column] = grouped[column].agg(_first_valid)
        elif pd.api.types.is_numeric_dtype(working[column].dtype):
            numeric = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            working[column] = numeric
            result[column] = working.groupby(window_col, sort=True, dropna=False)[column].mean()
        else:
            result[column] = grouped[column].agg(_first_valid)

    result = result.reset_index(drop=True)
    return result.loc[:, list(frame.columns)].sort_values(datetime_col, kind="stable").reset_index(drop=True)


def fit_and_assign_labels(
    aggregated: pd.DataFrame,
    feature_map: Mapping[str, str],
    k: int = KMEANS_K,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame, pd.Series]:
    """Fit the fixed k=3 pipeline and retain invalid windows with null labels."""
    if k != KMEANS_K:
        raise ValueError("k must be fixed at 3")
    columns = list(feature_map.values())
    missing = [column for column in columns if column not in aggregated.columns]
    if missing:
        raise ValueError("missing clustering features: %s" % ", ".join(missing))

    features = aggregated.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    array = features.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = pd.Series(
        np.isfinite(array).all(axis=1) & (array != 0.0).all(axis=1),
        index=aggregated.index,
        name="valid_for_training",
    )
    valid_features = features.loc[valid_mask].astype(float)
    if len(valid_features) < KMEANS_K:
        raise ValueError(
            f"at least {KMEANS_K} valid 10-minute windows are required; found {len(valid_features)}"
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=KMEANS_K, random_state=random_state, n_init=10),
            ),
        ]
    )
    labels = model.fit_predict(valid_features)
    labeled = aggregated.copy()
    labeled[LABEL_COLUMN] = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    labeled.loc[valid_mask, LABEL_COLUMN] = labels.astype(int)

    centers = model.named_steps["scaler"].inverse_transform(
        model.named_steps["kmeans"].cluster_centers_
    )
    centers_frame = pd.DataFrame(centers, columns=columns)
    centers_frame.insert(0, LABEL_COLUMN, range(KMEANS_K))
    return labeled, model, centers_frame, valid_mask


def output_paths(output_root: Path, prefix: str) -> OutputPaths:
    dataset_dir = output_root / "dataset"
    return OutputPaths(
        labeled_dataset=(
            dataset_dir
            / f"{prefix}_응집제공정_소독공정_통합_2차이상치처리_10m_군집라벨.parquet"
        ),
        cluster_datasets=tuple(dataset_dir / f"{prefix}_cluster{i}.parquet" for i in range(3)),
        model=output_root / "model" / f"{prefix}_kmeans_k3_model.joblib",
        report=output_root / "report" / f"{prefix}_군집분류_리포트.md",
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    labeled: pd.DataFrame,
    centers: pd.DataFrame,
    valid_mask: pd.Series,
    plant_name: str,
    feature_map: Mapping[str, str],
    source_rows: int,
    source_path: Path,
) -> str:
    valid_count = int(valid_mask.sum())
    distributions = []
    for cluster in range(KMEANS_K):
        count = int((labeled[LABEL_COLUMN] == cluster).sum())
        ratio = count / valid_count if valid_count else 0.0
        distributions.append((cluster, count, f"{ratio:.2%}"))
    center_rows = []
    for _, row in centers.iterrows():
        center_rows.append(
            [int(row[LABEL_COLUMN]), *[f"{float(row[column]):.6g}" for column in feature_map.values()]]
        )

    return "\n".join(
        [
            f"# {plant_name} 원수 수질 군집분류 리포트",
            "",
            "## 채택 근거",
            "",
            "원수 수질 상태를 재현성 있게 구분하고 중심점으로 설명할 수 있도록 "
            "KMeans를 채택했으며, 계약에 따라 k=3으로 고정했다. 단위 차이의 영향을 줄이기 위해 "
            "StandardScaler를 적용한 뒤 KMeans를 학습했다. 라벨 0/1/2는 수질 상태 ID일 뿐 "
            "특정 탁도 등급을 뜻하지 않는다.",
            "",
            "참고자료: `02_AI_모델선정서_v0.4_오현진_260626.pdf`, "
            "`09_데이터분석보고서_v0.2_오현진_260708.pdf`, "
            "`계측 시계열 와이드 테이블_0708.txt`, "
            "`계측_컬럼_매핑_덕남_용연_0709.html`.",
            "",
            "## 10분 고정 구간 집계",
            "",
            "10분 집계의 목적은 계측 노이즈를 완화해 **원수 상태 안정화**를 하는 것이며, "
            "**체류시간 반영이 아니다**. 시각을 10분 경계로 내림한 고정 구간만 사용했고 "
            "rolling 또는 다음 구간 관측값 차용은 하지 않았다.",
            "",
            "원수 5피처는 결측·0·무한대 셀을 제외한 구간 평균, 일반 수치 컬럼은 구간 평균, "
            "품질 flag는 구간 최대, source 및 비수치 컬럼은 시간순 첫 유효값으로 집계했다. "
            "학습 불가 구간도 삭제하지 않고 `cluster_label`을 null로 유지했다.",
            "",
            "## 데이터 요약",
            "",
            f"- 읽기 전용 입력: `{source_path.as_posix()}`",
            f"- 입력 행 수: {source_rows:,}",
            f"- 10분 고정 구간 수: {len(labeled):,}",
            f"- 학습·라벨 가능 구간 수: {valid_count:,}",
            f"- 학습 제외 후 유지한 구간 수: {len(labeled) - valid_count:,}",
            "",
            "## 군집 분포",
            "",
            _markdown_table(["상태 ID", "구간 수", "유효 구간 비율"], distributions),
            "",
            "## 군집 중심(원 단위)",
            "",
            _markdown_table(["상태 ID", *feature_map.values()], center_rows),
            "",
            "군집 중심과 분포는 상태 기술을 위한 값이며 운영 등급명으로 단정하지 않는다.",
            "",
        ]
    )


def save_outputs(
    *,
    labeled: pd.DataFrame,
    model: Pipeline,
    centers: pd.DataFrame,
    valid_mask: pd.Series,
    output_root: Path,
    prefix: str,
    plant_name: str,
    feature_map: Mapping[str, str],
    source_rows: int,
    source_path: Path,
) -> OutputPaths:
    paths = output_paths(output_root, prefix)
    for directory in {paths.labeled_dataset.parent, paths.model.parent, paths.report.parent}:
        directory.mkdir(parents=True, exist_ok=True)

    labeled.to_parquet(paths.labeled_dataset, index=False, compression="zstd")
    for cluster, path in enumerate(paths.cluster_datasets):
        labeled.loc[labeled[LABEL_COLUMN] == cluster].to_parquet(
            path, index=False, compression="zstd"
        )
    joblib.dump(model, paths.model)
    paths.report.write_text(
        build_report(
            labeled,
            centers,
            valid_mask,
            plant_name,
            feature_map,
            source_rows,
            source_path,
        ),
        encoding="utf-8",
    )
    return paths


PLANT_REGISTRY: dict[str, dict[str, object]] = {
    "deoknam": {
        "name": "덕남",
        "tree": "_deoknam",
        "prefix": "덕남",
        "datetime_col": "datetime",
        "raw_water_features": {
            "turbidity": "RCS_6.AI.착수정_TB",
            "ph": "RCS_6.AI.착수정_PH",
            "alkalinity": "RCS_6.AI.착수정_AL",
            "temperature": "RCS_6.AI.착수정_온도",
            "conductivity": "RCS_6.AI.착수정_전기전도도",
        },
    },
    "yongyeon": {
        "name": "용연",
        "tree": "_yongyeon",
        "prefix": "용연",
        "datetime_col": "datetime",
        "raw_water_features": {
            "turbidity": "원수 탁도",
            "ph": "원수 PH",
            "alkalinity": "원수 알카리도",
            "temperature": "원수 온도",
            "conductivity": "원수 전기전도도",
        },
    },
}


def load_plant_config(plant_key: str) -> PlantConfig:
    """Return the self-contained plant mapping (no external registry file)."""
    plant = PLANT_REGISTRY.get(plant_key)
    if plant is None:
        raise ValueError(f"unknown plant: {plant_key}")
    return PlantConfig(
        key=plant_key,
        name=str(plant["name"]),
        tree=str(plant["tree"]),
        prefix=str(plant["prefix"]),
        datetime_col=str(plant["datetime_col"]),
        raw_water_features=dict(plant["raw_water_features"]),  # type: ignore[arg-type]
    )


def input_path_for(repo_root: Path, config: PlantConfig) -> Path:
    return (
        repo_root
        / config.tree
        / "ml"
        / "00_preprocess_outlier"
        / "output"
        / "dataset"
        / f"{config.prefix}_응집제공정_소독공정_통합_1차이상치처리.parquet"
    )


def validate_input(path: Path, config: PlantConfig) -> tuple[int, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"contract input does not exist: {path}")
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    required = [config.datetime_col, *config.raw_water_features.values()]
    missing = [column for column in required if column not in names]
    if missing:
        raise ValueError("input schema is missing: %s" % ", ".join(missing))
    return parquet.metadata.num_rows, names


def read_input_frame(path: Path, datetime_col: str) -> pd.DataFrame:
    """Read a stage-1 parquet while preserving its datetime as a data column."""
    frame = pd.read_parquet(path)
    if datetime_col not in frame.columns and frame.index.name == datetime_col:
        frame = frame.reset_index()
    return frame


def run_for_plant(repo_root: Path, plant_key: str, k: int, dry_run: bool) -> OutputPaths | None:
    if k != KMEANS_K:
        raise ValueError("k must be fixed at 3")
    config = load_plant_config(plant_key)
    source_path = input_path_for(repo_root, config)
    source_rows, columns = validate_input(source_path, config)
    destination = repo_root / config.tree / "ml" / "01_clustering" / "output"
    paths = output_paths(destination, config.prefix)

    print(f"plant={config.key} ({config.name})")
    print(f"input={source_path.relative_to(repo_root)}")
    print(f"rows={source_rows:,}, columns={len(columns)}")
    print(f"features={list(config.raw_water_features.values())}")
    print(f"aggregation={FIXED_INTERVAL} fixed windows, rolling/backfill disabled")
    print(f"model=StandardScaler + KMeans(k=3, random_state={RANDOM_STATE}, n_init=10)")
    if dry_run:
        print("dry-run=OK (no files written)")
        for path in [paths.labeled_dataset, *paths.cluster_datasets, paths.model, paths.report]:
            print(f"planned={path.relative_to(repo_root)}")
        return None

    frame = read_input_frame(source_path, config.datetime_col)
    aggregated = aggregate_fixed_windows(
        frame, config.datetime_col, config.raw_water_features, interval=FIXED_INTERVAL
    )
    labeled, model, centers, valid_mask = fit_and_assign_labels(
        aggregated, config.raw_water_features, k=k, random_state=RANDOM_STATE
    )
    written = save_outputs(
        labeled=labeled,
        model=model,
        centers=centers,
        valid_mask=valid_mask,
        output_root=destination,
        prefix=config.prefix,
        plant_name=config.name,
        feature_map=config.raw_water_features,
        source_rows=source_rows,
        source_path=source_path.relative_to(repo_root),
    )
    print(f"10m_windows={len(labeled):,}, valid={int(valid_mask.sum()):,}")
    for cluster in range(KMEANS_K):
        count = int((labeled[LABEL_COLUMN] == cluster).sum())
        print(f"cluster{cluster}={count:,} ({count / int(valid_mask.sum()):.2%})")
    for path in [written.labeled_dataset, *written.cluster_datasets, written.model, written.report]:
        print(f"written={path.relative_to(repo_root)}")
    return written


def main(argv: Sequence[str] | None = None, default_plant: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train fixed-window raw-water KMeans clustering.")
    parser.add_argument("--plant", choices=("deoknam", "yongyeon"), default=default_plant, required=default_plant is None)
    parser.add_argument("--k", type=int, default=KMEANS_K)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        run_for_plant(repo_root, args.plant, args.k, args.dry_run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
