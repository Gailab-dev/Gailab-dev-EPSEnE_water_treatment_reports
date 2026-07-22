"""Cluster-local stage-2 outlier flagging with guarded sensor-failure removal."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


RANDOM_STATE = 42
IQR_MULTIPLIER = 3.0
ROBUST_Z_THRESHOLD = 5.0
IF_CONTAMINATION = 0.0025
IF_MIN_SAMPLES = 20
IF_MAX_SAMPLES = 256
LOCAL_BASELINE_RATIO = 1.5
LOCAL_EXTREME_RATIO = 8.0
MAX_LOCAL_GAP = pd.Timedelta("30min")
REQUIRED_FEATURE_KEYS = ("turbidity", "ph", "alkalinity", "temperature", "conductivity")
IQR_BIT = 1
ZSCORE_BIT = 2
IF_BIT = 4
REMOVAL_LOG_COLUMNS = [
    "plant", "stage", "rule_name", "datetime", "column", "original_value", "reason",
    "before_row_count", "after_row_count", "removed_row_count", "removed_rate",
]


class GuardBlockedError(RuntimeError):
    """Raised before writes when the 15% deletion guard would be exceeded."""


@dataclass(frozen=True)
class Stage2Config:
    plant_key: str
    plant_name: str
    prefix: str
    datetime_column: str
    raw_water_features: dict[str, str]
    input_template: Path
    output_template: Path
    report_template: Path
    log_path: Path


@dataclass(frozen=True)
class ClusterResult:
    frame: pd.DataFrame
    removal_mask: np.ndarray
    method_counts: dict[str, dict[str, int]]
    if_row_count: int

    @property
    def removed_count(self) -> int:
        return int(self.removal_mask.sum())


@dataclass(frozen=True)
class ClusterSummary:
    cluster: int
    input_rows: int
    output_rows: int
    removed_count: int
    removed_rate: float
    method_counts: dict[str, dict[str, int]]
    if_row_count: int
    guard: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "removed_count": self.removed_count,
            "removed_rate": self.removed_rate,
            "method_counts": self.method_counts,
            "if_row_count": self.if_row_count,
            "guard": self.guard,
        }


@dataclass(frozen=True)
class RunSummary:
    plant: str
    dry_run: bool
    cluster_summaries: dict[int, ClusterSummary]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plant": self.plant,
            "dry_run": self.dry_run,
            "clusters": {str(key): value.as_dict() for key, value in self.cluster_summaries.items()},
        }


def guard_status(rate: float) -> str:
    if rate > 0.15:
        return "blocked"
    if rate > 0.10:
        return "review_required"
    if rate > 0.05:
        return "warning"
    return "ok"


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    current = config_path.resolve().parent
    for candidate in (Path.cwd().resolve(), current, *current.parents):
        if (candidate / "dataset").is_dir() and (candidate / "water_treatment").is_dir():
            return candidate / path
    return current / path


def load_stage2_config(config_path: Path) -> Stage2Config:
    config_path = Path(config_path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "plant_key", "plant_name", "prefix", "datetime_column", "raw_water_features",
        "input_template", "output_template", "report_template", "log_path",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"config missing keys: {', '.join(missing)}")
    features = data["raw_water_features"]
    if tuple(features.keys()) != REQUIRED_FEATURE_KEYS:
        raise ValueError("raw_water_features must contain the five configured raw-water keys in order")
    if len(set(features.values())) != len(REQUIRED_FEATURE_KEYS):
        raise ValueError("raw_water_features values must be unique")
    return Stage2Config(
        plant_key=str(data["plant_key"]), plant_name=str(data["plant_name"]), prefix=str(data["prefix"]),
        datetime_column=str(data["datetime_column"]),
        raw_water_features={str(key): str(value) for key, value in features.items()},
        input_template=_resolve_path(str(data["input_template"]), config_path),
        output_template=_resolve_path(str(data["output_template"]), config_path),
        report_template=_resolve_path(str(data["report_template"]), config_path),
        log_path=_resolve_path(str(data["log_path"]), config_path),
    )


def _valid(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != 0.0)


def _iqr_and_robust_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid(values)
    iqr_flag = np.zeros(len(values), dtype=bool)
    z_flag = np.zeros(len(values), dtype=bool)
    if valid.sum() < 4:
        return iqr_flag, z_flag
    observed = values[valid]
    q1, q3 = np.quantile(observed, [0.25, 0.75])
    iqr = q3 - q1
    if np.isfinite(iqr) and iqr > 0.0:
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        iqr_flag[valid] = (observed < lower) | (observed > upper)
    median = np.median(observed)
    mad = np.median(np.abs(observed - median))
    if np.isfinite(mad) and mad > 0.0:
        robust_z = 0.67448975 * (observed - median) / mad
        z_flag[valid] = np.abs(robust_z) >= ROBUST_Z_THRESHOLD
    return iqr_flag, z_flag


def _if_flags(matrix: np.ndarray) -> np.ndarray:
    complete = np.all(np.isfinite(matrix) & (matrix != 0.0), axis=1)
    flags = np.zeros(len(matrix), dtype=bool)
    if int(complete.sum()) < IF_MIN_SAMPLES:
        return flags
    model = IsolationForest(
        n_estimators=100,
        contamination=IF_CONTAMINATION,
        max_samples=min(IF_MAX_SAMPLES, int(complete.sum())),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    flags[complete] = model.fit_predict(matrix[complete]) == -1
    return flags


def _local_sensor_failure(
    values: np.ndarray, statistical_flag: np.ndarray, timestamps: pd.Series
) -> np.ndarray:
    """Require a single extreme point bracketed by a nearby returned baseline."""

    removed = np.zeros(len(values), dtype=bool)
    times = pd.to_datetime(timestamps, errors="coerce")
    for position in np.flatnonzero(statistical_flag):
        if position == 0 or position == len(values) - 1:
            continue
        before, value, after = values[position - 1], values[position], values[position + 1]
        if not (np.isfinite(before) and np.isfinite(value) and np.isfinite(after)):
            continue
        if before <= 0.0 or value <= 0.0 or after <= 0.0:
            continue
        if statistical_flag[position - 1] or statistical_flag[position + 1]:
            continue
        before_time, current_time, after_time = times.iloc[position - 1 : position + 2]
        if pd.isna(before_time) or pd.isna(current_time) or pd.isna(after_time):
            continue
        if current_time - before_time > MAX_LOCAL_GAP or after_time - current_time > MAX_LOCAL_GAP:
            continue
        baseline_ratio = max(before, after) / min(before, after)
        extremity_ratio = max(value, (before + after) / 2.0) / min(value, (before + after) / 2.0)
        if baseline_ratio <= LOCAL_BASELINE_RATIO and extremity_ratio >= LOCAL_EXTREME_RATIO:
            removed[position] = True
    return removed


def analyze_cluster(
    frame: pd.DataFrame, raw_water_features: list[str], datetime_column: str
) -> ClusterResult:
    """Flag each feature by cluster distribution and identify guarded removal rows."""

    required = [datetime_column, "cluster_label", *raw_water_features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"cluster input missing required columns: {', '.join(missing)}")
    numeric = frame[raw_water_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(frame[datetime_column], errors="coerce")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    inverse_order = np.empty(len(order), dtype=int)
    inverse_order[order] = np.arange(len(order))
    sorted_values = numeric[order]
    sorted_times = timestamps.iloc[order].reset_index(drop=True)
    flags_sorted = np.zeros_like(sorted_values, dtype=np.uint8)
    method_counts: dict[str, dict[str, int]] = {}
    local_removed_sorted = np.zeros(len(frame), dtype=bool)
    for index, column in enumerate(raw_water_features):
        iqr, zscore = _iqr_and_robust_z(sorted_values[:, index])
        flags_sorted[:, index] |= np.where(iqr, IQR_BIT, 0).astype(np.uint8)
        flags_sorted[:, index] |= np.where(zscore, ZSCORE_BIT, 0).astype(np.uint8)
        statistical = iqr & zscore
        local_removed_sorted |= _local_sensor_failure(sorted_values[:, index], statistical, sorted_times)
        method_counts[column] = {
            "iqr": int(iqr.sum()), "robust_z": int(zscore.sum()), "if": 0,
            "statistical_consensus": int(statistical.sum()),
            "sensor_failure_candidate": int(_local_sensor_failure(sorted_values[:, index], statistical, sorted_times).sum()),
        }
    if_flags = _if_flags(sorted_values)
    flags_sorted[if_flags, :] |= IF_BIT
    for column in raw_water_features:
        method_counts[column]["if"] = int(if_flags.sum())
    flags = flags_sorted[inverse_order]
    removal_mask = local_removed_sorted[inverse_order]
    output = frame.copy()
    for index, column in enumerate(raw_water_features):
        output[f"{column}_stage2_flag"] = pd.Series(flags[:, index], index=output.index, dtype="uint8")
    return ClusterResult(output, removal_mask, method_counts, int(if_flags.sum()))


def _format(template: Path, cluster: int) -> Path:
    return Path(str(template).format(cluster=cluster))


def _read_cluster_metadata(path: Path, config: Stage2Config) -> tuple[int, list[str]]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"cluster input parquet not found: {path}")
    parquet = pq.ParquetFile(path)
    required = [config.datetime_column, "cluster_label", *config.raw_water_features.values()]
    missing = [column for column in required if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"input missing required columns: {', '.join(missing)}")
    return parquet.metadata.num_rows, parquet.schema_arrow.names


def _log_rows(
    config: Stage2Config, cluster: int, frame: pd.DataFrame, removal_mask: np.ndarray
) -> list[dict[str, Any]]:
    before = len(frame)
    removed = int(removal_mask.sum())
    after = before - removed
    rate = removed / before if before else 0.0
    rows: list[dict[str, Any]] = []
    features = list(config.raw_water_features.values())
    for position in np.flatnonzero(removal_mask):
        for column in features:
            flag = int(frame.iloc[position][f"{column}_stage2_flag"])
            if (flag & (IQR_BIT | ZSCORE_BIT)) != (IQR_BIT | ZSCORE_BIT):
                continue
            rows.append({
                "plant": config.plant_name, "stage": "stage2", "rule_name": "isolated_local_return_sensor_failure",
                "datetime": pd.Timestamp(frame.iloc[position][config.datetime_column]).isoformat(),
                "column": column, "original_value": frame.iloc[position][column],
                "reason": "IQR+robust_z consensus, isolated local spike, adjacent baseline returned; IF was not a deletion condition",
                "before_row_count": before, "after_row_count": after, "removed_row_count": removed, "removed_rate": rate,
            })
    return rows


def _report_text(config: Stage2Config, summary: ClusterSummary) -> str:
    labels = {"ok": "정상 (5% 이하)", "warning": "경고 (5% 초과)", "review_required": "사용자 검토 필요 (10% 초과)", "blocked": "저장 차단 (15% 초과)"}
    lines = [
        f"# {config.plant_name} cluster{summary.cluster} 2차 이상치처리 리포트", "",
        "## 실행 요약", "",
        f"- 입력 행 수: {summary.input_rows:,}", f"- 출력 행 수: {summary.output_rows:,}",
        f"- 삭제 행 수: {summary.removed_count:,} ({summary.removed_rate:.6%})",
        f"- 삭제율 가드: {labels[summary.guard]}", f"- Isolation Forest 행 플래그: {summary.if_row_count:,}", "",
        "## 군집 내 판정·결합 규칙", "",
        "- 이 파일의 `cluster_label` 내부 분포만 사용한다. 군집 간 분포를 합치지 않는다.",
        f"- IQR: Q1−{IQR_MULTIPLIER}×IQR 미만 또는 Q3+{IQR_MULTIPLIER}×IQR 초과를 flag한다.",
        f"- robust Z-score: median/MAD 기준 |Z|≥{ROBUST_Z_THRESHOLD}를 flag한다.",
        f"- Isolation Forest: 원수 5피처 다변량, contamination={IF_CONTAMINATION}, random_state={RANDOM_STATE}, n_estimators=100, max_samples≤{IF_MAX_SAMPLES}; anomaly는 flag 전용이다.",
        "- `<원본컬럼>_stage2_flag` 비트: 1=IQR, 2=robust Z-score, 4=Isolation Forest. 기존 `_flag`/`_src`는 변경하지 않는다.",
        "- 삭제는 IQR와 robust Z-score가 같은 셀에서 동시에 검출되고, 앞뒤 30분 이내 정상 기준선이 1.5배 이내로 복귀하며, 해당 값이 기준선 대비 8배 이상인 고립 센서 스파이크일 때만 수행한다. IF 단독 이상치는 절대 삭제하지 않는다.",
        "- 0, 결측, ±inf는 통계/IF 학습에서 제외하고 값과 행을 보존한다.", "",
        "## 피처별 flag·삭제 후보", "", "| 컬럼 | IQR | robust Z | IF 행 | 통계 합의 | 센서고장 후보 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for column, counts in summary.method_counts.items():
        lines.append(f"| `{column}` | {counts['iqr']:,} | {counts['robust_z']:,} | {counts['if']:,} | {counts['statistical_consensus']:,} | {counts['sensor_failure_candidate']:,} |")
    lines.extend(["", "## 저장 가드", "", "삭제율이 5% 초과면 경고, 10% 초과면 사용자 검토 필요, 15% 초과면 저장을 차단한다.", ""])
    return "\n".join(lines)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="snappy")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REMOVAL_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage2(config_path: Path, clusters: Iterable[int] = (0, 1, 2), dry_run: bool = False) -> RunSummary:
    config = load_stage2_config(config_path)
    selected_clusters = [int(cluster) for cluster in clusters]
    if sorted(set(selected_clusters)) != selected_clusters or any(cluster not in (0, 1, 2) for cluster in selected_clusters):
        raise ValueError("clusters must be a sorted subset of 0, 1, 2")
    if dry_run:
        summaries = {}
        for cluster in selected_clusters:
            rows, _ = _read_cluster_metadata(_format(config.input_template, cluster), config)
            summaries[cluster] = ClusterSummary(cluster, rows, rows, 0, 0.0, {}, 0, "ok")
        return RunSummary(config.plant_name, True, summaries)

    pending: list[tuple[int, ClusterResult, ClusterSummary, list[dict[str, Any]]]] = []
    all_log_rows: list[dict[str, Any]] = []
    for cluster in selected_clusters:
        input_path = _format(config.input_template, cluster)
        _read_cluster_metadata(input_path, config)
        frame = pd.read_parquet(input_path, engine="pyarrow")
        result = analyze_cluster(frame, list(config.raw_water_features.values()), config.datetime_column)
        rate = result.removed_count / len(frame) if len(frame) else 0.0
        summary = ClusterSummary(cluster, len(frame), len(frame) - result.removed_count, result.removed_count, rate, result.method_counts, result.if_row_count, guard_status(rate))
        if summary.guard == "blocked":
            raise GuardBlockedError(f"{config.plant_name} cluster{cluster} removal rate {rate:.6%} exceeds 15%")
        rows = _log_rows(config, cluster, result.frame, result.removal_mask)
        all_log_rows.extend(rows)
        pending.append((cluster, result, summary, rows))

    for cluster, result, summary, _ in pending:
        _atomic_write_parquet(_format(config.output_template, cluster), result.frame.loc[~result.removal_mask].copy())
        _atomic_write_text(_format(config.report_template, cluster), _report_text(config, summary))
    _atomic_write_log(config.log_path, all_log_rows)
    return RunSummary(config.plant_name, False, {summary.cluster: summary for _, _, summary, _ in pending})


def cli(config_path: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate input schemas and planned outputs without writing")
    parser.add_argument("--cluster", default="all", choices=("all", "0", "1", "2"))
    args = parser.parse_args()
    clusters = (0, 1, 2) if args.cluster == "all" else (int(args.cluster),)
    summary = run_stage2(config_path, clusters=clusters, dry_run=args.dry_run)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0
