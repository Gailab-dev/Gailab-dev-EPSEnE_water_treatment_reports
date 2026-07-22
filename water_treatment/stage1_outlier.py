"""Stage-1 rule-based spike detection and deterministic Kalman correction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RANDOM_STATE = 42
BASELINE_WINDOW = 3
EXTREME_RATIO = 5.0
BASELINE_RETURN_RATIO = 2.0
MAX_CORRECTED_RUN = 5
PROCESS_VARIANCE_SCALE = 1e-4
MEASUREMENT_VARIANCE_SCALE = 1.0
VARIANCE_FLOOR = 1e-9
LOG_COLUMNS = [
    "plant",
    "stage",
    "rule_name",
    "datetime",
    "column",
    "original_value",
    "corrected_value",
    "method",
    "run_length",
    "action",
    "before_row_count",
    "after_row_count",
    "removed_row_count",
    "removed_rate",
]
REQUIRED_FEATURE_KEYS = (
    "turbidity",
    "ph",
    "alkalinity",
    "temperature",
    "conductivity",
)


@dataclass(frozen=True)
class SpikeRun:
    """Half-open positional interval for a return-to-baseline spike."""

    start: int
    stop: int
    rule_name: str

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FeatureResult:
    """Corrected feature values and the positions affected by each action."""

    values: np.ndarray
    corrected_positions: np.ndarray
    flagged_positions: np.ndarray
    runs: tuple[SpikeRun, ...]


@dataclass(frozen=True)
class PlantConfig:
    plant_key: str
    plant_name: str
    prefix: str
    input_path: Path
    datetime_column: str
    raw_water_features: dict[str, str]
    output_dataset: Path
    output_report: Path
    output_log: Path


@dataclass(frozen=True)
class RunSummary:
    plant: str
    row_count: int
    corrected_count: int
    flagged_count: int
    removed_count: int
    correction_rate: float
    removal_rate: float
    guard: str
    dry_run: bool
    feature_counts: dict[str, dict[str, int]]
    rule_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plant": self.plant,
            "row_count": self.row_count,
            "corrected_count": self.corrected_count,
            "flagged_count": self.flagged_count,
            "removed_count": self.removed_count,
            "correction_rate": self.correction_rate,
            "removal_rate": self.removal_rate,
            "guard": self.guard,
            "dry_run": self.dry_run,
            "feature_counts": self.feature_counts,
            "rule_counts": self.rule_counts,
        }


class GuardBlockedError(RuntimeError):
    """Raised before output creation when the 15% save guard is exceeded."""


def _valid_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != 0.0)


def _rolling_previous_median(
    values: np.ndarray, valid: np.ndarray, window: int
) -> np.ndarray:
    previous = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) <= window:
        return previous
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid, window).all(axis=1)
    medians = np.full(len(windows), np.nan, dtype=np.float64)
    medians[valid_windows] = np.median(windows[valid_windows], axis=1)
    previous[window:] = medians[:-1]
    return previous


def _extreme_direction(value: float, baseline: float, ratio: float) -> int:
    if not np.isfinite(value) or not np.isfinite(baseline) or value == 0.0 or baseline <= 0.0:
        return 0
    if value >= baseline * ratio:
        return 1
    if value <= baseline / ratio:
        return -1
    return 0


def _same_baseline_level(before: float, after: float, max_ratio: float) -> bool:
    if not np.isfinite(before) or not np.isfinite(after) or before <= 0.0 or after <= 0.0:
        return False
    return max(before, after) / min(before, after) <= max_ratio


def detect_return_spike_runs(
    values: np.ndarray,
    baseline_window: int = BASELINE_WINDOW,
    extreme_ratio: float = EXTREME_RATIO,
    baseline_return_ratio: float = BASELINE_RETURN_RATIO,
) -> list[SpikeRun]:
    """Detect abrupt extreme runs whose following baseline returns to the prior level.

    Missing, zero and infinite observations split the series into independent sensor
    availability segments. No candidate can use context across such a boundary.
    """

    values = np.asarray(values, dtype=np.float64)
    if baseline_window < 1 or extreme_ratio <= 1.0 or baseline_return_ratio < 1.0:
        raise ValueError("invalid return-spike detection parameters")
    valid = _valid_mask(values)
    previous = _rolling_previous_median(values, valid, baseline_window)
    usable_baseline = np.isfinite(previous) & (previous > 0.0)
    candidate_starts = valid & usable_baseline & (
        (values >= previous * extreme_ratio) | (values <= previous / extreme_ratio)
    )

    runs: list[SpikeRun] = []
    position = baseline_window
    latest_start = len(values) - baseline_window
    while position < latest_start:
        if not candidate_starts[position]:
            position += 1
            continue
        before = previous[position]
        direction = _extreme_direction(values[position], before, extreme_ratio)
        stop = position
        while (
            stop < len(values)
            and valid[stop]
            and _extreme_direction(values[stop], before, extreme_ratio) == direction
        ):
            stop += 1
        if stop + baseline_window > len(values) or not valid[stop : stop + baseline_window].all():
            position += 1
            continue
        after = float(np.median(values[stop : stop + baseline_window]))
        returns = _same_baseline_level(before, after, baseline_return_ratio)
        extreme_to_after = all(
            _extreme_direction(value, after, extreme_ratio) == direction
            for value in values[position:stop]
        )
        if returns and extreme_to_after:
            rule_name = "single_return_spike" if stop - position == 1 else "sustained_return_spike"
            runs.append(SpikeRun(position, stop, rule_name))
            position = stop + baseline_window
        else:
            position += 1
    return runs


def _robust_difference_variance(values: np.ndarray) -> float:
    differences = np.diff(values)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return VARIANCE_FLOOR
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    return max((1.4826 * mad) ** 2, VARIANCE_FLOOR)


def _kalman_smooth_segment(
    original: np.ndarray,
    missing: np.ndarray,
    process_variance_scale: float,
    measurement_variance_scale: float,
    variance_floor: float,
) -> np.ndarray:
    base_variance = _robust_difference_variance(original)
    process_variance = max(base_variance * process_variance_scale, variance_floor)
    measurement_variance = max(base_variance * measurement_variance_scale, variance_floor)
    size = len(original)
    filtered_state = np.empty(size, dtype=np.float64)
    filtered_variance = np.empty(size, dtype=np.float64)

    first_observed = int(np.flatnonzero(~missing)[0])
    state = float(original[first_observed])
    variance = measurement_variance
    for index in range(size):
        predicted_state = state
        predicted_variance = variance + process_variance
        if missing[index]:
            state = predicted_state
            variance = predicted_variance
        else:
            gain = predicted_variance / (predicted_variance + measurement_variance)
            state = predicted_state + gain * (float(original[index]) - predicted_state)
            variance = (1.0 - gain) * predicted_variance
        filtered_state[index] = state
        filtered_variance[index] = variance

    smoothed = filtered_state.copy()
    for index in range(size - 2, -1, -1):
        predicted_variance = filtered_variance[index] + process_variance
        gain = filtered_variance[index] / predicted_variance
        smoothed[index] = filtered_state[index] + gain * (
            smoothed[index + 1] - filtered_state[index]
        )
    return smoothed


def correct_feature(
    values: np.ndarray,
    runs: list[SpikeRun],
    max_corrected_run: int = MAX_CORRECTED_RUN,
    process_variance_scale: float = PROCESS_VARIANCE_SCALE,
    measurement_variance_scale: float = MEASUREMENT_VARIANCE_SCALE,
    variance_floor: float = VARIANCE_FLOOR,
) -> FeatureResult:
    """Replace only short detected runs with local-level RTS-smoothed states."""

    original = np.asarray(values, dtype=np.float64)
    corrected = original.copy()
    valid = _valid_mask(original)
    correction_mask = np.zeros(len(original), dtype=bool)
    flagged_mask = np.zeros(len(original), dtype=bool)
    for run in runs:
        target = correction_mask if run.length <= max_corrected_run else flagged_mask
        target[run.start : run.stop] = True

    invalid_positions = np.flatnonzero(~valid)
    handled_segments: set[tuple[int, int]] = set()
    for run in runs:
        if run.length > max_corrected_run:
            continue
        left_invalid = invalid_positions[invalid_positions < run.start]
        right_invalid = invalid_positions[invalid_positions >= run.stop]
        start = int(left_invalid[-1] + 1) if len(left_invalid) else 0
        stop = int(right_invalid[0]) if len(right_invalid) else len(original)
        segment_key = (start, stop)
        if segment_key in handled_segments:
            continue
        handled_segments.add(segment_key)
        local_missing = correction_mask[start:stop]
        smoothed = _kalman_smooth_segment(
            original[start:stop],
            local_missing,
            process_variance_scale,
            measurement_variance_scale,
            variance_floor,
        )
        local_positions = np.flatnonzero(local_missing) + start
        corrected[local_positions] = smoothed[local_positions - start]

    return FeatureResult(
        values=corrected,
        corrected_positions=np.flatnonzero(correction_mask),
        flagged_positions=np.flatnonzero(flagged_mask),
        runs=tuple(runs),
    )


def guard_status(rate: float) -> str:
    """Return the contract guard state for a fractional correction/removal rate."""

    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("rate must be a non-negative finite fraction")
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


def load_plant_config(config_path: Path) -> PlantConfig:
    """Load and validate a plant mapping without touching output paths."""

    config_path = Path(config_path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "plant_key",
        "plant_name",
        "prefix",
        "input_path",
        "datetime_column",
        "raw_water_features",
        "outputs",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"config missing keys: {', '.join(missing)}")
    features = data["raw_water_features"]
    if tuple(features.keys()) != REQUIRED_FEATURE_KEYS:
        raise ValueError(
            "raw_water_features must contain, in order: " + ", ".join(REQUIRED_FEATURE_KEYS)
        )
    if len(set(features.values())) != len(REQUIRED_FEATURE_KEYS):
        raise ValueError("raw_water_features columns must be unique")
    outputs = data["outputs"]
    if set(outputs) != {"dataset", "report", "log"}:
        raise ValueError("outputs must contain exactly dataset, report and log")
    return PlantConfig(
        plant_key=str(data["plant_key"]),
        plant_name=str(data["plant_name"]),
        prefix=str(data["prefix"]),
        input_path=_resolve_path(data["input_path"], config_path),
        datetime_column=str(data["datetime_column"]),
        raw_water_features={str(key): str(value) for key, value in features.items()},
        output_dataset=_resolve_path(outputs["dataset"], config_path),
        output_report=_resolve_path(outputs["report"], config_path),
        output_log=_resolve_path(outputs["log"], config_path),
    )


def _read_and_validate_input(
    config: PlantConfig,
) -> tuple[pq.ParquetFile, np.ndarray, dict[str, np.ndarray]]:
    if not config.input_path.is_file():
        raise FileNotFoundError(f"input parquet not found: {config.input_path}")
    parquet = pq.ParquetFile(config.input_path)
    schema = parquet.schema_arrow
    required_columns = [config.datetime_column, *config.raw_water_features.values()]
    missing = [column for column in required_columns if column not in schema.names]
    if missing:
        raise ValueError(f"input missing required columns: {', '.join(missing)}")
    non_numeric = [
        column
        for column in config.raw_water_features.values()
        if not (pa.types.is_floating(schema.field(column).type) or pa.types.is_integer(schema.field(column).type))
    ]
    if non_numeric:
        raise TypeError(f"raw-water features must be numeric: {', '.join(non_numeric)}")

    selected = pq.read_table(config.input_path, columns=required_columns)
    datetimes = selected[config.datetime_column].combine_chunks().to_numpy(zero_copy_only=False)
    if len(datetimes) != parquet.metadata.num_rows:
        raise ValueError("datetime length does not match parquet row count")
    if len(datetimes) > 1 and np.any(datetimes[1:] <= datetimes[:-1]):
        raise ValueError("datetime must be strictly increasing and unique")
    values = {
        column: selected[column].combine_chunks().to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False
        )
        for column in config.raw_water_features.values()
    }
    return parquet, datetimes, values


def _iso_datetime(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="auto")


def _build_log_rows(
    config: PlantConfig,
    datetimes: np.ndarray,
    original_values: dict[str, np.ndarray],
    results: dict[str, FeatureResult],
    row_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, result in results.items():
        original = original_values[column]
        for run in result.runs:
            corrected = run.length <= MAX_CORRECTED_RUN
            for position in range(run.start, run.stop):
                rows.append(
                    {
                        "plant": config.plant_name,
                        "stage": "stage1",
                        "rule_name": run.rule_name,
                        "datetime": _iso_datetime(datetimes[position]),
                        "column": column,
                        "original_value": float(original[position]),
                        "corrected_value": (
                            float(result.values[position]) if corrected else ""
                        ),
                        "method": (
                            "kalman_local_level_rts" if corrected else "run_limit_exceeded"
                        ),
                        "run_length": run.length,
                        "action": "corrected" if corrected else "flagged",
                        "before_row_count": row_count,
                        "after_row_count": row_count,
                        "removed_row_count": 0,
                        "removed_rate": 0.0,
                    }
                )
    return rows


def _summarize(
    config: PlantConfig,
    row_count: int,
    results: dict[str, FeatureResult],
    log_rows: list[dict[str, Any]],
    dry_run: bool,
) -> RunSummary:
    corrected_count = sum(len(result.corrected_positions) for result in results.values())
    flagged_count = sum(len(result.flagged_positions) for result in results.values())
    denominator = row_count * len(config.raw_water_features)
    correction_rate = corrected_count / denominator if denominator else 0.0
    feature_counts = {
        column: {
            "corrected": int(len(result.corrected_positions)),
            "flagged": int(len(result.flagged_positions)),
        }
        for column, result in results.items()
    }
    rule_counts = dict(Counter(row["rule_name"] for row in log_rows))
    return RunSummary(
        plant=config.plant_name,
        row_count=row_count,
        corrected_count=corrected_count,
        flagged_count=flagged_count,
        removed_count=0,
        correction_rate=correction_rate,
        removal_rate=0.0,
        guard=guard_status(correction_rate),
        dry_run=dry_run,
        feature_counts=feature_counts,
        rule_counts=rule_counts,
    )


def _report_text(config: PlantConfig, summary: RunSummary) -> str:
    guard_labels = {
        "ok": "정상 (5% 이하)",
        "warning": "경고 (5% 초과)",
        "review_required": "사용자 검토 필요 (10% 초과)",
        "blocked": "저장 차단 (15% 초과)",
    }
    lines = [
        f"# {config.plant_name} 1차 이상치처리 리포트",
        "",
        "## 실행 요약",
        "",
        f"- 전체 행 수: {summary.row_count:,}",
        f"- 원수 피처 수: {len(config.raw_water_features)}",
        f"- 보정 건수: {summary.corrected_count:,}",
        f"- 보정율(보정 셀 / 전체 행×5피처): {summary.correction_rate:.6%}",
        f"- 장기 run flag 건수: {summary.flagged_count:,}",
        "- 폴백 삭제 건수: 0 (행 보존)",
        "- 폴백 삭제율: 0.000000%",
        f"- 가드 판정: {guard_labels[summary.guard]}",
        "",
        "## 탐지·보정 규칙",
        "",
        "- 직전/직후 3개 유효 관측 중앙값이 2배 이내로 복귀하는 run만 탐지",
        "- 양쪽 기준선 대비 5배 이상 또는 1/5 이하의 같은 방향 극단 run",
        f"- Kalman 보정 run 상한: {MAX_CORRECTED_RUN}행(1분 데이터 기준 5분)",
        "- 6행 이상 run은 값과 행을 유지하고 correction log에 flagged 기록",
        "- 결측·0·±inf는 탐지/보정하지 않으며 세그먼트 경계로 사용",
        "",
        "## Kalman 재현 파라미터",
        "",
        f"- random_state: {RANDOM_STATE} (결정론적 알고리즘의 실행 표준값)",
        f"- process_variance_scale: {PROCESS_VARIANCE_SCALE}",
        f"- measurement_variance_scale: {MEASUREMENT_VARIANCE_SCALE}",
        f"- variance_floor: {VARIANCE_FLOOR}",
        "- method: 1차원 local-level/random-walk Kalman filter + RTS smoother",
        "",
        "## 컬럼별 처리 건수",
        "",
        "| 의미 | 컬럼 | 보정 | flagged |",
        "|---|---|---:|---:|",
    ]
    reverse_features = {value: key for key, value in config.raw_water_features.items()}
    for column, counts in summary.feature_counts.items():
        lines.append(
            f"| {reverse_features[column]} | `{column}` | {counts['corrected']:,} | {counts['flagged']:,} |"
        )
    lines.extend(
        [
            "",
            "## 규칙별 탐지 건수",
            "",
            "| 규칙 | 셀 수 |",
            "|---|---:|",
        ]
    )
    for rule in ("single_return_spike", "sustained_return_spike"):
        lines.append(f"| {rule} | {summary.rule_counts.get(rule, 0):,} |")
    lines.extend(
        [
            "",
            "## 과도 보정·삭제 가드",
            "",
            "보정율과 삭제율 모두 5% 초과 시 경고, 10% 초과 시 사용자 검토 필요, "
            "15% 초과 시 명시 승인 전 저장 차단을 적용한다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_parquet(
    parquet: pq.ParquetFile,
    destination: Path,
    results: dict[str, FeatureResult],
) -> None:
    offset = 0
    with pq.ParquetWriter(destination, parquet.schema_arrow, compression="snappy") as writer:
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group)
            size = table.num_rows
            for column, result in results.items():
                index = table.schema.get_field_index(column)
                field = table.schema.field(index)
                replacement = pa.array(
                    result.values[offset : offset + size], type=field.type, from_pandas=True
                )
                table = table.set_column(index, field, replacement)
            writer.write_table(table, row_group_size=size)
            offset += size


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_temp_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    os.close(handle)
    return Path(name)


def _write_outputs(
    config: PlantConfig,
    parquet: pq.ParquetFile,
    results: dict[str, FeatureResult],
    log_rows: list[dict[str, Any]],
    report: str,
) -> None:
    finals = [config.output_dataset, config.output_report, config.output_log]
    temporary: list[Path] = []
    try:
        temporary = [_prepare_temp_path(path) for path in finals]
        _write_parquet(parquet, temporary[0], results)
        temporary[1].write_text(report, encoding="utf-8")
        _write_csv(temporary[2], log_rows)
        for source, destination in zip(temporary, finals):
            source.replace(destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def run_stage1(config_path: Path, dry_run: bool = False) -> RunSummary:
    """Execute stage1 analysis and optionally create the three contract outputs."""

    config = load_plant_config(config_path)
    parquet, datetimes, original_values = _read_and_validate_input(config)
    results: dict[str, FeatureResult] = {}
    for column, values in original_values.items():
        runs = detect_return_spike_runs(values)
        results[column] = correct_feature(values, runs)
    row_count = parquet.metadata.num_rows
    log_rows = _build_log_rows(
        config, datetimes, original_values, results, row_count
    )
    summary = _summarize(config, row_count, results, log_rows, dry_run)
    if not dry_run:
        if summary.guard == "blocked":
            raise GuardBlockedError(
                f"{config.plant_name} correction rate {summary.correction_rate:.6%} exceeds 15%"
            )
        _write_outputs(config, parquet, results, log_rows, _report_text(config, summary))
    return summary


def cli(config_path: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="analyze without writing")
    arguments = parser.parse_args()
    summary = run_stage1(config_path, dry_run=arguments.dry_run)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 2 if summary.guard == "blocked" else 0
