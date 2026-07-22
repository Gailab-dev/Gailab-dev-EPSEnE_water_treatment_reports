import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from water_treatment.stage1_outlier import (
    GuardBlockedError,
    LOG_COLUMNS,
    correct_feature,
    detect_return_spike_runs,
    guard_status,
    run_stage1,
)


class ReturnSpikeDetectionTests(unittest.TestCase):
    def test_detects_single_return_spike(self):
        values = np.array([10.0, 10.1, 9.9, 100.0, 10.0, 10.1, 9.9])

        runs = detect_return_spike_runs(values)

        self.assertEqual(1, len(runs))
        self.assertEqual((3, 4), (runs[0].start, runs[0].stop))
        self.assertEqual("single_return_spike", runs[0].rule_name)

    def test_detects_short_sustained_return_spike(self):
        values = np.array([10.0, 10.1, 9.9, 100.0, 101.0, 99.0, 10.0, 10.1, 9.9])

        runs = detect_return_spike_runs(values)

        self.assertEqual([(3, 6, "sustained_return_spike")], [
            (run.start, run.stop, run.rule_name) for run in runs
        ])

    def test_detects_long_run_for_flagging(self):
        values = np.array(
            [10.0, 10.1, 9.9] + [100.0] * 6 + [10.0, 10.1, 9.9],
            dtype=float,
        )

        runs = detect_return_spike_runs(values)

        self.assertEqual(6, runs[0].length)

    def test_preserves_genuine_level_shift_without_return(self):
        values = np.array([10.0, 10.1, 9.9, 100.0, 101.0, 99.0, 100.0, 100.5, 99.5])

        self.assertEqual([], detect_return_spike_runs(values))

    def test_does_not_cross_invalid_value_boundaries(self):
        for invalid in (np.nan, 0.0, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                values = np.array([10.0, 10.1, 9.9, invalid, 100.0, 10.0, 10.1, 9.9])
                self.assertEqual([], detect_return_spike_runs(values))


class KalmanCorrectionTests(unittest.TestCase):
    def test_corrects_only_short_detected_run(self):
        values = np.array([10.0, 10.1, 9.9, 100.0, 10.0, 10.1, 9.9])
        original = values.copy()

        result = correct_feature(values, detect_return_spike_runs(values))

        self.assertEqual([3], result.corrected_positions.tolist())
        self.assertEqual([], result.flagged_positions.tolist())
        self.assertLess(result.values[3], 20.0)
        np.testing.assert_array_equal(result.values[:3], original[:3])
        np.testing.assert_array_equal(result.values[4:], original[4:])

    def test_flags_long_run_without_changing_it(self):
        values = np.array([10.0, 10.1, 9.9] + [100.0] * 6 + [10.0, 10.1, 9.9])

        result = correct_feature(values, detect_return_spike_runs(values), max_corrected_run=5)

        self.assertEqual([], result.corrected_positions.tolist())
        self.assertEqual(list(range(3, 9)), result.flagged_positions.tolist())
        np.testing.assert_array_equal(values, result.values)

    def test_invalid_values_remain_bitwise_equivalent(self):
        values = np.array([10.0, 10.1, 9.9, 100.0, 10.0, np.nan, 0.0, np.inf, -np.inf])

        result = correct_feature(values, detect_return_spike_runs(values))

        self.assertTrue(np.isnan(result.values[5]))
        self.assertEqual(0.0, result.values[6])
        self.assertEqual(np.inf, result.values[7])
        self.assertEqual(-np.inf, result.values[8])


class GuardTests(unittest.TestCase):
    def test_guard_boundaries_are_strictly_greater_than_thresholds(self):
        self.assertEqual("ok", guard_status(0.05))
        self.assertEqual("warning", guard_status(0.050001))
        self.assertEqual("warning", guard_status(0.10))
        self.assertEqual("review_required", guard_status(0.100001))
        self.assertEqual("review_required", guard_status(0.15))
        self.assertEqual("blocked", guard_status(0.150001))


class Stage1PipelineTests(unittest.TestCase):
    def _write_fixture(self, root: Path, values: list[float]) -> Path:
        index = pd.date_range("2026-01-01", periods=len(values), freq="min", name="datetime")
        frame = pd.DataFrame(
            {
                "feature_1": values,
                "feature_2": values,
                "feature_3": values,
                "feature_4": values,
                "feature_5": values,
                "untouched": np.arange(len(values), dtype=np.int64),
            },
            index=index,
        )
        input_path = root / "input.parquet"
        frame.to_parquet(input_path)
        config = {
            "plant_key": "test",
            "plant_name": "테스트",
            "prefix": "테스트",
            "input_path": str(input_path),
            "datetime_column": "datetime",
            "raw_water_features": {
                "turbidity": "feature_1",
                "ph": "feature_2",
                "alkalinity": "feature_3",
                "temperature": "feature_4",
                "conductivity": "feature_5",
            },
            "outputs": {
                "dataset": str(root / "out" / "dataset.parquet"),
                "report": str(root / "out" / "report.md"),
                "log": str(root / "out" / "correction.csv"),
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return config_path

    def test_dry_run_computes_summary_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_fixture(
                root, [10.0, 10.1, 9.9, 100.0, 10.0, 10.1, 9.9]
            )

            summary = run_stage1(config_path, dry_run=True)

            self.assertEqual(5, summary.corrected_count)
            self.assertEqual("review_required", summary.guard)
            self.assertFalse((root / "out").exists())

    def test_actual_run_preserves_rows_schema_invalids_and_untouched_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [10.0, 10.1, 9.9, 100.0, 10.0, 10.1, 9.9, np.nan, 0.0]
            config_path = self._write_fixture(root, values)
            original = pd.read_parquet(root / "input.parquet")

            summary = run_stage1(config_path, dry_run=False)

            output = pd.read_parquet(root / "out" / "dataset.parquet")
            log = pd.read_csv(root / "out" / "correction.csv", encoding="utf-8-sig")
            self.assertEqual(len(original), len(output))
            self.assertEqual(list(original.columns), list(output.columns))
            self.assertEqual(original.index.name, output.index.name)
            pd.testing.assert_series_equal(original["untouched"], output["untouched"])
            self.assertTrue(pd.isna(output.iloc[7]["feature_1"]))
            self.assertEqual(0.0, output.iloc[8]["feature_1"])
            self.assertEqual(LOG_COLUMNS, log.columns.tolist())
            self.assertEqual(summary.corrected_count, len(log))
            self.assertTrue((root / "out" / "report.md").is_file())

    def test_blocking_guard_prevents_all_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_fixture(
                root, [10.0, 10.1, 9.9, 100.0, 100.0, 10.0, 10.1, 9.9]
            )

            with self.assertRaises(GuardBlockedError):
                run_stage1(config_path, dry_run=False)

            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
