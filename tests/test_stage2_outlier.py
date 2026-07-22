import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from water_treatment.stage2_outlier import (
    REMOVAL_LOG_COLUMNS,
    analyze_cluster,
    run_stage2,
)


class Stage2OutlierTests(unittest.TestCase):
    def _frame(self, rows=41, spike_at=None):
        index = pd.date_range("2026-01-01", periods=rows, freq="10min")
        base = np.array([10.0 + (index_ % 3) * 0.05 for index_ in range(rows)])
        data = {"datetime": index, "cluster_label": np.zeros(rows, dtype=np.int8)}
        for number, name in enumerate(("turbidity", "ph", "alkalinity", "temperature", "conductivity")):
            data[name] = base + number
        if spike_at is not None:
            data["turbidity"][spike_at] = 1000.0
        return pd.DataFrame(data)

    def test_local_statistical_spike_is_flagged_and_marked_for_removal(self):
        frame = self._frame(spike_at=20)

        result = analyze_cluster(
            frame,
            raw_water_features=["turbidity", "ph", "alkalinity", "temperature", "conductivity"],
            datetime_column="datetime",
        )

        flag = int(result.frame.loc[20, "turbidity_stage2_flag"])
        self.assertEqual(3, flag & 3)  # IQR and robust Z-score bits
        self.assertTrue(result.removal_mask[20])
        self.assertEqual(1, result.removed_count)

    def test_if_flag_without_statistical_spike_never_deletes_a_row(self):
        frame = self._frame()
        frame.loc[20, "ph"] = 10.8  # locally plausible but distinct multivariate combination

        result = analyze_cluster(
            frame,
            raw_water_features=["turbidity", "ph", "alkalinity", "temperature", "conductivity"],
            datetime_column="datetime",
        )

        self.assertEqual(0, result.removed_count)
        self.assertFalse(result.removal_mask.any())

    def test_run_preserves_source_columns_and_writes_required_stage2_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input_cluster0.parquet"
            original = self._frame(spike_at=20)
            original["existing_flag"] = 7
            original.to_parquet(source, index=False)
            config = {
                "plant_key": "test",
                "plant_name": "테스트",
                "prefix": "테스트",
                "datetime_column": "datetime",
                "raw_water_features": {
                    "turbidity": "turbidity",
                    "ph": "ph",
                    "alkalinity": "alkalinity",
                    "temperature": "temperature",
                    "conductivity": "conductivity",
                },
                "input_template": str(root / "input_cluster{cluster}.parquet"),
                "output_template": str(root / "stage2" / "out_cluster{cluster}.parquet"),
                "report_template": str(root / "report" / "report_cluster{cluster}.md"),
                "log_path": str(root / "log" / "removed.csv"),
            }
            config_path = root / "stage2.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

            summary = run_stage2(config_path, clusters=[0])

            output = pd.read_parquet(root / "stage2" / "out_cluster0.parquet")
            log = pd.read_csv(root / "log" / "removed.csv", encoding="utf-8-sig")
            self.assertEqual(len(original) - 1, len(output))
            self.assertEqual(7, int(output["existing_flag"].iloc[0]))
            self.assertIn("turbidity_stage2_flag", output.columns)
            self.assertEqual(REMOVAL_LOG_COLUMNS, log.columns.tolist())
            self.assertEqual(1, summary.cluster_summaries[0].removed_count)
            self.assertTrue((root / "report" / "report_cluster0.md").is_file())


if __name__ == "__main__":
    unittest.main()
