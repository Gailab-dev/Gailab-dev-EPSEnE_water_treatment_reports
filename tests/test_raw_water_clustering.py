# -*- coding: utf-8 -*-
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from water_treatment.clustering import (
    LABEL_COLUMN,
    aggregate_fixed_windows,
    fit_and_assign_labels,
    read_input_frame,
    save_outputs,
)


FEATURES = {
    "turbidity": "raw_turbidity",
    "ph": "raw_ph",
    "alkalinity": "raw_alkalinity",
    "temperature": "raw_temperature",
    "conductivity": "raw_conductivity",
}


def _row(timestamp: str, base: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "datetime": pd.Timestamp(timestamp),
        "raw_turbidity": base,
        "raw_ph": base + 1,
        "raw_alkalinity": base + 2,
        "raw_temperature": base + 3,
        "raw_conductivity": base + 4,
        "flow": base * 10,
        "sensor_flag": 0,
        "sensor_src": 7,
        "status": "ok",
    }
    row.update(overrides)
    return row


class RawWaterClusteringTests(unittest.TestCase):
    def test_read_input_restores_datetime_stored_as_parquet_index(self):
        source = pd.DataFrame(
            {
                "raw_turbidity": [1.0, 2.0],
                "raw_ph": [7.0, 7.1],
            },
            index=pd.DatetimeIndex(
                ["2026-01-01 00:00:00", "2026-01-01 00:01:00"], name="datetime"
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indexed-datetime.parquet"
            source.to_parquet(path)

            loaded = read_input_frame(path, "datetime")

        self.assertIn("datetime", loaded.columns)
        self.assertEqual(loaded["datetime"].tolist(), source.index.tolist())

    def test_fixed_ten_minute_windows_do_not_borrow_from_next_window(self):
        frame = pd.DataFrame(
        [
            _row("2026-01-01 00:10:00", 50),
            _row("2026-01-01 00:00:00", 10),
            _row(
                "2026-01-01 00:09:00",
                30,
                raw_turbidity=0.0,
                sensor_flag=1,
                sensor_src=8,
                status="warning",
            ),
        ]
    )

        aggregated = aggregate_fixed_windows(frame, "datetime", FEATURES, interval="10min")

        self.assertEqual(
            aggregated["datetime"].tolist(),
            [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 00:10:00")],
        )
        first = aggregated.iloc[0]
        self.assertEqual(first["raw_turbidity"], 10.0)
        self.assertEqual(first["raw_ph"], 21.0)
        self.assertEqual(first["flow"], 200.0)
        self.assertEqual(first["sensor_flag"], 1)
        self.assertEqual(first["sensor_src"], 7)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(aggregated.iloc[1]["raw_turbidity"], 50.0)


    def test_fit_keeps_invalid_window_with_nullable_label(self):
        aggregated = pd.DataFrame(
        [
            _row("2026-01-01 00:00:00", 1),
            _row("2026-01-01 00:10:00", 10),
            _row("2026-01-01 00:20:00", 100),
            _row("2026-01-01 00:30:00", 5, raw_ph=np.nan),
        ]
    )

        labeled, model, centers, valid_mask = fit_and_assign_labels(
            aggregated, FEATURES, k=3, random_state=42
        )

        self.assertEqual(len(labeled), 4)
        self.assertEqual(valid_mask.tolist(), [True, True, True, False])
        self.assertEqual(labeled[LABEL_COLUMN].dtype, pd.Int64Dtype())
        self.assertTrue(pd.isna(labeled.iloc[-1][LABEL_COLUMN]))
        self.assertEqual(sorted(labeled.loc[valid_mask, LABEL_COLUMN].astype(int).unique()), [0, 1, 2])
        self.assertEqual(model.named_steps["kmeans"].n_clusters, 3)
        self.assertEqual(centers.shape, (3, 6))


    def test_k_other_than_three_is_rejected(self):
        aggregated = pd.DataFrame([_row("2026-01-01 00:00:00", 1)] * 4)

        with self.assertRaisesRegex(ValueError, "k must be fixed at 3"):
            fit_and_assign_labels(aggregated, FEATURES, k=2)


    def test_save_outputs_writes_full_split_model_and_report(self):
        aggregated = pd.DataFrame(
        [
            _row("2026-01-01 00:00:00", 1),
            _row("2026-01-01 00:10:00", 10),
            _row("2026-01-01 00:20:00", 100),
            _row("2026-01-01 00:30:00", 5, raw_ph=np.nan),
        ]
    )
        labeled, model, centers, valid_mask = fit_and_assign_labels(aggregated, FEATURES)

        with tempfile.TemporaryDirectory() as directory:
            paths = save_outputs(
                labeled=labeled,
                model=model,
                centers=centers,
                valid_mask=valid_mask,
                output_root=Path(directory),
                prefix="테스트",
                plant_name="테스트정수장",
                feature_map=FEATURES,
                source_rows=40,
                source_path=Path("readonly-input.parquet"),
            )

            self.assertTrue(paths.labeled_dataset.exists())
            self.assertEqual([path.exists() for path in paths.cluster_datasets], [True, True, True])
            self.assertTrue(paths.model.exists())
            self.assertTrue(paths.report.exists())
            self.assertEqual(len(pd.read_parquet(paths.labeled_dataset)), 4)
            self.assertEqual(sum(len(pd.read_parquet(path)) for path in paths.cluster_datasets), 3)
            self.assertEqual(joblib.load(paths.model).named_steps["kmeans"].n_clusters, 3)
            report = paths.report.read_text(encoding="utf-8")
            self.assertIn("원수 상태 안정화", report)
            self.assertIn("체류시간 반영이 아니다", report)
            self.assertNotIn("저탁도", report)
            self.assertNotIn("중탁도", report)
            self.assertNotIn("고탁도", report)


if __name__ == "__main__":
    unittest.main()
