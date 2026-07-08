import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_common_module():
    module_path = Path("dataset") / "전처리" / "outlier_preprocess_common.py"
    spec = importlib.util.spec_from_file_location("outlier_preprocess_common", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OutlierPreprocessCommonTests(unittest.TestCase):
    def test_detection_plan_keeps_only_core_sensor_columns(self):
        common = load_common_module()
        index = pd.date_range("2026-01-01", periods=3, freq="min")
        df = pd.DataFrame(
            {
                "raw_water_FLOW": [1.0, 2.0, 3.0],
                "settled_PH": [7.0, 7.1, 7.2],
                "raw_water_FLOW_flag": [0, 0, 1],
                "PAC_target": [10.0, 10.0, 10.0],
                "coagulant_SV": [5.0, 5.0, 5.0],
                "daily_KG": [1.0, 2.0, 3.0],
                "pump_status": [1.0, 1.0, 0.0],
                "total_accumulated": [100.0, 101.0, 102.0],
            },
            index=index,
        )

        plan = common.build_detection_plan(df)

        self.assertEqual(["raw_water_FLOW", "settled_PH"], plan.target_columns)
        self.assertEqual(["raw_water_FLOW", "settled_PH"], plan.kalman_columns)
        self.assertIn("raw_water_FLOW_flag", plan.excluded_columns)
        self.assertIn("coagulant_SV", plan.excluded_columns)
        self.assertIn("daily_KG", plan.excluded_columns)
        self.assertIn("pump_status", plan.excluded_columns)

    def test_statistical_detection_excludes_missing_and_zero_values(self):
        common = load_common_module()
        index = pd.date_range("2026-01-01", periods=9, freq="min")
        df = pd.DataFrame(
            {
                "sensor_a": [10.0, 11.0, 10.0, 0.0, np.nan, 10.0, 11.0, 10.0, 1000.0],
                "sensor_b": [5.0] * 9,
            },
            index=index,
        )

        result = common.detect_statistical_outliers(df)

        self.assertFalse(result.row_mask.loc[index[3]])
        self.assertFalse(result.row_mask.loc[index[4]])
        self.assertTrue(result.row_mask.loc[index[8]])
        self.assertEqual(1, int(result.column_report.loc["sensor_a", "IQR_이상치_개수"]))
        self.assertEqual(0, int(result.column_report.loc["sensor_b", "통계_이상치_합집합_개수"]))

    def test_statistical_detection_requires_two_methods_on_the_same_row(self):
        common = load_common_module()
        index = pd.date_range("2026-01-01", periods=9, freq="min")
        df = pd.DataFrame(
            {
                "sensor_a": [10.0, 11.0, 10.0, 10.0, 11.0, 10.0, 11.0, 10.0, 1000.0],
            },
            index=index,
        )

        result = common.detect_statistical_outliers(
            df,
            kalman_columns=[],
            z_threshold=999.0,
            kalman_threshold=999.0,
        )

        self.assertFalse(result.row_mask.loc[index[8]])
        self.assertEqual(1, int(result.column_report.loc["sensor_a", "IQR_이상치_개수"]))
        self.assertEqual(1, int(result.column_report.loc["sensor_a", "통계_이상치_합집합_개수"]))
        self.assertEqual(0, int(result.column_report.loc["sensor_a", "통계_제거대상_개수"]))

    def test_ai_detection_runs_per_column_and_ignores_missing_and_zero_values(self):
        common = load_common_module()
        index = pd.date_range("2026-01-01", periods=103, freq="min")
        normal_values = [10.0 + (i % 5) * 0.01 for i in range(100)]
        df = pd.DataFrame(
            {
                "sensor_a": normal_values + [0.0, np.nan, 500.0],
                "sensor_b": [1.0] * 103,
            },
            index=index,
        )

        result = common.detect_ai_outliers(df, contamination=0.01, random_state=42)

        self.assertFalse(result.row_mask.loc[index[100]])
        self.assertFalse(result.row_mask.loc[index[101]])
        self.assertTrue(result.row_mask.loc[index[102]])
        self.assertGreaterEqual(int(result.column_report.loc["sensor_a", "AI_IsolationForest_이상치_개수"]), 1)
        self.assertEqual(0, int(result.column_report.loc["sensor_b", "AI_IsolationForest_이상치_개수"]))

    def test_pipeline_preserves_index_columns_and_remaining_zero_missing_values(self):
        common = load_common_module()
        index = pd.date_range("2026-01-01", periods=10, freq="min", name="datetime")
        df = pd.DataFrame(
            {
                "raw_water_FLOW": [10.0, 11.0, 10.0, 0.0, np.nan, 10.0, 11.0, 10.0, 12.0, 1000.0],
                "raw_water_FLOW_flag": [0.0] * 10,
            },
            index=index,
        )

        processed, summary, report = common.preprocess_dataframe(df, plant_name="테스트")

        self.assertEqual(["raw_water_FLOW", "raw_water_FLOW_flag"], list(processed.columns))
        self.assertEqual("datetime", processed.index.name)
        self.assertIn(index[3], processed.index)
        self.assertIn(index[4], processed.index)
        self.assertEqual(0.0, processed.loc[index[3], "raw_water_FLOW"])
        self.assertTrue(pd.isna(processed.loc[index[4], "raw_water_FLOW"]))
        self.assertNotIn(index[9], processed.index)
        self.assertEqual(len(df), summary["원본 행 수"])
        self.assertEqual(len(processed), summary["최종 행 수"])
        self.assertIn("컬럼명", report.columns)
        self.assertEqual(["raw_water_FLOW"], summary["실제 탐지 대상 컬럼 목록"])
        self.assertEqual(["raw_water_FLOW_flag"], summary["탐지 제외 컬럼 목록"])

    def test_removal_rate_policy_warns_and_blocks_high_removal_save(self):
        common = load_common_module()

        self.assertEqual("", common.removal_rate_warning(20.0))
        self.assertIn("20%", common.removal_rate_warning(20.1))
        self.assertTrue(common.should_save_outputs(30.0))
        self.assertFalse(common.should_save_outputs(30.1))


if __name__ == "__main__":
    unittest.main()
