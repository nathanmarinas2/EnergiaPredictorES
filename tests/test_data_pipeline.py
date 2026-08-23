import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import add_lag_features, regularize_time_index
from src.models.baseline import select_feature_columns


class DataPipelineTests(unittest.TestCase):
    def test_hourly_target_features_use_only_history(self):
        index = pd.date_range("2020-01-01", periods=400, freq="h")
        target = pd.Series(np.arange(len(index), dtype=float) ** 2, index=index)
        frame = pd.DataFrame({"total load actual": target}, index=index)

        result = add_lag_features(frame, frequency="h")

        self.assertNotIn("load_diff_1h", result.columns)
        self.assertNotIn("load_ratio_24h", result.columns)
        self.assertAlmostEqual(
            result.loc[index[200], "load_diff_1step"],
            target.iloc[199] - target.iloc[198],
        )
        self.assertAlmostEqual(
            result.loc[index[200], "load_diff_1day"],
            target.iloc[176] - target.iloc[175],
        )
        self.assertNotEqual(
            result.loc[index[200], "load_diff_1step"],
            target.iloc[200] - target.iloc[199],
        )

    def test_daily_frequency_uses_day_and_week_lags(self):
        index = pd.date_range("2020-01-01", periods=30, freq="D")
        frame = pd.DataFrame(
            {"total load actual": np.arange(len(index), dtype=float)}, index=index
        )

        result = add_lag_features(frame, frequency="D")

        self.assertEqual(result.loc[index[10], "load_lag_1day"], 9.0)
        self.assertEqual(result.loc[index[10], "load_lag_1week"], 3.0)

    def test_feature_selector_rejects_target_differences_and_ratios(self):
        frame = pd.DataFrame(
            {
                "total load actual": [1.0],
                "load_lag_1day": [1.0],
                "load_diff_1step": [0.0],
                "load_ratio_1day": [1.0],
                "hour": [0],
            }
        )

        features = select_feature_columns(frame, "total load actual")

        self.assertEqual(features, ["load_lag_1day", "hour"])

    def test_regularize_daily_index_does_not_fill_target(self):
        index = pd.to_datetime(["2020-01-01 23:00", "2020-01-03 23:00"])
        frame = pd.DataFrame({"total load actual": [10.0, 30.0]}, index=index)

        result = regularize_time_index(frame, frequency="D")

        self.assertEqual(result.index.freqstr, "D")
        self.assertTrue(np.isnan(result.loc["2020-01-02", "total load actual"]))


if __name__ == "__main__":
    unittest.main()
