import unittest
import pandas as pd
import os
from analyzers.hourly_swing_failure_analyzer import load_data, is_bearish_swing_failure, is_bullish_swing_failure

TEST_DATA_DIR = "tests/test_data"
SAMPLE_DATA_FILE = os.path.join(TEST_DATA_DIR, "sample_h1_data.csv")

# Ensure test data directory exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Create a dummy sample data file for testing
dummy_data = {
    'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-01 03:00:00', '2023-01-01 04:00:00']),
    'open': [100, 101, 102, 103, 104],
    'high': [101, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
}
dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv(SAMPLE_DATA_FILE, index=False)

class TestHourlySwingFailureAnalyzer(unittest.TestCase):

    def test_load_data(self):
        # Test if data loads correctly and has expected columns/index
        df = load_data(SAMPLE_DATA_FILE)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_is_bearish_swing_failure_valid(self):
        # Test a clear valid bearish swing failure scenario
        c0 = pd.Series({'open': 100, 'high': 105, 'low': 99, 'close': 104})
        c1 = pd.Series({'open': 104.5, 'high': 106, 'low': 103, 'close': 103.5})
        # Use typical threshold values for testing
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertTrue(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should identify valid bearish SFP")

    def test_is_bearish_swing_failure_invalid_c0_type(self):
        # Test scenario where C0 is not bullish
        c0 = pd.Series({'open': 104, 'high': 105, 'low': 99, 'close': 100})
        c1 = pd.Series({'open': 100.5, 'high': 106, 'low': 99.5, 'close': 101})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bearish SFP if C0 is not bullish")

    def test_is_bearish_swing_failure_invalid_c1_wick(self):
        # Test scenario where C1 does not wick above C0 high
        c0 = pd.Series({'open': 100, 'high': 105, 'low': 99, 'close': 104})
        c1 = pd.Series({'open': 104.5, 'high': 104.8, 'low': 103, 'close': 103.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bearish SFP if C1 does not wick above C0 high")

    def test_is_bearish_swing_failure_invalid_c1_close(self):
        # Test scenario where C1 closes outside C0 range (below C0 low)
        c0 = pd.Series({'open': 100, 'high': 105, 'low': 99, 'close': 104})
        c1 = pd.Series({'open': 104.5, 'high': 106, 'low': 98, 'close': 98.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bearish SFP if C1 closes below C0 low")

    def test_is_bearish_swing_failure_invalid_c1_height(self):
        # Test scenario where C1 height is too large
        c0 = pd.Series({'open': 100, 'high': 105, 'low': 99, 'close': 104})
        c1 = pd.Series({'open': 104.5, 'high': 106, 'low': 100.5, 'close': 105.5})
        c0_c1_relative_height_threshold = 0.1 # Make threshold strict for this test
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bearish SFP if C1 height is too large")

    def test_is_bearish_swing_failure_invalid_c1_retracement(self):
        # Test scenario where C1 low retraces too deep into C0
        c0 = pd.Series({'open': 100, 'high': 105, 'low': 99, 'close': 104})
        c1 = pd.Series({'open': 104.5, 'high': 106, 'low': 99.8, 'close': 103.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.1 # Make threshold strict for this test

        self.assertFalse(is_bearish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bearish SFP if C1 low retraces too deep into C0")

    def test_is_bullish_swing_failure_valid(self):
        # Test a clear valid bullish swing failure scenario
        c0 = pd.Series({'open': 105, 'high': 106, 'low': 100, 'close': 101})
        c1 = pd.Series({'open': 100.5, 'high': 102, 'low': 99, 'close': 101.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertTrue(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should identify valid bullish SFP")

    def test_is_bullish_swing_failure_invalid_c0_type(self):
        # Test scenario where C0 is not bearish
        c0 = pd.Series({'open': 101, 'high': 106, 'low': 100, 'close': 105})
        c1 = pd.Series({'open': 104.5, 'high': 105.5, 'low': 99, 'close': 104})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bullish SFP if C0 is not bearish")

    def test_is_bullish_swing_failure_invalid_c1_wick(self):
        # Test scenario where C1 does not wick below C0 low
        c0 = pd.Series({'open': 105, 'high': 106, 'low': 100, 'close': 101})
        c1 = pd.Series({'open': 100.5, 'high': 102, 'low': 100.2, 'close': 101.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bullish SFP if C1 does not wick below C0 low")

    def test_is_bullish_swing_failure_invalid_c1_close(self):
        # Test scenario where C1 closes outside C0 range (above C0 high)
        c0 = pd.Series({'open': 105, 'high': 106, 'low': 100, 'close': 101})
        c1 = pd.Series({'open': 100.5, 'high': 107, 'low': 99, 'close': 106.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bullish SFP if C1 closes above C0 high")

    def test_is_bullish_swing_failure_invalid_c1_height(self):
        # Test scenario where C1 height is too large
        c0 = pd.Series({'open': 105, 'high': 106, 'low': 100, 'close': 101})
        c1 = pd.Series({'open': 100.5, 'high': 105.5, 'low': 101.5, 'close': 100.5})
        c0_c1_relative_height_threshold = 0.1 # Make threshold strict for this test
        c1_retracement_depth_threshold = 0.5

        self.assertFalse(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bullish SFP if C1 height is too large")

    def test_is_bullish_swing_failure_invalid_c1_retracement(self):
        # Test scenario where C1 high retraces too deep into C0
        c0 = pd.Series({'open': 105, 'high': 106, 'low': 100, 'close': 101})
        c1 = pd.Series({'open': 100.5, 'high': 105.2, 'low': 99, 'close': 101.5})
        c0_c1_relative_height_threshold = 0.9
        c1_retracement_depth_threshold = 0.1 # Make threshold strict for this test

        self.assertFalse(is_bullish_swing_failure(
            c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold
        ), "Should not identify bullish SFP if C1 high retraces too deep into C0")

if __name__ == '__main__':
    unittest.main() 