import pandas as pd
import os
import glob

# --- Configuration ---
RESULTS_BASE_DIR_1H = "analysis_results"
RESULTS_BASE_DIR_30MIN = "analysis_results_30min"
RESULTS_BASE_DIR_5MIN = "analysis_results_5min"

# --- Helper Functions ---
def find_latest_results_dir(base_dir):
    """Finds the path to the latest timestamped results directory within a base directory."""
    # List all directories in the base directory
    all_entries = glob.glob(os.path.join(base_dir, '*'))
    # Filter for directories that match the expected timestamp format (contains _YYYYMMDD_HHMMSS)
    # This assumes the naming convention established in the analyzer scripts
    result_dirs = [d for d in all_entries if os.path.isdir(d) and len(os.path.basename(d).split('_')) >= 3 and len(os.path.basename(d).split('_')[-1]) == 6]
    
    if not result_dirs:
        return None
    
    # Sort directories by name (which includes the timestamp) to get the latest
    latest_dir = sorted(result_dirs)[-1]
    return latest_dir

def load_raw_pattern_results(results_dir, filename):
    """Loads the raw pattern results DataFrame from a specified directory and filename."""
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure timestamp column is datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print(f"Error: Raw pattern results file not found at {filepath}")
        return None

# --- Cross-Timeframe Analysis Function ---
def analyze_cross_timeframe_correlation(df_1h, df_30min, df_5min):
    """Analyzes correlations between patterns found in different timeframes."""
    print("\nPerforming Cross-Timeframe Correlation Analysis...")

    correlation_results = []

    # Analyze 1h patterns and presence/type of 30min/5min patterns within them
    if df_1h is not None and not df_1h.empty:
        print("Analyzing 1-hour patterns...")
        for index_1h, pattern_1h in df_1h.iterrows():
            # Calculate the time window for the 1h pattern (from C0 timestamp to C2 timestamp)
            c0_ts_1h = pattern_1h['timestamp']
            # C2 timestamp is C0 timestamp + 2 * interval (1 hour = 60 minutes)
            c2_ts_1h = c0_ts_1h + pd.Timedelta(minutes=2 * 60)

            # Find 30min patterns within this 1h window
            nested_30min_patterns = []
            if df_30min is not None and not df_30min.empty:
                 # A 30min pattern is considered nested if its C0 timestamp is within the 1h C0-C2 window
                 nested_30min_patterns = df_30min[
                    (df_30min['timestamp'] >= c0_ts_1h) &
                    (df_30min['timestamp'] <= c2_ts_1h)
                ]

            # Find 5min patterns within this 1h window
            nested_5min_patterns_in_1h = []
            if df_5min is not None and not df_5min.empty:
                 # A 5min pattern is considered nested if its C0 timestamp is within the 1h C0-C2 window
                 nested_5min_patterns_in_1h = df_5min[
                     (df_5min['timestamp'] >= c0_ts_1h) &
                     (df_5min['timestamp'] <= c2_ts_1h)
                 ]

            # Determine presence and type of nested patterns
            has_30min_bearish = any(nested_30min_patterns['type'] == 'bearish')
            has_30min_bullish = any(nested_30min_patterns['type'] == 'bullish')
            has_5min_bearish_in_1h = any(nested_5min_patterns_in_1h['type'] == 'bearish')
            has_5min_bullish_in_1h = any(nested_5min_patterns_in_1h['type'] == 'bullish')

            correlation_results.append({
                '1h_timestamp': pattern_1h['timestamp'],
                '1h_type': pattern_1h['type'],
                '1h_swept_mid': pattern_1h['swept_mid'],
                '1h_swept_first': pattern_1h['swept_first'],
                '1h_swept_open': pattern_1h['swept_open'],
                'nested_30min_bearish': has_30min_bearish,
                'nested_30min_bullish': has_30min_bullish,
                'nested_5min_bearish_in_1h': has_5min_bearish_in_1h,
                'nested_5min_bullish_in_1h': has_5min_bullish_in_1h,
                'nested_30min_count': len(nested_30min_patterns),
                'nested_5min_count_in_1h': len(nested_5min_patterns_in_1h),
            })

    correlation_df_1h = pd.DataFrame(correlation_results)

    # Analyze 30min patterns and presence/type of 5min patterns within them
    correlation_results_30min = []
    if df_30min is not None and not df_30min.empty:
        print("Analyzing 30-minute patterns...")
        for index_30min, pattern_30min in df_30min.iterrows():
            # Calculate the time window for the 30min pattern (from C0 timestamp to C2 timestamp)
            c0_ts_30min = pattern_30min['timestamp']
            # C2 timestamp is C0 timestamp + 2 * interval (30 minutes)
            c2_ts_30min = c0_ts_30min + pd.Timedelta(minutes=2 * 30)

            # Find 5min patterns within this 30min window
            nested_5min_patterns_in_30min = []
            if df_5min is not None and not df_5min.empty:
                 # A 5min pattern is considered nested if its C0 timestamp is within the 30min C0-C2 window
                 nested_5min_patterns_in_30min = df_5min[
                     (df_5min['timestamp'] >= c0_ts_30min) &
                     (df_5min['timestamp'] <= c2_ts_30min)
                 ]

            # Determine presence and type of nested patterns
            has_5min_bearish_in_30min = any(nested_5min_patterns_in_30min['type'] == 'bearish')
            has_5min_bullish_in_30min = any(nested_5min_patterns_in_30min['type'] == 'bullish')

            correlation_results_30min.append({
                '30min_timestamp': pattern_30min['timestamp'],
                '30min_type': pattern_30min['type'],
                '30min_swept_mid': pattern_30min['swept_mid'],
                '30min_swept_first': pattern_30min['swept_first'],
                '30min_swept_open': pattern_30min['swept_open'],
                'nested_5min_bearish_in_30min': has_5min_bearish_in_30min,
                'nested_5min_bullish_in_30min': has_5min_bullish_in_30min,
                'nested_5min_count_in_30min': len(nested_5min_patterns_in_30min),
            })
    correlation_df_30min = pd.DataFrame(correlation_results_30min)

    # --- Step 2: Report Findings ---
    print("\nCross-Timeframe Correlation Summary:")

    # Report for 1h patterns
    if not correlation_df_1h.empty:
        print("\nAnalysis for 1-hour Patterns:")
        for pattern_type_1h in ['bearish', 'bullish']:
            print(f"  1-hour {pattern_type_1h.capitalize()} patterns:")
            patterns_1h = correlation_df_1h[correlation_df_1h['1h_type'] == pattern_type_1h]
            total_1h_patterns = len(patterns_1h)

            if total_1h_patterns > 0:
                # Define categories for nested patterns
                nested_categories = {
                    'Any 30min': patterns_1h['nested_30min_bearish'] | patterns_1h['nested_30min_bullish'],
                    'Bearish 30min': patterns_1h['nested_30min_bearish'],
                    'Bullish 30min': patterns_1h['nested_30min_bullish'],
                    'Any 5min': patterns_1h['nested_5min_bearish_in_1h'] | patterns_1h['nested_5min_bullish_in_1h'],
                    'Bearish 5min': patterns_1h['nested_5min_bearish_in_1h'],
                    'Bullish 5min': patterns_1h['nested_5min_bullish_in_1h'],
                    'No nested patterns': ~(patterns_1h['nested_30min_bearish'] | patterns_1h['nested_30min_bullish'] | patterns_1h['nested_5min_bearish_in_1h'] | patterns_1h['nested_5min_bullish_in_1h'])
                }

                for category, mask in nested_categories.items():
                    categorized_patterns = patterns_1h[mask]
                    count = len(categorized_patterns)
                    if count > 0:
                        hit_rate_mid = (categorized_patterns['1h_swept_mid'].sum() / count) * 100
                        hit_rate_first = (categorized_patterns['1h_swept_first'].sum() / count) * 100
                        hit_rate_open = (categorized_patterns['1h_swept_open'].sum() / count) * 100
                        print(f"    - {category}: Count = {count}, Hit% (Mid, 1st, Opn) = ({hit_rate_mid:.2f}, {hit_rate_first:.2f}, {hit_rate_open:.2f})")

    else:
        print("No 1-hour patterns found for detailed cross-timeframe analysis.")

    # Report for 30min patterns
    if not correlation_df_30min.empty:
        print("\nAnalysis for 30-minute Patterns:")
        for pattern_type_30min in ['bearish', 'bullish']:
            print(f"  30-minute {pattern_type_30min.capitalize()} patterns:")
            patterns_30min = correlation_df_30min[correlation_df_30min['30min_type'] == pattern_type_30min]
            total_30min_patterns = len(patterns_30min)

            if total_30min_patterns > 0:
                # Define categories for nested 5min patterns within 30min patterns
                nested_categories_30min = {
                    'Any 5min': patterns_30min['nested_5min_bearish_in_30min'] | patterns_30min['nested_5min_bullish_in_30min'],
                    'Bearish 5min': patterns_30min['nested_5min_bearish_in_30min'],
                    'Bullish 5min': patterns_30min['nested_5min_bullish_in_30min'],
                    'No nested 5min patterns': ~(patterns_30min['nested_5min_bearish_in_30min'] | patterns_30min['nested_5min_bullish_in_30min'])
                }

                for category, mask in nested_categories_30min.items():
                    categorized_patterns = patterns_30min[mask]
                    count = len(categorized_patterns)
                    if count > 0:
                         hit_rate_mid = (categorized_patterns['30min_swept_mid'].sum() / count) * 100
                         hit_rate_first = (categorized_patterns['30min_swept_first'].sum() / count) * 100
                         hit_rate_open = (categorized_patterns['30min_swept_open'].sum() / count) * 100
                         print(f"    - {category}: Count = {count}, Hit% (Mid, 1st, Opn) = ({hit_rate_mid:.2f}, {hit_rate_first:.2f}, {hit_rate_open:.2f})")

    else:
        print("No 30-minute patterns found for detailed cross-timeframe analysis.")

    # Add analysis for other combinations and potential insights into R:R
    # This can involve looking at the price movement after the C2 of the larger timeframe pattern
    # based on the nested patterns, but that requires accessing the full price data within this script.
    # For now, the analysis focuses on the Hit% of the larger timeframe pattern's C2.


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Cross-Timeframe Correlation Analysis...")

    # 1. Find the latest results directories
    latest_dir_1h = find_latest_results_dir(RESULTS_BASE_DIR_1H)
    latest_dir_30min = find_latest_results_dir(RESULTS_BASE_DIR_30MIN)
    latest_dir_5min = find_latest_results_dir(RESULTS_BASE_DIR_5MIN)

    if not latest_dir_1h:
        print(f"Could not find latest 1-hour results directory in {RESULTS_BASE_DIR_1H}")
    if not latest_dir_30min:
        print(f"Could not find latest 30-minute results directory in {RESULTS_BASE_DIR_30MIN}")
    if not latest_dir_5min:
        print(f"Could not find latest 5-minute results directory in {RESULTS_BASE_DIR_5MIN}")

    if latest_dir_1h and latest_dir_30min and latest_dir_5min:
        print(f"Found latest 1-hour results in: {latest_dir_1h}")
        print(f"Found latest 30-minute results in: {latest_dir_30min}")
        print(f"Found latest 5-minute results in: {latest_dir_5min}")

        # 2. Load the raw pattern results
        df_1h = load_raw_pattern_results(latest_dir_1h, "raw_pattern_results.csv")
        df_30min = load_raw_pattern_results(latest_dir_30min, "raw_pattern_results_30min.csv")
        df_5min = load_raw_pattern_results(latest_dir_5min, "raw_pattern_results_5min.csv")

        # 3. Perform cross-timeframe analysis
        # Ensure at least the base timeframe data is loaded for analysis
        if df_1h is not None or df_30min is not None:
             analyze_cross_timeframe_correlation(df_1h, df_30min, df_5min)
        else:
             print("Could not load sufficient data to perform cross-timeframe analysis (need at least 1h or 30min data).")

    else:
        print("Could not locate latest results directories for all necessary timeframes (1h, 30min, 5min). Please run the individual timeframe analyzers first.") 