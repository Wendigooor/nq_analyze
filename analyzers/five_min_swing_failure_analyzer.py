import pandas as pd
import numpy as np
import plotly.graph_objects as go # For plotting
from plotly.subplots import make_subplots # For plotting
import os # For creating plot directories
import datetime # For timestamping results
import sys # To capture stdout
import io # To capture stdout

# --- 1. Configuration ---
DATA_FILE = "frd_sample_futures_NQ/NQ_5min_sample.csv"
OUTPUT_DIR = "plots_swing_failures_5min" # Updated output directory
RESULTS_BASE_DIR = "analysis_results_5min" # Updated results base directory
START_ANALYSIS_DATE = "2010-01-01" # Or as desired (Note: Sample data is recent)
# Parameters for Swing Failure Pattern definition (can be experimented with)
C0_C1_RELATIVE_HEIGHT_THRESHOLD = 0.9 # e.g., C1 height <= 0.9 * C0 height
C1_RETRACEMENT_DEPTH_THRESHOLD = 0.5 # e.g., C1 must not retrace more than 50% into C0 against expected direction

# --- 2. Data Loading and Preprocessing ---
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Adjust column name for timestamp based on the sample data format
    if 'timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['timestamp'])
        df.drop(columns=['timestamp'], inplace=True)
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    elif 'Datetime' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Datetime'])
        df.drop(columns=['Datetime'], inplace=True)
    else:
        raise ValueError("Timestamp or Datetime column not found in data file.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('Timestamp', inplace=True)

    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True) # Standardize
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.dropna(inplace=True)
    df = df[df.index >= pd.to_datetime(START_ANALYSIS_DATE)]
    
    # Calculate features for 5-minute data
    df['hour'] = df.index.hour # Still useful for hourly grouping of 5min patterns
    df['c0_height'] = abs(df['open'] - df['close'])

    # Calculate net change over previous periods (assuming 288 periods per day for 5min data)
    # Commenting out previous day analysis for now in lower timeframes
    # df['prev_1d_change'] = df['close'].pct_change(periods=288) * 100 # 24 hours * 12 (5min in an hour)
    # df['prev_2d_change'] = df['close'].pct_change(periods=576) * 100 # 48 hours * 12
    # df['prev_3d_change'] = df['close'].pct_change(periods=864) * 100 # 72 hours * 12
    
    # if 'prev_1d_change' in df.columns: df.dropna(inplace=True) # Drop rows with NaN values if calculated

    return df

# --- Helper function to categorize daily change (from hourly, not directly used in 5min SFP analysis yet) ---
# def categorize_change(change_pct):
#     if change_pct < -1.0:
#         return 'Strongly Down'
#     elif -1.0 <= change_pct < -0.1:
#         return 'Slightly Down'
#     elif -0.1 <= change_pct <= 0.1:
#         return 'Flat'
#     elif 0.1 < change_pct <= 1.0:
#         return 'Slightly Up'
#     else:
#         return 'Strongly Up'

# --- 3. Pattern Definition Functions ---
def is_bearish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
    """
    c0: pandas Series for the first candle
    c1: pandas Series for the second candle
    """
    if not (c0['close'] > c0['open']): return False # C0 must be bullish
    if not (c1['high'] > c0['high']): return False # C1 wicks above C0 high
    if not (c1['close'] < c0['high'] and c1['close'] > c0['low']): return False # C1 closes inside C0 range

    c0_height = abs(c0['open'] - c0['close'])
    c1_height = abs(c1['open'] - c1['close'])

    # Use c1_retracement_depth_threshold for C1 low condition
    if not (c1['low'] > c0['close'] - c1_retracement_depth_threshold * c0_height): return False

    # Use c0_c1_relative_height_threshold for C1 height condition
    if not (c1_height <= c0_c1_relative_height_threshold * c0_height): return False

    return True

def is_bullish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
    """
    c0: pandas Series for the first candle
    c1: pandas Series for the second candle
    """
    if not (c0['close'] < c0['open']): return False # C0 must be bearish
    if not (c1['low'] < c0['low']): return False # C1 wicks below C0 low
    if not (c1['close'] > c0['low'] and c1['close'] < c0['high']): return False # C1 closes inside C0 range

    c0_height = abs(c0['open'] - c0['close'])
    c1_height = abs(c1['open'] - c1['close'])

    # Use c1_retracement_depth_threshold for C1 high condition
    if not (c1['high'] < c0['close'] + c1_retracement_depth_threshold * c0_height): return False

    # Use c0_c1_relative_height_threshold for C1 height condition
    if not (c1_height <= c0_c1_relative_height_threshold * c0_height): return False

    return True

# --- 4. Analysis Loop ---
def analyze_swing_failures(df, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
    results = [] # To store pattern occurrences and outcomes
    # Need 3 candles: c0, c1, c2
    # Iterate up to the third to last candle
    for i in range(len(df) - 2):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        c2 = df.iloc[i+2]

        # Timestamps for context
        ts0, ts1, ts2 = df.index[i], df.index[i+1], df.index[i+2]

        # Previous day's category is not directly applicable to 5min patterns in the same way
        # We will not include prev_1d_cat in the 5min results for now.
        # prev_1d_cat = categorize_change(df['prev_1d_change'].iloc[i])

        # Bearish SFP
        if is_bearish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['low'] <= c1['low']
            swept_first = c2['low'] <= c0['low']
            swept_open = c2['low'] <= c0['open']
            results.append({
                'timestamp': ts0, 'type': 'bearish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open
            })

        # Bullish SFP
        if is_bullish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['high'] >= c1['high']
            swept_first = c2['high'] >= c0['high']
            swept_open = c2['high'] >= c0['open'] # Assuming target is C0 open for bullish too
            results.append({
                'timestamp': ts0, 'type': 'bullish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open
            })
    return pd.DataFrame(results)

# --- 5. Statistics Aggregation & Output ---
def aggregate_and_print_stats(results_df, output_file=None):
    if results_df.empty:
        output = "No patterns found.\n"
        print(output)
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(output)
        return

    # Capture print output
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    # NY Time hours for aggregation (0-23)
    stats_summary = []
    # Aggregate by hour
    # Previous day's category aggregation is removed for 5min analysis for now
    # for category in ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']:
    for hour in range(24):
        # hourly_patterns = results_df[(results_df['hour_c0'] == hour) & (results_df['prev_1d_cat'] == category)]
        hourly_patterns = results_df[results_df['hour_c0'] == hour]

        bear_patterns = hourly_patterns[hourly_patterns['type'] == 'bearish']
        bull_patterns = hourly_patterns[hourly_patterns['type'] == 'bullish']

        bear_occurrences = len(bear_patterns)
        bull_occurrences = len(bull_patterns)

        # For 5min data, total candles per hour will be different
        # Need original df to calculate occurrences / total candles in that hour
        # Temporarily removing this part of the metric
        # bear_total_in_hour = len(df[df['hour']==hour])
        # bull_total_in_hour = len(df[df['hour']==hour])

        if bear_occurrences > 0:
            # Hit rate as percentage of total bearish patterns found across all hours
            total_bear_patterns = len(results_df[results_df['type'] == 'bearish'])
            bear_hit_rate = (bear_occurrences / total_bear_patterns) * 100 if total_bear_patterns > 0 else 0
            bear_swept_mid_pct = (bear_patterns['swept_mid'].sum() / bear_occurrences) * 100
            bear_swept_first_pct = (bear_patterns['swept_first'].sum() / bear_occurrences) * 100
            bear_swept_open_pct = (bear_patterns['swept_open'].sum() / bear_occurrences) * 100
        else:
            bear_hit_rate, bear_swept_mid_pct, bear_swept_first_pct, bear_swept_open_pct = 0,0,0,0

        if bull_occurrences > 0:
            # Hit rate as percentage of total bullish patterns found across all hours
            total_bull_patterns = len(results_df[results_df['type'] == 'bullish'])
            bull_hit_rate = (bull_occurrences / total_bull_patterns) * 100 if total_bull_patterns > 0 else 0
            bull_swept_mid_pct = (bull_patterns['swept_mid'].sum() / bull_occurrences) * 100
            bull_swept_first_pct = (bull_patterns['swept_first'].sum() / bull_occurrences) * 100
            bull_swept_open_pct = (bull_patterns['swept_open'].sum() / bull_occurrences) * 100
        else:
            bull_hit_rate, bull_swept_mid_pct, bull_swept_first_pct, bull_swept_open_pct = 0,0,0,0

        # Create the hour triplet string - Note: this is still based on C0 hour, not a 5min interval triplet
        h0 = hour
        h1 = (hour + 1) % 24 # Still show the hour of C1 and C2 for context, even if it spans across 5min intervals
        h2 = (hour + 2) % 24
        hour_triplet_str = f"{h0:02d},{h1:02d},{h2:02d}"

        # Only add rows if there are occurrences in this hour
        if bear_occurrences > 0 or bull_occurrences > 0:
            stats_summary.append({
                # 'Prev Day': category, # Removed for 5min analysis
                'Hours': hour_triplet_str,
                'Bear Nr.': bear_occurrences,
                'Hit%': f"{bear_hit_rate:.2f}",
                'Mid%': f"{bear_swept_mid_pct:.2f}",
                '1st%': f"{bear_swept_first_pct:.2f}",
                'Opn%': f"{bear_swept_open_pct:.2f}",
                '| Bull Nr.': bull_occurrences,
                'Hit%_bull': f"{bull_hit_rate:.2f}",
                'Mid%_bull': f"{bull_swept_mid_pct:.2f}",
                '1st%_bull': f"{bull_swept_first_pct:.2f}",
                'Opn%_bull': f"{bull_swept_open_pct:.2f}"
            })

    summary_df = pd.DataFrame(stats_summary)
    # Sort by Hour
    # category_order = ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']
    # summary_df['Prev Day'] = pd.Categorical(summary_df['Prev Day'], categories=category_order, ordered=True)
    # summary_df.sort_values(by=['Prev Day', 'Hours'], inplace=True)
    summary_df.sort_values(by=['Hours'], inplace=True)

    print("Aggregated 3-candle swing-failure stats for 5-minute data by Hour (NY time of C0):")
    # Loaded candles count might not be accurate per hour after filtering, remove for now or get total
    # print(f"Loaded {len(df)} 5-minute candles")
    print(summary_df.to_string(index=False))

    # Restore stdout and get the captured output
    sys.stdout = old_stdout
    output = redirected_output.getvalue()
    print(output) # Print to console as before

    # Save output to file if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)

# --- 6. (Optional) Plotting Function ---
def plot_pattern_example(row, output_dir, filename_prefix="pattern"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = make_subplots(rows=1, cols=1)
    candles_to_plot = []
    # Need to adjust time spacing for 5min data
    time_delta = pd.Timedelta(minutes=5)
    for i in range(3): # c0, c1, c2
        c_data = {
            'open': row[f'c{i}_o'], 'high': row[f'c{i}_h'],
            'low': row[f'c{i}_l'], 'close': row[f'c{i}_c']
        }
        candles_to_plot.append(c_data)

    fig.add_trace(go.Candlestick(
        # Adjusted x-axis to space candles by 5 minutes
        x=[row['timestamp'] + time_delta * i for i in range(3)],
        open=[c['open'] for c in candles_to_plot],
        high=[c['high'] for c in candles_to_plot],
        low=[c['low'] for c in candles_to_plot],
        close=[c['close'] for c in candles_to_plot],
        name=f"{row['type']} SFP"
    ))
    fig.update_layout(
        title=f"{row['type'].capitalize()} Swing Failure Example (5min) - {row['timestamp']}",
        xaxis_title="Time", yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    ts_str = row['timestamp'].strftime('%Y-%m-%d_%H%M') # Use minute in filename for 5min data
    filepath = os.path.join(output_dir, f"{filename_prefix}_{row['type']}_{ts_str}.png")
    fig.write_image(filepath)

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data(DATA_FILE)

    # Get current parameter values for filename
    current_c0_c1_height_threshold = C0_C1_RELATIVE_HEIGHT_THRESHOLD
    current_c1_retracement_depth_threshold = C1_RETRACEMENT_DEPTH_THRESHOLD

    # Create a timestamped directory for this run's results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"5min_height{current_c0_c1_height_threshold}_depth{current_c1_retracement_depth_threshold}_{timestamp}"
    current_results_dir = os.path.join(RESULTS_BASE_DIR, run_dir_name)
    os.makedirs(current_results_dir, exist_ok=True)

    # Save configuration
    config_filepath = os.path.join(current_results_dir, "configuration.txt")
    with open(config_filepath, "w") as f:
        f.write(f"DATA_FILE: {DATA_FILE}\n")
        f.write(f"START_ANALYSIS_DATE: {START_ANALYSIS_DATE}\n")
        f.write(f"C0_C1_RELATIVE_HEIGHT_THRESHOLD: {current_c0_c1_height_threshold}\n")
        f.write(f"C1_RETRACEMENT_DEPTH_THRESHOLD: {current_c1_retracement_depth_threshold}\n")

    # Run swing failure analysis and save results
    swing_failure_results_filepath = os.path.join(current_results_dir, "swing_failure_stats.txt")
    pattern_results = analyze_swing_failures(df, current_c0_c1_height_threshold, current_c1_retracement_depth_threshold)
    aggregate_and_print_stats(pattern_results, swing_failure_results_filepath)

    # Save the pattern_results DataFrame to a file for later analysis
    pattern_results_filepath = os.path.join(current_results_dir, "raw_pattern_results_5min.csv")
    pattern_results.to_csv(pattern_results_filepath, index=False)
    print(f"Raw pattern results saved to ./{pattern_results_filepath}")

    # Optional: Plot some examples within the run-specific directory
    if not pattern_results.empty:
        plots_output_dir = os.path.join(current_results_dir, "plots")
        print(f"\nSaving example plots to ./{plots_output_dir}/ ...")
        # Plot a few bearish and bullish examples
        bearish_examples = pattern_results[pattern_results['type'] == 'bearish'].head(10)
        bullish_examples = pattern_results[pattern_results['type'] == 'bullish'].head(10)

        for idx, row in bearish_examples.iterrows():
            plot_pattern_example(row, plots_output_dir, "bearish_sfp_5min")
        for idx, row in bullish_examples.iterrows():
            plot_pattern_example(row, plots_output_dir, "bullish_sfp_5min")
        print("Example plots saved.") 