import pandas as pd
import numpy as np
import plotly.graph_objects as go # For plotting
from plotly.subplots import make_subplots # For plotting
import os # For creating plot directories
import datetime # For timestamping results
import sys # To capture stdout
import io # To capture stdout

# --- 1. Configuration ---
DATA_FILE = "frd_sample_futures_NQ/NQ_1hour_sample.csv"
OUTPUT_DIR = "plots_swing_failures"
RESULTS_BASE_DIR = "analysis_results"
START_ANALYSIS_DATE = "2010-01-01" # Or as desired (Note: Sample data is recent)
C0_C1_RELATIVE_HEIGHT_THRESHOLD = 0.9 # e.g., C1 height <= 0.9 * C0 height
C1_RETRACEMENT_DEPTH_THRESHOLD = 0.5 # e.g., C1 must not retrace more than 50% into C0 against expected direction
POST_C2_ANALYSIS_CANDLES = 5 # Number of candles to look at after C2 for R:R

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
    df['hour'] = df.index.hour
    df['c0_height'] = abs(df['open'] - df['close'])

    # Calculate net change over previous days
    # Assuming 24 hours per day for simplicity in hourly data
    df['prev_1d_change'] = df['close'].pct_change(periods=24) * 100
    df['prev_2d_change'] = df['close'].pct_change(periods=48) * 100
    df['prev_3d_change'] = df['close'].pct_change(periods=72) * 100

    # Drop rows with NaN values resulting from pct_change
    df.dropna(inplace=True)

    return df

# --- Helper function to categorize daily change ---
def categorize_change(change_pct):
    if change_pct < -1.0:
        return 'Strongly Down'
    elif -1.0 <= change_pct < -0.1:
        return 'Slightly Down'
    elif -0.1 <= change_pct <= 0.1:
        return 'Flat'
    elif 0.1 < change_pct <= 1.0:
        return 'Slightly Up'
    else:
        return 'Strongly Up'

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
def analyze_swing_failures(df, c0_c1_relative_height_threshold, c1_retracement_depth_threshold, post_c2_analysis_candles):
    results = [] # To store pattern occurrences and outcomes
    # Need 3 candles: c0, c1, c2, plus candles after c2 for R:R analysis
    for i in range(len(df) - 2 - post_c2_analysis_candles):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        c2 = df.iloc[i+2]
        # Get subsequent candles for R:R analysis
        subsequent_candles = df.iloc[i+3 : i+3+post_c2_analysis_candles]

        # Timestamps for context
        ts0, ts1, ts2 = df.index[i], df.index[i+1], df.index[i+2]

        # Get previous day's change category
        prev_1d_cat = categorize_change(df['prev_1d_change'].iloc[i])

        # --- Calculate R:R related metrics ---
        potential_stop_loss = None
        potential_reward_excursion = None
        hit_reward_target = False # Did price reach a simple target (e.g., C0 open)?
        hit_stop_loss = False # Did price hit the stop loss before the reward target?

        # Bearish SFP
        if is_bearish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['low'] <= c1['low']
            swept_first = c2['low'] <= c0['low']
            swept_open = c2['low'] <= c0['open']

            # Calculate potential Stop Loss (distance from C2 close to C1 high)
            potential_stop_loss = c1['high'] - c2['close'] if c1['high'] > c2['close'] else 0

            # Calculate potential Reward Excursion (max drop after C2)
            if not subsequent_candles.empty:
                min_low_after_c2 = subsequent_candles['low'].min()
                potential_reward_excursion = c2['close'] - min_low_after_c2 if c2['close'] > min_low_after_c2 else 0

                # Check if a simple target (e.g., C0 open) was hit before stop loss (C1 high)
                # This is a simplified check. A more accurate check would involve iterating candle by candle.
                target_price = c0['open']
                stop_loss_price = c1['high']

                # Check if target was hit
                if any(subsequent_candles['low'] <= target_price):
                     hit_reward_target = True

                # Check if stop loss was hit
                if any(subsequent_candles['high'] >= stop_loss_price):
                     hit_stop_loss = True

                # Refine hit_reward_target and hit_stop_loss by checking order of events
                # A more accurate check would involve iterating candle by candle and comparing hit times.
                # For now, a simplified approach: if both are hit in the window, check which *would* be hit first based on price levels
                # Or, as a conservative approach, assume stop loss if both are hit (or overlapping within the same candle)
                # The current boolean flags already represent if the levels were touched. To know which was hit *first* requires iterating.
                # Let's keep the simple boolean check for now and refine this if needed.

            results.append({
                'timestamp': ts0, 'type': 'bearish', 'hour_c0': c0['hour'],
                'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                'hit_reward_target': hit_reward_target,
                'hit_stop_loss': hit_stop_loss
            })

        # Bullish SFP
        if is_bullish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['high'] >= c1['high']
            swept_first = c2['high'] >= c0['high']
            swept_open = c2['high'] >= c0['open'] # Assuming target is C0 open for bullish too

            # Calculate potential Stop Loss (distance from C2 close to C1 low)
            potential_stop_loss = c2['close'] - c1['low'] if c2['close'] > c1['low'] else 0

            # Calculate potential Reward Excursion (max rise after C2)
            if not subsequent_candles.empty:
                max_high_after_c2 = subsequent_candles['high'].max()
                potential_reward_excursion = max_high_after_c2 - c2['close'] if max_high_after_c2 > c2['close'] else 0

                # Check if a simple target (e.g., C0 open) was hit before stop loss (C1 low)
                target_price = c0['open']
                stop_loss_price = c1['low']

                # Check if target was hit
                if any(subsequent_candles['high'] >= target_price):
                    hit_reward_target = True

                # Check if stop loss was hit
                if any(subsequent_candles['low'] <= stop_loss_price):
                    hit_stop_loss = True

                 # Refine hit_reward_target and hit_stop_loss by checking order of events
                 # A more accurate check would involve iterating candle by candle and comparing hit times.
                 # For now, a simplified approach: if both are hit in the window, check which *would* be hit first based on price levels
                 # Or, as a conservative approach, assume stop loss if both are hit (or overlapping within the same candle)
                 # The current boolean flags already represent if the levels were touched. To know which was hit *first* requires iterating.
                 # Let's keep the simple boolean check for now and refine this if needed.

            results.append({
                'timestamp': ts0, 'type': 'bullish', 'hour_c0': c0['hour'],
                'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                 'hit_reward_target': hit_reward_target,
                 'hit_stop_loss': hit_stop_loss
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
    # Aggregate by previous day's category and hour
    for category in ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']:
        for hour in range(24):
            hourly_patterns = results_df[(results_df['hour_c0'] == hour) & (results_df['prev_1d_cat'] == category)]

            bear_patterns = hourly_patterns[hourly_patterns['type'] == 'bearish']
            bull_patterns = hourly_patterns[hourly_patterns['type'] == 'bullish']

            bear_occurrences = len(bear_patterns)
            bull_occurrences = len(bull_patterns)

            if bear_occurrences > 0:
                total_bear_patterns_in_cat = len(results_df[(results_df['type'] == 'bearish') & (results_df['prev_1d_cat'] == category)])
                bear_hit_rate = (bear_occurrences / total_bear_patterns_in_cat) * 100 if total_bear_patterns_in_cat > 0 else 0
                bear_swept_mid_pct = (bear_patterns['swept_mid'].sum() / bear_occurrences) * 100
                bear_swept_first_pct = (bear_patterns['swept_first'].sum() / bear_occurrences) * 100
                bear_swept_open_pct = (bear_patterns['swept_open'].sum() / bear_occurrences) * 100

                # Calculate R:R metrics for bearish patterns
                bear_avg_sl = bear_patterns['potential_stop_loss'].mean() if bear_occurrences > 0 else 0
                bear_avg_re = bear_patterns['potential_reward_excursion'].mean() if bear_occurrences > 0 else 0
                bear_hit_target_pct = (bear_patterns['hit_reward_target'].sum() / bear_occurrences) * 100 if bear_occurrences > 0 else 0
                bear_hit_sl_pct = (bear_patterns['hit_stop_loss'].sum() / bear_occurrences) * 100 if bear_occurrences > 0 else 0

            else:
                bear_hit_rate, bear_swept_mid_pct, bear_swept_first_pct, bear_swept_open_pct = 0,0,0,0
                bear_avg_sl, bear_avg_re, bear_hit_target_pct, bear_hit_sl_pct = 0,0,0,0

            if bull_occurrences > 0:
                total_bull_patterns_in_cat = len(results_df[(results_df['type'] == 'bullish') & (results_df['prev_1d_cat'] == category)])
                bull_hit_rate = (bull_occurrences / total_bull_patterns_in_cat) * 100 if total_bull_patterns_in_cat > 0 else 0
                bull_swept_mid_pct = (bull_patterns['swept_mid'].sum() / bull_occurrences) * 100
                bull_swept_first_pct = (bull_patterns['swept_first'].sum() / bull_occurrences) * 100
                bull_swept_open_pct = (bull_patterns['swept_open'].sum() / bull_occurrences) * 100

                 # Calculate R:R metrics for bullish patterns
                bull_avg_sl = bull_patterns['potential_stop_loss'].mean() if bull_occurrences > 0 else 0
                bull_avg_re = bull_patterns['potential_reward_excursion'].mean() if bull_occurrences > 0 else 0
                bull_hit_target_pct = (bull_patterns['hit_reward_target'].sum() / bull_occurrences) * 100 if bull_occurrences > 0 else 0
                bull_hit_sl_pct = (bull_patterns['hit_stop_loss'].sum() / bull_occurrences) * 100 if bull_occurrences > 0 else 0

            else:
                bull_hit_rate, bull_swept_mid_pct, bull_swept_first_pct, bull_swept_open_pct = 0,0,0,0
                bull_avg_sl, bull_avg_re, bull_hit_target_pct, bull_hit_sl_pct = 0,0,0,0

            # Create the hour triplet string like the presenter
            h0 = hour
            h1 = (hour + 1) % 24
            h2 = (hour + 2) % 24
            hour_triplet_str = f"{h0:02d},{h1:02d},{h2:02d}"

            # Only add rows if there are occurrences in this category and hour
            if bear_occurrences > 0 or bull_occurrences > 0:
                stats_summary.append({
                    'Prev Day': category,
                    'Hours': hour_triplet_str,
                    'Bear Nr.': bear_occurrences,
                    'Hit%_Mid': f"{bear_swept_mid_pct:.2f}", # Renamed for clarity
                    'Hit%_1st': f"{bear_swept_first_pct:.2f}", # Renamed for clarity
                    'Hit%_Opn': f"{bear_swept_open_pct:.2f}", # Renamed for clarity
                    'Avg_SL': f"{bear_avg_sl:.2f}", # Added avg SL
                    'Avg_RE': f"{bear_avg_re:.2f}", # Added avg RE
                    'Hit_T%': f"{bear_hit_target_pct:.2f}", # Added Hit Target %
                    'Hit_SL%': f"{bear_hit_sl_pct:.2f}", # Added Hit SL %
                    '| Bull Nr.': bull_occurrences,
                    'Hit%_Mid_bull': f"{bull_swept_mid_pct:.2f}", # Renamed for clarity
                    'Hit%_1st_bull': f"{bull_swept_first_pct:.2f}", # Renamed for clarity
                    'Hit%_Opn_bull': f"{bull_swept_open_pct:.2f}", # Renamed for clarity
                    'Avg_SL_bull': f"{bull_avg_sl:.2f}", # Added avg SL
                    'Avg_RE_bull': f"{bull_avg_re:.2f}", # Added avg RE
                    'Hit_T%_bull': f"{bull_hit_target_pct:.2f}", # Added Hit Target %
                    'Hit_SL%_bull': f"{bull_hit_sl_pct:.2f}" # Added Hit SL %
                })

    summary_df = pd.DataFrame(stats_summary)
    # Sort by Prev Day category and then Hour
    category_order = ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']
    summary_df['Prev Day'] = pd.Categorical(summary_df['Prev Day'], categories=category_order, ordered=True)
    summary_df.sort_values(by=['Prev Day', 'Hours'], inplace=True)

    print("Aggregated 3-candle swing-failure stats by Previous Day's Distribution (NY time):")
    # Removed the total loaded candles print as it's less relevant with this aggregation
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
    for i in range(3): # c0, c1, c2
        c_data = {
            'open': row[f'c{i}_o'], 'high': row[f'c{i}_h'],
            'low': row[f'c{i}_l'], 'close': row[f'c{i}_c']
        }
        candles_to_plot.append(c_data)

    fig.add_trace(go.Candlestick(
        x=[row['timestamp'] + pd.Timedelta(hours=i) for i in range(3)], # Spaced out for viz
        open=[c['open'] for c in candles_to_plot],
        high=[c['high'] for c in candles_to_plot],
        low=[c['low'] for c in candles_to_plot],
        close=[c['close'] for c in candles_to_plot],
        name=f"{row['type']} SFP"
    ))
    fig.update_layout(
        title=f"{row['type'].capitalize()} Swing Failure Example - {row['timestamp']}",
        xaxis_title="Time", yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    ts_str = row['timestamp'].strftime('%Y-%m-%d_%H-%M')
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
    run_dir_name = f"height{current_c0_c1_height_threshold}_depth{current_c1_retracement_depth_threshold}_{timestamp}"
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
    pattern_results = analyze_swing_failures(df, current_c0_c1_height_threshold, current_c1_retracement_depth_threshold, POST_C2_ANALYSIS_CANDLES)
    aggregate_and_print_stats(pattern_results, swing_failure_results_filepath)

    # Save the pattern_results DataFrame to a file for later analysis
    pattern_results_filepath = os.path.join(current_results_dir, "raw_pattern_results.csv")
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
            plot_pattern_example(row, plots_output_dir, "bearish_sfp")
        for idx, row in bullish_examples.iterrows():
            plot_pattern_example(row, plots_output_dir, "bullish_sfp")
        print("Example plots saved.")

    # Note: Net Change Analysis results will need to be added to this directory separately or by modifying net_change_analyzer.py