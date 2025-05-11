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
POST_C2_ANALYSIS_CANDLES = 180 # Number of candles after C2 to analyze for Stop Loss/Reward Excursion (e.g., 15 hours for 5min)

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
    # Need at least C0, C1, C2, and then candles for post-C2 analysis
    for i in range(len(df) - 2 - POST_C2_ANALYSIS_CANDLES):
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        c2 = df.iloc[i+2]

        # Timestamps for context
        ts0, ts1, ts2 = df.index[i], df.index[i+1], df.index[i+2]

        # Previous day's category is not directly applicable to 5min patterns in the same way
        # We will not include prev_1d_cat in the 5min results for now.
        # prev_1d_cat = categorize_change(df['prev_1d_change'].iloc[i])

        # Initialize R:R variables
        potential_stop_loss = None
        potential_reward_excursion = None
        hit_target = False
        hit_stop_loss = False
        reward_achieved = 0.0
        risk_taken = 0.0

        # Bearish SFP
        if is_bearish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['low'] <= c1['low']
            swept_first = c2['low'] <= c0['low']
            swept_open = c2['low'] <= c0['open']

            # R:R Calculation for Bearish Pattern
            potential_stop_loss = c1['high'] # Stop loss typically above C1 high
            # Reward excursion could be based on distance from C2 close to C0 low, for example
            potential_reward_excursion = c2['close'] - c0['low']

            # Analyze candles after C2 for Hit Target/Stop Loss
            # Check if price goes below (C2 close - Potential Reward Excursion) or above Potential Stop Loss
            target_price = c2['close'] - potential_reward_excursion
            stop_loss_price = potential_stop_loss

            for j in range(i + 3, min(i + 3 + POST_C2_ANALYSIS_CANDLES, len(df))):
                current_candle = df.iloc[j]
                if current_candle['low'] <= target_price:
                    hit_target = True
                    reward_achieved = c2['close'] - target_price # Reward is the price difference
                    risk_taken = abs(c2['close'] - stop_loss_price) # Risk is the distance to SL
                    break # Exit loop once target is hit
                if current_candle['high'] >= stop_loss_price:
                    hit_stop_loss = True
                    reward_achieved = 0.0 # No reward if SL is hit first
                    risk_taken = abs(c2['close'] - stop_loss_price) # Risk is the distance to SL
                    break # Exit loop once SL is hit

            results.append({
                'timestamp': ts0, 'type': 'bearish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                'hit_target': hit_target,
                'hit_stop_loss': hit_stop_loss,
                'reward_achieved': reward_achieved,
                'risk_taken': risk_taken
            })

        # Bullish SFP
        if is_bullish_swing_failure(c0, c1, c0_c1_relative_height_threshold, c1_retracement_depth_threshold):
            swept_mid = c2['high'] >= c1['high']
            swept_first = c2['high'] >= c0['high']
            swept_open = c2['high'] >= c0['open'] # Assuming target is C0 open for bullish too

            # R:R Calculation for Bullish Pattern
            potential_stop_loss = c1['low'] # Stop loss typically below C1 low
            # Reward excursion could be based on distance from C2 close to C0 high, for example
            potential_reward_excursion = c0['high'] - c2['close']

            # Analyze candles after C2 for Hit Target/Stop Loss
            # Check if price goes above (C2 close + Potential Reward Excursion) or below Potential Stop Loss
            target_price = c2['close'] + potential_reward_excursion
            stop_loss_price = potential_stop_loss

            for j in range(i + 3, min(i + 3 + POST_C2_ANALYSIS_CANDLES, len(df))):
                current_candle = df.iloc[j]
                if current_candle['high'] >= target_price:
                    hit_target = True
                    reward_achieved = target_price - c2['close'] # Reward is the price difference
                    risk_taken = abs(c2['close'] - stop_loss_price) # Risk is the distance to SL
                    break # Exit loop once target is hit
                if current_candle['low'] <= stop_loss_price:
                    hit_stop_loss = True
                    reward_achieved = 0.0 # No reward if SL is hit first
                    risk_taken = abs(c2['close'] - stop_loss_price) # Risk is the distance to SL
                    break # Exit loop once SL is hit

            results.append({
                'timestamp': ts0, 'type': 'bullish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                'hit_target': hit_target,
                'hit_stop_loss': hit_stop_loss,
                'reward_achieved': reward_achieved,
                'risk_taken': risk_taken
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

            # R:R Stats for Bearish Patterns
            bear_hit_target_count = bear_patterns['hit_target'].sum()
            bear_hit_stop_loss_count = bear_patterns['hit_stop_loss'].sum()
            bear_hit_target_pct = (bear_hit_target_count / bear_occurrences) * 100 if bear_occurrences > 0 else 0
            bear_hit_stop_loss_pct = (bear_hit_stop_loss_count / bear_occurrences) * 100 if bear_occurrences > 0 else 0
            bear_avg_rr = (bear_patterns['reward_achieved'] / bear_patterns['risk_taken']).replace([np.inf, -np.inf], np.nan).mean() if bear_hit_target_count > 0 else 0

        else:
            bear_hit_rate, bear_swept_mid_pct, bear_swept_first_pct, bear_swept_open_pct = 0,0,0,0
            bear_hit_target_pct, bear_hit_stop_loss_pct, bear_avg_rr = 0,0,0

        if bull_occurrences > 0:
            # Hit rate as percentage of total bullish patterns found across all hours
            total_bull_patterns = len(results_df[results_df['type'] == 'bullish'])
            bull_hit_rate = (bull_occurrences / total_bull_patterns) * 100 if total_bull_patterns > 0 else 0
            bull_swept_mid_pct = (bull_patterns['swept_mid'].sum() / bull_occurrences) * 100
            bull_swept_first_pct = (bull_patterns['swept_first'].sum() / bull_occurrences) * 100
            bull_swept_open_pct = (bull_patterns['swept_open'].sum() / bull_occurrences) * 100

            # R:R Stats for Bullish Patterns
            bull_hit_target_count = bull_patterns['hit_target'].sum()
            bull_hit_stop_loss_count = bull_patterns['hit_stop_loss'].sum()
            bull_hit_target_pct = (bull_hit_target_count / bull_occurrences) * 100 if bull_occurrences > 0 else 0
            bull_hit_stop_loss_pct = (bull_hit_stop_loss_count / bull_occurrences) * 100 if bull_occurrences > 0 else 0
            bull_avg_rr = (bull_patterns['reward_achieved'] / bull_patterns['risk_taken']).replace([np.inf, -np.inf], np.nan).mean() if bull_hit_target_count > 0 else 0

        else:
            bull_hit_rate, bull_swept_mid_pct, bull_swept_first_pct, bull_swept_open_pct = 0,0,0,0
            bull_hit_target_pct, bull_hit_stop_loss_pct, bull_avg_rr = 0,0,0

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
                'Target%': f"{bear_hit_target_pct:.2f}",
                'Stop%': f"{bear_hit_stop_loss_pct:.2f}",
                'Avg R:R': f"{bear_avg_rr:.2f}",
                '| Bull Nr.': bull_occurrences,
                'Hit%_bull': f"{bull_hit_rate:.2f}",
                'Mid%_bull': f"{bull_swept_mid_pct:.2f}",
                '1st%_bull': f"{bull_swept_first_pct:.2f}",
                'Opn%_bull': f"{bull_swept_open_pct:.2f}",
                'Target%_bull': f"{bull_hit_target_pct:.2f}",
                'Stop%_bull': f"{bull_hit_stop_loss_pct:.2f}",
                'Avg R:R_bull': f"{bull_avg_rr:.2f}"
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

    # Save raw results to CSV
    raw_results_dir = os.path.join(RESULTS_BASE_DIR, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(raw_results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(raw_results_dir, 'raw_patterns_5min.csv'), index=False)

# --- 6. (Optional) Plotting Function ---
def plot_pattern_example(row, output_dir, filename_prefix="pattern"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load full data for plotting context (adjust as needed)
    df_full = load_data(DATA_FILE) # Load the full data for plotting context

    # Find the index of the pattern in the full dataframe
    try:
        idx = df_full.index.get_loc(row['timestamp'])
    except KeyError:
        print(f"Timestamp {row['timestamp']} not found in full data for plotting.")
        return

    # Define the window for plotting (e.g., 20 candles before to 30 candles after C0)
    plot_window_start = max(0, idx - 20)
    plot_window_end = min(len(df_full) - 1, idx + 2 + POST_C2_ANALYSIS_CANDLES + 10) # Extend window to show R:R analysis
    df_plot = df_full.iloc[plot_window_start : plot_window_end + 1].copy()

    # Create subplot with 1 row and 1 column
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "candlestick"}]])

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Candlesticks'
    ), row=1, col=1)

    # Add markers for C0, C1, C2
    c0_time = row['timestamp']
    c1_time = df_full.index[idx + 1]
    c2_time = df_full.index[idx + 2]

    fig.add_trace(go.Scatter(
        x=[c0_time], y=[row['c0_h'] + (row['c0_h'] - row['c0_l'])*0.1], mode='text', text=['C0'],
        textposition='top center', marker=dict(color='blue', size=10),
        name='C0'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c1_time], y=[row['c1_h'] + (row['c1_h'] - row['c1_l'])*0.1], mode='text', text=['C1'],
        textposition='top center', marker=dict(color='orange', size=10),
        name='C1'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c2_time], y=[row['c2_h'] + (row['c2_h'] - row['c2_l'])*0.1], mode='text', text=['C2'],
        textposition='top center', marker=dict(color='green', size=10),
        name='C2'), row=1, col=1)

    # Add R:R lines if calculated
    if row['potential_stop_loss'] is not None:
        # Stop Loss Line (horizontal from C2 close)
        sl_price = row['potential_stop_loss']
        sl_color = 'red' if row['hit_stop_loss'] else 'gray'
        fig.add_shape(type="line",
                      x0=c2_time,
                      y0=sl_price,
                      x1=df_plot.index[-1],
                      y1=sl_price,
                      line=dict(color=sl_color, width=1, dash='dash'),
                      name='Stop Loss')

        # Target Line (horizontal from C2 close)
        # Target price is relative to C2 close and potential_reward_excursion
        if row['type'] == 'bearish':
            target_price = row['c2_c'] - row['potential_reward_excursion']
        else:
            target_price = row['c2_c'] + row['potential_reward_excursion']

        target_color = 'green' if row['hit_target'] else 'gray'
        fig.add_shape(type="line",
                      x0=c2_time,
                      y0=target_price,
                      x1=df_plot.index[-1],
                      y1=target_price,
                      line=dict(color=target_color, width=1, dash='dash'),
                      name='Target')

    # Update layout
    pattern_type = row['type'].capitalize()
    hit_status = "Target Hit" if row['hit_target'] else ("Stop Loss Hit" if row['hit_stop_loss'] else "No Hit")
    rr_ratio = (row['reward_achieved'] / row['risk_taken']) if row['risk_taken'] > 0 else 0

    fig.update_layout(
        title=f'{pattern_type} Swing Failure Pattern at {c0_time}<br>Hit Status: {hit_status} (R:R = {rr_ratio:.2f})',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    # Save the plot
    filename = os.path.join(output_dir, f'{filename_prefix}_{c0_time.strftime('%Y%m%d_%H%M%S')}.png')
    try:
        fig.write_image(filename)
        # print(f"Saved plot to {filename}")
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading data from {DATA_FILE}...")
    df = load_data(DATA_FILE)
    print(f"Data loaded. Shape: {df.shape}")

    print("Analyzing for Swing Failure Patterns...")
    # Pass thresholds to analysis function
    pattern_results = analyze_swing_failures(df, C0_C1_RELATIVE_HEIGHT_THRESHOLD, C1_RETRACEMENT_DEPTH_THRESHOLD)
    print(f"Analysis complete. Found {len(pattern_results)} patterns.")

    # Timestamp the results directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_results_dir = os.path.join(RESULTS_BASE_DIR, timestamp)
    os.makedirs(current_results_dir, exist_ok=True)
    stats_output_file = os.path.join(current_results_dir, 'stats_summary_5min.txt')

    print("Aggregating and printing statistics...")
    # Pass original df to aggregate_and_print_stats if needed for per-hour total candle count (currently removed)
    aggregate_and_print_stats(pattern_results, output_file=stats_output_file)
    print(f"Statistics saved to {stats_output_file}")

    print("Generating example plots...")
    # Create a dedicated directory for example plots within the timestamped results
    plots_output_dir = os.path.join(current_results_dir, OUTPUT_DIR)
    os.makedirs(plots_output_dir, exist_ok=True)

    # Plot a few examples (e.g., first 5 bullish and first 5 bearish if available)
    bearish_examples = pattern_results[pattern_results['type'] == 'bearish'].head(5)
    bullish_examples = pattern_results[pattern_results['type'] == 'bullish'].head(5)

    for idx, row in bearish_examples.iterrows():
        plot_pattern_example(row, plots_output_dir, filename_prefix="bearish_pattern")

    for idx, row in bullish_examples.iterrows():
        plot_pattern_example(row, plots_output_dir, filename_prefix="bullish_pattern")

    print(f"Example plots saved to {plots_output_dir}") 