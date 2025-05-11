import pandas as pd
import numpy as np
import plotly.graph_objects as go # For plotting
from plotly.subplots import make_subplots # For plotting
import os # For creating plot directories
import datetime # For timestamping results
import sys # To capture stdout
import io # To capture stdout

# --- 1. Configuration ---
DATA_FILE = "frd_sample_futures_NQ/NQ_30min_sample.csv"
OUTPUT_DIR = "plots_swing_failures_30min" # Updated output directory
RESULTS_BASE_DIR = "analysis_results_30min" # Updated results base directory
START_ANALYSIS_DATE = "2010-01-01" # Or as desired (Note: Sample data is recent)
# Parameters for Swing Failure Pattern definition (can be experimented with)
C0_C1_RELATIVE_HEIGHT_THRESHOLD = 0.9 # e.g., C1 height <= 0.9 * C0 height
C1_RETRACEMENT_DEPTH_THRESHOLD = 0.5 # e.g., C1 must not retrace more than 50% into C0 against expected direction
POST_C2_ANALYSIS_CANDLES = 10 # Number of candles to look at after C2 for R:R (adjusted for 30min timeframe)

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
    
    # Calculate features for 30-minute data
    df['hour'] = df.index.hour # Still useful for hourly grouping of 30min patterns
    df['c0_height'] = abs(df['open'] - df['close'])

    # Calculate net change over previous periods (assuming 48 periods per day for 30min data)
    # Commenting out previous day analysis for now in lower timeframes
    # df['prev_1d_change'] = df['close'].pct_change(periods=48) * 100 # 24 hours * 2 (30min in an hour)
    # df['prev_2d_change'] = df['close'].pct_change(periods=96) * 100 # 48 hours * 2
    # df['prev_3d_change'] = df['close'].pct_change(periods=144) * 100 # 72 hours * 2
    
    # if 'prev_1d_change' in df.columns: df.dropna(inplace=True) # Drop rows with NaN values if calculated

    return df

# --- Helper function to categorize daily change (from hourly, not directly used in 30min SFP analysis yet) ---
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

        # Previous day's category is not directly applicable to 30min patterns in the same way
        # We will not include prev_1d_cat in the 30min results for now.
        # prev_1d_cat = categorize_change(df['prev_1d_change'].iloc[i])

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
                target_price = c0['open']
                stop_loss_price = c1['high']

                # Simple check if levels were touched in subsequent candles
                target_touched = any(subsequent_candles['low'] <= target_price)
                sl_touched = any(subsequent_candles['high'] >= stop_loss_price)

                # More accurate check: iterate candle by candle to see which was hit first
                hit_target_first = False
                hit_sl_first = False

                if target_touched or sl_touched:
                    for k in range(len(subsequent_candles)):
                         current_candle = subsequent_candles.iloc[k]
                         # Check if SL is hit in current candle
                         if current_candle['high'] >= stop_loss_price:
                             hit_sl_first = True
                             break # SL hit first or at the same time as target in this candle
                         # Check if target is hit in current candle
                         if current_candle['low'] <= target_price:
                             hit_target_first = True
                             break # Target hit first

                hit_reward_target = hit_target_first
                hit_stop_loss = hit_sl_first

            # Calculate reward achieved and risk taken based on which level was hit first
            reward_achieved = 0.0
            risk_taken = 0.0
            if hit_reward_target:
                reward_achieved = abs(c2['close'] - target_price)
                risk_taken = abs(c2['close'] - stop_loss_price)
            elif hit_stop_loss:
                reward_achieved = 0.0 # No reward if SL hit first
                risk_taken = abs(c2['close'] - stop_loss_price)

            results.append({
                'timestamp': ts0, 'type': 'bearish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                'hit_reward_target': hit_reward_target,
                'hit_stop_loss': hit_stop_loss,
                'reward_achieved': reward_achieved,
                'risk_taken': risk_taken
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
                target_price = c0['open'] # Or c0['high'] or other levels
                stop_loss_price = c1['low']

                # Simple check if levels were touched in subsequent candles
                target_touched = any(subsequent_candles['high'] >= target_price)
                sl_touched = any(subsequent_candles['low'] <= stop_loss_price)

                # More accurate check: iterate candle by candle to see which was hit first
                hit_target_first = False
                hit_sl_first = False

                if target_touched or sl_touched:
                     for k in range(len(subsequent_candles)):
                          current_candle = subsequent_candles.iloc[k]
                          # Check if SL is hit in current candle
                          if current_candle['low'] <= stop_loss_price:
                              hit_sl_first = True
                              break # SL hit first or at the same time as target in this candle
                          # Check if target is hit in current candle
                          if current_candle['high'] >= target_price:
                               hit_target_first = True
                               break # Target hit first

                hit_reward_target = hit_target_first
                hit_stop_loss = hit_sl_first

            # Calculate reward achieved and risk taken based on which level was hit first
            reward_achieved = 0.0
            risk_taken = 0.0
            if hit_reward_target:
                reward_achieved = abs(target_price - c2['close'])
                risk_taken = abs(c2['close'] - stop_loss_price)
            elif hit_stop_loss:
                reward_achieved = 0.0 # No reward if SL hit first
                risk_taken = abs(c2['close'] - stop_loss_price)

            results.append({
                'timestamp': ts0, 'type': 'bullish', 'hour_c0': c0['hour'],
                # 'prev_1d_cat': prev_1d_cat,
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open,
                'potential_stop_loss': potential_stop_loss,
                'potential_reward_excursion': potential_reward_excursion,
                 'hit_reward_target': hit_reward_target,
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
    # Previous day's category aggregation is removed for 30min analysis for now
    # for category in ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']:
    for hour in range(24):
        # hourly_patterns = results_df[(results_df['hour_c0'] == hour) & (results_df['prev_1d_cat'] == category)]
        hourly_patterns = results_df[results_df['hour_c0'] == hour]

        bear_patterns = hourly_patterns[hourly_patterns['type'] == 'bearish']
        bull_patterns = hourly_patterns[hourly_patterns['type'] == 'bullish']

        bear_occurrences = len(bear_patterns)
        bull_occurrences = len(bull_patterns)

        if bear_occurrences > 0:
            # Hit rate as percentage of total bearish patterns found across all hours
            total_bear_patterns = len(results_df[results_df['type'] == 'bearish'])
            bear_hit_rate = (bear_occurrences / total_bear_patterns) * 100 if total_bear_patterns > 0 else 0 # This was occurrences percentage, not hit rate
            bear_swept_mid_pct = (bear_patterns['swept_mid'].sum() / bear_occurrences) * 100
            bear_swept_first_pct = (bear_patterns['swept_first'].sum() / bear_occurrences) * 100
            bear_swept_open_pct = (bear_patterns['swept_open'].sum() / bear_occurrences) * 100

            # R:R Stats for Bearish Patterns
            bear_hit_target_count = bear_patterns['hit_reward_target'].sum()
            bear_hit_stop_loss_count = bear_patterns['hit_stop_loss'].sum()
            bear_hit_target_pct = (bear_hit_target_count / bear_occurrences) * 100 if bear_occurrences > 0 else 0
            bear_hit_stop_loss_pct = (bear_hit_stop_loss_count / bear_occurrences) * 100 if bear_occurrences > 0 else 0
            # Calculate average R:R for patterns that hit target
            bear_avg_rr = (bear_patterns[bear_patterns['hit_reward_target'] == True]['reward_achieved'] / bear_patterns[bear_patterns['hit_reward_target'] == True]['risk_taken']).replace([np.inf, -np.inf], np.nan).mean()
            bear_avg_rr = 0 if pd.isna(bear_avg_rr) else bear_avg_rr # Handle NaN if no targets were hit

        else:
            bear_hit_rate, bear_swept_mid_pct, bear_swept_first_pct, bear_swept_open_pct = 0,0,0,0
            bear_hit_target_count, bear_hit_stop_loss_count = 0,0
            bear_avg_rr = 0

        if bull_occurrences > 0:
            # Hit rate as percentage of total bullish patterns found across all hours
            total_bull_patterns = len(results_df[results_df['type'] == 'bullish'])
            bull_hit_rate = (bull_occurrences / total_bull_patterns) * 100 if total_bull_patterns > 0 else 0 # This was occurrences percentage, not hit rate
            bull_swept_mid_pct = (bull_patterns['swept_mid'].sum() / bull_occurrences) * 100
            bull_swept_first_pct = (bull_patterns['swept_first'].sum() / bull_occurrences) * 100
            bull_swept_open_pct = (bull_patterns['swept_open'].sum() / bull_occurrences) * 100

             # R:R Stats for Bullish Patterns
            bull_hit_target_count = bull_patterns['hit_reward_target'].sum()
            bull_hit_stop_loss_count = bull_patterns['hit_stop_loss'].sum()
            bull_hit_target_pct = (bull_hit_target_count / bull_occurrences) * 100 if bull_occurrences > 0 else 0
            bull_hit_stop_loss_pct = (bull_hit_stop_loss_count / bull_occurrences) * 100 if bull_occurrences > 0 else 0
            # Calculate average R:R for patterns that hit target
            bull_avg_rr = (bull_patterns[bull_patterns['hit_reward_target'] == True]['reward_achieved'] / bull_patterns[bull_patterns['hit_reward_target'] == True]['risk_taken']).replace([np.inf, -np.inf], np.nan).mean()
            bull_avg_rr = 0 if pd.isna(bull_avg_rr) else bull_avg_rr # Handle NaN if no targets were hit

        else:
            bull_hit_rate, bull_swept_mid_pct, bull_swept_first_pct, bull_swept_open_pct = 0,0,0,0
            bull_hit_target_count, bull_hit_stop_loss_count = 0,0
            bull_avg_rr = 0

        # Create the hour triplet string - Note: this is still based on C0 hour, not a 30min interval triplet
        h0 = hour
        h1 = (hour + 1) % 24 # Still show the hour of C1 and C2 for context, even if it spans across 30min intervals
        h2 = (hour + 2) % 24
        hour_triplet_str = f"{h0:02d},{h1:02d},{h2:02d}"

        # Only add rows if there are occurrences in this hour
        if bear_occurrences > 0 or bull_occurrences > 0:
            stats_summary.append({
                # 'Prev Day': category, # Removed for 30min analysis
                'Hours': hour_triplet_str,
                'Bear Nr.': bear_occurrences,
                'Hit%_Mid': f"{bear_swept_mid_pct:.2f}", # Renamed for clarity
                'Hit%_1st': f"{bear_swept_first_pct:.2f}", # Renamed for clarity
                'Hit%_Opn': f"{bear_swept_open_pct:.2f}", # Renamed for clarity
                'Avg_SL': f"{bear_patterns['potential_stop_loss'].mean():.2f}", # Added avg SL
                'Avg_RE': f"{bear_patterns['potential_reward_excursion'].mean():.2f}", # Added avg RE
                'Hit_T%': f"{bear_hit_target_pct:.2f}", # Added Hit Target %
                'Hit_SL%': f"{bear_hit_stop_loss_pct:.2f}", # Added Hit SL %
                'Avg R:R': f"{bear_avg_rr:.2f}", # Added R:R stat
                '| Bull Nr.': bull_occurrences,
                'Hit%_Mid_bull': f"{bull_swept_mid_pct:.2f}", # Renamed for clarity
                'Hit%_1st_bull': f"{bull_swept_first_pct:.2f}", # Renamed for clarity
                'Hit%_Opn_bull': f"{bull_swept_open_pct:.2f}", # Renamed for clarity
                'Avg_SL_bull': f"{bull_patterns['potential_stop_loss'].mean():.2f}", # Added avg SL
                'Avg_RE_bull': f"{bull_patterns['potential_reward_excursion'].mean():.2f}", # Added avg RE
                'Hit_T%_bull': f"{bull_hit_target_pct:.2f}", # Added Hit Target %
                'Hit_SL%_bull': f"{bull_hit_stop_loss_pct:.2f}", # Added Hit SL %
                'Avg R:R_bull': f"{bull_avg_rr:.2f}" # Added R:R stat
            })

    summary_df = pd.DataFrame(stats_summary)
    # Sort by Hour
    # category_order = ['Strongly Down', 'Slightly Down', 'Flat', 'Slightly Up', 'Strongly Up']
    # summary_df['Prev Day'] = pd.Categorical(summary_df['Prev Day'], categories=category_order, ordered=True)
    # summary_df.sort_values(by=['Prev Day', 'Hours'], inplace=True)
    summary_df.sort_values(by=['Hours'], inplace=True)

    print("Aggregated 3-candle swing-failure stats for 30-minute data by Hour (NY time of C0):")
    # Loaded candles count might not be accurate per hour after filtering, remove for now or get total
    # print(f"Loaded {len(df)} 30-minute candles")
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
        textposition='top center', mode='markers+text', marker=dict(color='blue', size=10),
        name='C0'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c1_time], y=[row['c1_h'] + (row['c1_h'] - row['c1_l'])*0.1], mode='text', text=['C1'],
        textposition='top center', mode='markers+text', marker=dict(color='orange', size=10),
        name='C1'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[c2_time], y=[row['c2_h'] + (row['c2_h'] - row['c2_l'])*0.1], mode='text', text=['C2'],
        textposition='top center', mode='markers+text', marker=dict(color='green', size=10),
        name='C2'), row=1, col=1)

    # Add R:R lines if calculated
    if 'potential_stop_loss' in row and row['potential_stop_loss'] is not None:
        # For bearish, SL is above C1 high. For bullish, SL is below C1 low.
        sl_price = row['c1_h'] if row['type'] == 'bearish' else row['c1_l']
        sl_color = 'red' if row['hit_stop_loss'] else 'gray'
        fig.add_shape(type="line",
                      x0=c2_time,
                      y0=sl_price,
                      x1=df_plot.index[-1],
                      y1=sl_price,
                      line=dict(color=sl_color, width=1, dash='dash'),
                      name='Stop Loss')

        # For bearish, target is C0 low. For bullish, target is C0 high (or open).
        target_price = row['c0_l'] if row['type'] == 'bearish' else row['c0_open'] # Using C0 open as a simple target
        target_color = 'green' if row['hit_reward_target'] else 'gray'
        fig.add_shape(type="line",
                      x0=c2_time,
                      y0=target_price,
                      x1=df_plot.index[-1],
                      y1=target_price,
                      line=dict(color=target_color, width=1, dash='dash'),
                      name='Target')

    # Update layout
    pattern_type = row['type'].capitalize()
    hit_status = "Target Hit" if row['hit_reward_target'] else ("Stop Loss Hit" if row['hit_stop_loss'] else "No Hit - Outside Window")
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
    except Exception as e:
        print(f"Error saving plot: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data(DATA_FILE)

    # Get current parameter values for filename
    current_c0_c1_height_threshold = C0_C1_RELATIVE_HEIGHT_THRESHOLD
    current_c1_retracement_depth_threshold = C1_RETRACEMENT_DEPTH_THRESHOLD

    # Create a timestamped directory for this run's results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir_name = f"30min_height{current_c0_c1_height_threshold}_depth{current_c1_retracement_depth_threshold}_{timestamp}"
    current_results_dir = os.path.join(RESULTS_BASE_DIR, run_dir_name)
    os.makedirs(current_results_dir, exist_ok=True)

    # Save configuration
    config_filepath = os.path.join(current_results_dir, "configuration.txt")
    with open(config_filepath, "w") as f:
        f.write(f"DATA_FILE: {DATA_FILE}\n")
        f.write(f"START_ANALYSIS_DATE: {START_ANALYSIS_DATE}\n")
        f.write(f"C0_C1_RELATIVE_HEIGHT_THRESHOLD: {current_c0_c1_height_threshold}\n")
        f.write(f"C1_RETRACEMENT_DEPTH_THRESHOLD: {current_c1_retracement_depth_threshold}\n")
        f.write(f"POST_C2_ANALYSIS_CANDLES: {POST_C2_ANALYSIS_CANDLES}\n")

    # Run swing failure analysis and save results
    swing_failure_results_filepath = os.path.join(current_results_dir, "swing_failure_stats.txt")
    pattern_results = analyze_swing_failures(df, current_c0_c1_height_threshold, current_c1_retracement_depth_threshold, POST_C2_ANALYSIS_CANDLES)
    aggregate_and_print_stats(pattern_results, swing_failure_results_filepath)

    # Save the pattern_results DataFrame to a file for later analysis
    pattern_results_filepath = os.path.join(current_results_dir, "raw_pattern_results_30min.csv")
    pattern_results.to_csv(pattern_results_filepath, index=False)
    print(f"Raw pattern results saved to ./{pattern_results_filepath}")

    # Optional: Plot some examples within the run-specific directory
    if not pattern_results.empty:
        plots_output_dir = os.path.join(current_results_dir, "plots")
        print(f"\nSaving example plots to ./{plots_output_dir}/ ...")
        # Plot a few bearish and bullish examples
        bearish_examples = pattern_results[pattern_results['type'] == 'bearish'].head(5)
        bullish_examples = pattern_results[pattern_results['type'] == 'bullish'].head(5)

        for idx, row in bearish_examples.iterrows():
            plot_pattern_example(row, plots_output_dir, filename_prefix="bearish_pattern")
        for idx, row in bullish_examples.iterrows():
            plot_pattern_example(row, plots_output_dir, filename_prefix="bullish_pattern")
        print("Example plots saved.") 