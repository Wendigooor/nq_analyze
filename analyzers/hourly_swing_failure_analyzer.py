import pandas as pd
import numpy as np
import plotly.graph_objects as go # For plotting
from plotly.subplots import make_subplots # For plotting
import os # For creating plot directories

# --- 1. Configuration ---
DATA_FILE = "frd_sample_futures_NQ/NQ_1hour_sample.csv"
OUTPUT_DIR = "plots_swing_failures"
START_ANALYSIS_DATE = "2010-01-01" # Or as desired
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
    df['hour'] = df.index.hour
    df['c0_height'] = abs(df['open'] - df['close'])
    return df

# --- 3. Pattern Definition Functions ---
def is_bearish_swing_failure(c0, c1):
    """
    c0: pandas Series for the first candle
    c1: pandas Series for the second candle
    """
    if not (c0['close'] > c0['open']): return False # C0 must be bullish
    if not (c1['high'] > c0['high']): return False # C1 wicks above C0 high
    if not (c1['close'] < c0['high'] and c1['close'] > c0['low']): return False # C1 closes inside C0 range

    c0_height = abs(c0['open'] - c0['close'])
    c1_height = abs(c1['open'] - c1['close'])

    # Modify condition: C1 low must not go below C0 open
    if not (c1['low'] > c0['open']): return False

    # Modify condition: C1 height relative to C0 height (<= 90%)
    if not (c1_height <= 0.9 * c0_height): return False

    return True

def is_bullish_swing_failure(c0, c1):
    """
    c0: pandas Series for the first candle
    c1: pandas Series for the second candle
    """
    if not (c0['close'] < c0['open']): return False # C0 must be bearish
    if not (c1['low'] < c0['low']): return False # C1 wicks below C0 low
    if not (c1['close'] > c0['low'] and c1['close'] < c0['high']): return False # C1 closes inside C0 range

    c0_height = abs(c0['open'] - c0['close'])
    c1_height = abs(c1['open'] - c1['close'])

    # Modify condition: C1 high must not go above C0 open
    if not (c1['high'] < c0['open']): return False

    # Modify condition: C1 height relative to C0 height (<= 50%)
    if not (c1_height <= 0.5 * c0_height): return False
    return True

# --- 4. Analysis Loop ---
def analyze_swing_failures(df):
    results = [] # To store pattern occurrences and outcomes
    for i in range(len(df) - 2): # Need 3 candles: c0, c1, c2
        c0 = df.iloc[i]
        c1 = df.iloc[i+1]
        c2 = df.iloc[i+2]

        # Timestamps for context
        ts0, ts1, ts2 = df.index[i], df.index[i+1], df.index[i+2]

        # Bearish SFP
        if is_bearish_swing_failure(c0, c1):
            swept_mid = c2['low'] <= c1['low']
            swept_first = c2['low'] <= c0['low']
            swept_open = c2['low'] <= c0['open']
            results.append({
                'timestamp': ts0, 'type': 'bearish', 'hour_c0': c0['hour'],
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open
            })

        # Bullish SFP
        if is_bullish_swing_failure(c0, c1):
            swept_mid = c2['high'] >= c1['high']
            swept_first = c2['high'] >= c0['high']
            swept_open = c2['high'] >= c0['open'] # Assuming target is C0 open for bullish too
            results.append({
                'timestamp': ts0, 'type': 'bullish', 'hour_c0': c0['hour'],
                'c0_o':c0['open'], 'c0_h':c0['high'], 'c0_l':c0['low'], 'c0_c':c0['close'],
                'c1_o':c1['open'], 'c1_h':c1['high'], 'c1_l':c1['low'], 'c1_c':c1['close'],
                'c2_o':c2['open'], 'c2_h':c2['high'], 'c2_l':c2['low'], 'c2_c':c2['close'],
                'swept_mid': swept_mid, 'swept_first': swept_first, 'swept_open': swept_open
            })
    return pd.DataFrame(results)

# --- 5. Statistics Aggregation & Output ---
def aggregate_and_print_stats(results_df):
    if results_df.empty:
        print("No patterns found.")
        return

    # NY Time hours for aggregation (0-23)
    # The presenter's table shows triplets like 00,01,02. We'll use the hour of C0.
    stats_summary = []
    for hour in range(24):
        hourly_patterns = results_df[results_df['hour_c0'] == hour]

        bear_patterns = hourly_patterns[hourly_patterns['type'] == 'bearish']
        bull_patterns = hourly_patterns[hourly_patterns['type'] == 'bullish']

        bear_occurrences = len(bear_patterns)
        bull_occurrences = len(bull_patterns)

        if bear_occurrences > 0:
            bear_hit_rate = (bear_occurrences / len(results_df[results_df['hour_c0'] == hour])) * 100 # % of total patterns this hour
            bear_swept_mid_pct = (bear_patterns['swept_mid'].sum() / bear_occurrences) * 100
            bear_swept_first_pct = (bear_patterns['swept_first'].sum() / bear_occurrences) * 100
            bear_swept_open_pct = (bear_patterns['swept_open'].sum() / bear_occurrences) * 100
        else:
            bear_hit_rate, bear_swept_mid_pct, bear_swept_first_pct, bear_swept_open_pct = 0,0,0,0

        if bull_occurrences > 0:
            bull_hit_rate = (bull_occurrences / len(results_df[results_df['hour_c0'] == hour])) * 100
            bull_swept_mid_pct = (bull_patterns['swept_mid'].sum() / bull_occurrences) * 100
            bull_swept_first_pct = (bull_patterns['swept_first'].sum() / bull_occurrences) * 100
            bull_swept_open_pct = (bull_patterns['swept_open'].sum() / bull_occurrences) * 100
        else:
            bull_hit_rate, bull_swept_mid_pct, bull_swept_first_pct, bull_swept_open_pct = 0,0,0,0
        
        # Create the hour triplet string like the presenter
        h0 = hour
        h1 = (hour + 1) % 24
        h2 = (hour + 2) % 24
        hour_triplet_str = f"{h0:02d},{h1:02d},{h2:02d}"


        stats_summary.append({
            'Hours': hour_triplet_str,
            'Bear Nr.': f"{bear_occurrences}/{len(df[df['hour']==hour])}", # Occurrences / Total candles in that hour
            'Hit%': f"{bear_hit_rate:.2f}",
            'Mid%': f"{bear_swept_mid_pct:.2f}",
            '1st%': f"{bear_swept_first_pct:.2f}",
            'Opn%': f"{bear_swept_open_pct:.2f}",
            '| Bull Nr.': f"{bull_occurrences}/{len(df[df['hour']==hour])}",
            'Hit%_bull': f"{bull_hit_rate:.2f}", # Renamed to avoid duplicate column name
            'Mid%_bull': f"{bull_swept_mid_pct:.2f}",
            '1st%_bull': f"{bull_swept_first_pct:.2f}",
            'Opn%_bull': f"{bull_swept_open_pct:.2f}"
        })

    summary_df = pd.DataFrame(stats_summary)
    print("Aggregated 3-candle swing-failure stats for all hour triples (NY time):")
    print(f"Loaded {len(df)} hourly candles")
    print(summary_df.to_string(index=False))

# --- 6. (Optional) Plotting Function ---
def plot_pattern_example(row, filename_prefix="pattern"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
    filepath = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{row['type']}_{ts_str}.png")
    fig.write_image(filepath)
    # print(f"Saved plot: {filepath}")


# --- Main Execution ---
if __name__ == "__main__":
    df = load_data(DATA_FILE)
    pattern_results = analyze_swing_failures(df)
    aggregate_and_print_stats(pattern_results)

    # Optional: Plot some examples
    if not pattern_results.empty:
        print(f"\nSaving example plots to ./{OUTPUT_DIR}/ ...")
        # Plot a few bearish and bullish examples
        bearish_examples = pattern_results[pattern_results['type'] == 'bearish'].head(5)
        bullish_examples = pattern_results[pattern_results['type'] == 'bullish'].head(5)

        for idx, row in bearish_examples.iterrows():
            plot_pattern_example(row, "bearish_sfp")
        for idx, row in bullish_examples.iterrows():
            plot_pattern_example(row, "bullish_sfp")
        print("Example plots saved.")