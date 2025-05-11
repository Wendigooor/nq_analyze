import pandas as pd
import numpy as np

# --- Configuration ---
DATA_FILE = "frd_sample_futures_NQ/NQ_1hour_sample.csv"
# Define percentile range for outlier filtering
LOWER_PERCENTILE = 3
UPPER_PERCENTILE = 97

# --- Data Loading and Preprocessing ---
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
    df['hour'] = df.index.hour
    return df

# --- Net Change Analysis ---
def analyze_net_change(df):
    # Calculate net percentage change
    df['net_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100

    # Filter outliers based on percentiles
    lower_bound = np.percentile(df['net_change_pct'], LOWER_PERCENTILE)
    upper_bound = np.percentile(df['net_change_pct'], UPPER_PERCENTILE)
    df_filtered = df[(df['net_change_pct'] >= lower_bound) & (df['net_change_pct'] <= upper_bound)].copy()

    # Calculate and print overall statistics
    print("\n--- Overall Net Change Statistics (Filtered Outliers) ---")
    print(f"Number of candles analyzed (filtered): {len(df_filtered)}")
    print(f"Mean Net Change (%): {df_filtered['net_change_pct'].mean():.4f}")
    print(f"Median Net Change (%): {df_filtered['net_change_pct'].median():.4f}")
    print(f"Standard Deviation Net Change (%): {df_filtered['net_change_pct'].std():.4f}")

    # Calculate and print standard deviation thresholds
    mean_change = df_filtered['net_change_pct'].mean()
    std_change = df_filtered['net_change_pct'].std()
    print("\n--- Standard Deviation Thresholds ---")
    print(f"Mean +/- 0.5*Std: {mean_change - 0.5*std_change:.4f} % to {mean_change + 0.5*std_change:.4f} %")
    print(f"Mean +/- 1.0*Std: {mean_change - 1.0*std_change:.4f} % to {mean_change + 1.0*std_change:.4f} %")
    print(f"Mean +/- 1.5*Std: {mean_change - 1.5*std_change:.4f} % to {mean_change + 1.5*std_change:.4f} %")
    print(f"Mean +/- 2.0*Std: {mean_change - 2.0*std_change:.4f} % to {mean_change + 2.0*std_change:.4f} %")

    # Aggregate and print hourly statistics
    print("\n--- Hourly Net Change Statistics (Filtered Outliers) ---")
    hourly_stats = df_filtered.groupby('hour')['net_change_pct'].agg(['count', 'mean', 'median', 'std']).reset_index()
    hourly_stats.columns = ['Hour (NY Time)', 'Count', 'Mean (%)', 'Median (%)', 'Std Dev (%)']
    print(hourly_stats.to_string(index=False))

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data(DATA_FILE)
    analyze_net_change(df) 