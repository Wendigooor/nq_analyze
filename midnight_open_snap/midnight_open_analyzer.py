#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, time
import pytz

# Configuration
DATA_DIR = "../tradingview_futures_NQ"
DATA_FILE_1H = os.path.join(DATA_DIR, "CME_MINI_DL_NQ1!, 60, 30 apr 2021 - 11 May 2025.csv")
DATA_FILE_5M = os.path.join(DATA_DIR, "CME_MINI_DL_NQ1!, 60, 30 apr 2021 - 11 May 2025.csv")  # Using hourly data as fallback
DATA_FILE_30M = os.path.join(DATA_DIR, "CME_MINI_DL_NQ1!, 60, 30 apr 2021 - 11 May 2025.csv")  # Using hourly data as fallback
RESULTS_DIR = "results"
WINDOW_START = "08:30"
WINDOW_END = "12:00"
TOLERANCE_POINTS = 10  # Acceptable deviation from TMO to determine touch
GAP_THRESHOLD_PCT = 0.5  # Minimum gap size in % for gap fill analysis
GAP_FILL_PCT = 50  # Percentage of gap filling to count in statistics

# Create directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
result_dir = os.path.join(RESULTS_DIR, f"results_{timestamp}")
os.makedirs(result_dir, exist_ok=True)

def load_data(filepath):
    """Load data and convert timezone to America/New_York"""
    try:
        df = pd.read_csv(filepath)
        
        # Check for TradingView format
        if 'time' in df.columns:
            # TradingView data format
            # Convert Unix timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            
            # Normalize column names
            rename_cols = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close'
            }
            
            # Rename columns that exist in the dataframe
            existing_cols = set(df.columns).intersection(set(rename_cols.keys()))
            df = df.rename(columns={col: rename_cols[col] for col in existing_cols})
            
            # Set the timestamp as index
            df.set_index('timestamp', inplace=True)
            
        else:
            # Fallback for other formats
            # Check time column format
            time_column = 'Timestamp' if 'Timestamp' in df.columns else 'timestamp'
            
            # Convert time column to timezone-aware datetime with NY timezone
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Set index and normalize column names
            df.set_index(time_column, inplace=True)
            
            # Normalize column names (convert to lowercase)
            df.columns = [col.lower() for col in df.columns]
        
        # If data doesn't have timezone info, assume it's UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        # Convert to America/New_York
        df.index = df.index.tz_convert('America/New_York')
        
        # Ensure OHLC columns are present
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"File doesn't contain required OHLC columns: {required_cols}")
            
        return df[list(required_cols)]
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def identify_tmo_levels(df_1h):
    """Identify TMO (True Midnight Open) levels for each day"""
    # Select all bars opening at 00:00 NY time
    tmo_df = df_1h[df_1h.index.map(lambda x: x.time() == time(0, 0))]
    
    # Create Series with date as index and open price as value
    tmo_series = pd.Series(tmo_df['open'].values, index=tmo_df.index.date, name='tmo')
    
    # Convert Series to DataFrame for convenience
    tmo_data = tmo_series.reset_index()
    tmo_data.columns = ['date', 'tmo']
    
    # Save TMO levels to CSV
    tmo_data.to_csv(os.path.join(result_dir, 'tmo_levels.csv'), index=False)
    
    print(f"Found {len(tmo_data)} TMO levels")
    return tmo_data

def analyze_touches_by_time(df_5m, tmo_data):
    """Analyze TMO touches broken down by minute intervals (0, 15, 30, 45)"""
    touches_by_minute = {0: 0, 15: 0, 30: 0, 45: 0}
    total_by_minute = {0: 0, 15: 0, 30: 0, 45: 0}
    
    # Create dataframe to store results
    results = []
    
    # Analyze each day with TMO
    for _, row in tmo_data.iterrows():
        date = row['date']
        tmo_level = row['tmo']
        
        # Select data for this day within analysis window
        day_data = df_5m[df_5m.index.date == date]
        window_data = day_data.between_time(WINDOW_START, WINDOW_END)
        
        # For each 5-minute candle, check if there was a touch of TMO
        for idx, candle in window_data.iterrows():
            minute = idx.minute
            minute_group = (minute // 15) * 15  # Group by 15-minute intervals (0, 15, 30, 45)
            
            # Increment counter for corresponding minute interval
            total_by_minute[minute_group] += 1
            
            # Check touch
            touched = (candle['low'] <= tmo_level + TOLERANCE_POINTS) and (candle['high'] >= tmo_level - TOLERANCE_POINTS)
            
            if touched:
                touches_by_minute[minute_group] += 1
                results.append({
                    'date': date,
                    'timestamp': idx,
                    'weekday': idx.day_name(),
                    'minute': minute,
                    'minute_group': minute_group,
                    'tmo': tmo_level,
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'touch': True
                })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    if not results_df.empty:
        results_df.to_csv(os.path.join(result_dir, 'touches_by_minute.csv'), index=False)
    
    # Calculate touch percentages for each minute interval
    minute_stats = {}
    for minute in touches_by_minute.keys():
        total = total_by_minute[minute]
        touches = touches_by_minute[minute]
        pct = (touches / total * 100) if total > 0 else 0
        minute_stats[minute] = {
            'total': total,
            'touches': touches,
            'percentage': pct
        }
    
    return minute_stats, results_df

def analyze_touches_by_hour(df_5m, tmo_data):
    """Analyze TMO touches broken down by hour in the analysis window"""
    # Define market open period (8:00-10:30 AM)
    MARKET_OPEN_START = "08:00"
    MARKET_OPEN_END = "10:30"
    
    # Dictionary to store touches by hour
    hours = [8, 9, 10, 11]
    touches_by_hour = {h: 0 for h in hours}
    total_by_hour = {h: 0 for h in hours}
    
    # For market open period specifically
    market_open_touches = 0
    market_open_total = 0
    
    # Create dataframe to store results
    results = []
    
    # Analyze each day with TMO
    for _, row in tmo_data.iterrows():
        date = row['date']
        tmo_level = row['tmo']
        
        # Select data for this day within analysis window
        day_data = df_5m[df_5m.index.date == date]
        window_data = day_data.between_time(WINDOW_START, WINDOW_END)
        
        # For market open specific window
        market_open_data = day_data.between_time(MARKET_OPEN_START, MARKET_OPEN_END)
        
        # Check if there was a touch in the market open period
        market_open_touch = False
        if len(market_open_data) > 0:
            market_open_total += 1
            market_open_touch = any((candle['low'] <= tmo_level + TOLERANCE_POINTS) and 
                                   (candle['high'] >= tmo_level - TOLERANCE_POINTS) 
                                   for _, candle in market_open_data.iterrows())
            if market_open_touch:
                market_open_touches += 1
        
        # For each candle, check hour and if there was a touch of TMO
        for idx, candle in window_data.iterrows():
            hour = idx.hour
            
            # Only count hours in our analysis window
            if hour in hours:
                # Increment counter for corresponding hour
                total_by_hour[hour] += 1
                
                # Check touch
                touched = (candle['low'] <= tmo_level + TOLERANCE_POINTS) and (candle['high'] >= tmo_level - TOLERANCE_POINTS)
                
                if touched:
                    touches_by_hour[hour] += 1
                    results.append({
                        'date': date,
                        'timestamp': idx,
                        'weekday': idx.day_name(),
                        'hour': hour,
                        'tmo': tmo_level,
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'touch': True
                    })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    if not results_df.empty:
        results_df.to_csv(os.path.join(result_dir, 'touches_by_hour.csv'), index=False)
    
    # Calculate touch percentages for each hour
    hour_stats = {}
    for hour in hours:
        total = total_by_hour[hour]
        touches = touches_by_hour[hour]
        pct = (touches / total * 100) if total > 0 else 0
        hour_stats[hour] = {
            'total': total,
            'touches': touches,
            'percentage': pct
        }
    
    # Calculate market open period statistics
    market_open_pct = (market_open_touches / market_open_total * 100) if market_open_total > 0 else 0
    market_open_stats = {
        'total': market_open_total,
        'touches': market_open_touches,
        'percentage': market_open_pct
    }
    
    return hour_stats, market_open_stats, results_df

def analyze_midnight_snap(df_1h, df_5m, df_30m, tmo_data):
    """Main analysis of Midnight Open Snap pattern"""
    results = []
    snap_back_cases = []
    gap_fill_cases = []
    
    # Analyze each day with TMO
    for _, row in tmo_data.iterrows():
        date = row['date']
        tmo_level = row['tmo']
        
        # Data for analysis
        day_df_1h = df_1h[df_1h.index.date == date]
        day_df_5m = df_5m[df_5m.index.date == date]
        day_df_30m = df_30m[df_30m.index.date == date]
        
        # Analysis windows for different timeframes
        window_1h = day_df_1h.between_time(WINDOW_START, WINDOW_END)
        window_5m = day_df_5m.between_time(WINDOW_START, WINDOW_END)
        window_30m = day_df_30m.between_time(WINDOW_START, WINDOW_END)
        
        # Check for data availability
        if len(window_1h) == 0 or len(window_5m) == 0:
            continue
            
        # Check for TMO touches
        touches_1h = (window_1h['low'] <= tmo_level + TOLERANCE_POINTS) & (window_1h['high'] >= tmo_level - TOLERANCE_POINTS)
        touches_5m = (window_5m['low'] <= tmo_level + TOLERANCE_POINTS) & (window_5m['high'] >= tmo_level - TOLERANCE_POINTS)
        
        # Check for touch in hourly timeframe
        touched_1h = touches_1h.any()
        
        # Information about first touch (if any)
        first_touch_time = None
        first_touch_timeframe = None
        
        if touched_1h and touches_1h.sum() > 0:
            first_touch_time = window_1h[touches_1h].index[0]
            first_touch_timeframe = "1H"
        elif touches_5m.any():
            first_touch_time = window_5m[touches_5m].index[0]
            first_touch_timeframe = "5M"
        
        # Check snap back (false breakout with return)
        snap_back = False
        if first_touch_time is not None:
            # Analyze 5-minute candles after first touch
            post_touch = window_5m[window_5m.index > first_touch_time].head(3)
            
            if len(post_touch) >= 2:
                # Check if there was a false breakout and return
                broke_above = (post_touch['high'].iloc[0] > tmo_level + TOLERANCE_POINTS)
                broke_below = (post_touch['low'].iloc[0] < tmo_level - TOLERANCE_POINTS)
                
                returned_above = broke_below and (post_touch['high'].iloc[1] >= tmo_level - TOLERANCE_POINTS)
                returned_below = broke_above and (post_touch['low'].iloc[1] <= tmo_level + TOLERANCE_POINTS)
                
                snap_back = returned_above or returned_below
                
                if snap_back:
                    snap_back_cases.append({
                        'date': date,
                        'touch_time': first_touch_time,
                        'weekday': first_touch_time.day_name(),
                        'tmo': tmo_level,
                        'direction': 'up' if returned_above else 'down'
                    })
        
        # Analyze overnight gap and gap fill
        if len(day_df_1h) > 0:
            # Get previous day
            prev_date = pd.Timestamp(date) - pd.Timedelta(days=1)
            prev_df = df_1h[df_1h.index.date == prev_date.date()]
            
            if len(prev_df) > 0:
                # Get previous day's close and current day's open
                prev_close = prev_df['close'].iloc[-1]
                current_open = day_df_1h['open'].iloc[0]
                
                # Calculate gap size in percent
                gap_size_pct = abs(current_open - prev_close) / prev_close * 100
                
                # If gap is large enough for analysis
                if gap_size_pct >= GAP_THRESHOLD_PCT:
                    # Determine gap direction
                    gap_direction = 'up' if current_open > prev_close else 'down'
                    
                    # Check if gap was filled during the day
                    if gap_direction == 'up':
                        # For gap up, check if price went down to prev_close level
                        gap_fill_level = prev_close + (current_open - prev_close) * (1 - GAP_FILL_PCT/100)
                        gap_filled = (day_df_1h['low'].min() <= gap_fill_level)
                    else:
                        # For gap down, check if price went up to prev_close level
                        gap_fill_level = prev_close - (prev_close - current_open) * (1 - GAP_FILL_PCT/100)
                        gap_filled = (day_df_1h['high'].max() >= gap_fill_level)
                    
                    gap_fill_cases.append({
                        'date': date,
                        'weekday': pd.Timestamp(date).day_name(),
                        'prev_close': prev_close,
                        'open': current_open,
                        'gap_size_pct': gap_size_pct,
                        'gap_direction': gap_direction,
                        'gap_filled': gap_filled,
                        'gap_fill_level': gap_fill_level
                    })
        
        # Add result for current day
        results.append({
            'date': date,
            'weekday': first_touch_time.day_name() if first_touch_time else pd.Timestamp(date).day_name(),
            'touch': first_touch_time is not None,
            'touch_time': first_touch_time,
            'touch_timeframe': first_touch_timeframe,
            'snap_back': snap_back,
            'tmo': tmo_level
        })
    
    # Create DataFrames with results
    results_df = pd.DataFrame(results)
    snap_back_df = pd.DataFrame(snap_back_cases) if snap_back_cases else pd.DataFrame()
    gap_fill_df = pd.DataFrame(gap_fill_cases) if gap_fill_cases else pd.DataFrame()
    
    # Save results to CSV
    if not results_df.empty:
        results_df.to_csv(os.path.join(result_dir, 'midnight_snap_results.csv'), index=False)
    
    if not snap_back_df.empty:
        snap_back_df.to_csv(os.path.join(result_dir, 'snap_back_cases.csv'), index=False)
    
    if not gap_fill_df.empty:
        gap_fill_df.to_csv(os.path.join(result_dir, 'gap_fill_cases.csv'), index=False)
    
    return results_df, snap_back_df, gap_fill_df

def aggregate_statistics(results_df, snap_back_df=None, gap_fill_df=None):
    """Aggregate statistics by day of week and create summary table"""
    if results_df.empty:
        print("No data for statistics aggregation")
        return None
    
    # Aggregate by day of week
    weekday_stats = results_df.groupby('weekday').agg(
        total_days=('date', 'count'),
        touch_count=('touch', 'sum'),
        snap_back_count=('snap_back', 'sum')
    ).reset_index()
    
    # Calculate percentages
    weekday_stats['touch_percentage'] = weekday_stats['touch_count'] / weekday_stats['total_days'] * 100
    weekday_stats['snap_back_percentage'] = weekday_stats['snap_back_count'] / weekday_stats['touch_count'] * 100
    
    # Correct order of days of the week
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_stats['weekday'] = pd.Categorical(weekday_stats['weekday'], categories=weekday_order, ordered=True)
    weekday_stats = weekday_stats.sort_values('weekday')
    
    # Gap fill statistics, if available
    if gap_fill_df is not None and not gap_fill_df.empty:
        gap_fill_stats = gap_fill_df.groupby('weekday').agg(
            gap_count=('date', 'count'),
            gap_filled_count=('gap_filled', 'sum')
        ).reset_index()
        
        gap_fill_stats['gap_fill_percentage'] = gap_fill_stats['gap_filled_count'] / gap_fill_stats['gap_count'] * 100
        gap_fill_stats['weekday'] = pd.Categorical(gap_fill_stats['weekday'], categories=weekday_order, ordered=True)
        gap_fill_stats = gap_fill_stats.sort_values('weekday')
        
        # Save gap fill statistics
        gap_fill_stats.to_csv(os.path.join(result_dir, 'gap_fill_statistics.csv'), index=False)
    
    # Save statistics
    weekday_stats.to_csv(os.path.join(result_dir, 'weekday_statistics.csv'), index=False)
    
    # Overall statistics
    total_days = len(results_df)
    total_touches = results_df['touch'].sum()
    total_snap_backs = results_df['snap_back'].sum()
    
    overall_touch_pct = total_touches / total_days * 100 if total_days > 0 else 0
    overall_snap_back_pct = total_snap_backs / total_touches * 100 if total_touches > 0 else 0
    
    # Save overall statistics to text file
    with open(os.path.join(result_dir, 'overall_statistics.txt'), 'w') as f:
        f.write(f"Midnight Open Snap Overall Statistics\n")
        f.write(f"Analysis period: {results_df['date'].min()} - {results_df['date'].max()}\n")
        f.write(f"Total days: {total_days}\n")
        f.write(f"Total TMO touches: {total_touches} ({overall_touch_pct:.2f}%)\n")
        f.write(f"Total snap backs: {total_snap_backs} ({overall_snap_back_pct:.2f}% of touches)\n")
        
        if gap_fill_df is not None and not gap_fill_df.empty:
            total_gaps = len(gap_fill_df)
            total_gap_fills = gap_fill_df['gap_filled'].sum()
            overall_gap_fill_pct = total_gap_fills / total_gaps * 100 if total_gaps > 0 else 0
            
            f.write(f"\nGap Fill Statistics\n")
            f.write(f"Total gaps ≥{GAP_THRESHOLD_PCT}%: {total_gaps}\n")
            f.write(f"Filled gaps ≥{GAP_FILL_PCT}%: {total_gap_fills} ({overall_gap_fill_pct:.2f}%)\n")
    
    print(f"Overall statistics saved to {os.path.join(result_dir, 'overall_statistics.txt')}")
    
    return weekday_stats

def plot_tmo_examples(df_5m, results_df, tmo_data, max_examples=5):
    """Create plots with examples of TMO touches for different days"""
    if results_df.empty or not results_df['touch'].any():
        print("No data for plotting")
        return
    
    # Create directory for plots
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Select days with TMO touches
    touch_days = results_df[results_df['touch']].sort_values('date')
    
    # Select snap_back cases, if any
    snap_back_days = results_df[results_df['snap_back']].sort_values('date')
    
    # Build plots for regular touches
    for i, (_, row) in enumerate(touch_days.iterrows()):
        if i >= max_examples:
            break
            
        date = row['date']
        tmo_level = row['tmo']
        
        # Get data for this day in analysis window
        day_data = df_5m[df_5m.index.date == date].between_time(WINDOW_START, WINDOW_END)
        
        if len(day_data) == 0:
            continue
        
        # Create plot
        fig = go.Figure(data=[go.Candlestick(
            x=day_data.index,
            open=day_data['open'],
            high=day_data['high'],
            low=day_data['low'],
            close=day_data['close'],
            name='Price'
        )])
        
        # Add TMO line
        fig.add_hline(
            y=tmo_level,
            line_dash="dash",
            line_color="blue",
            annotation_text="TMO",
            annotation_position="right"
        )
        
        # Add tolerance zone around TMO
        fig.add_hline(
            y=tmo_level + TOLERANCE_POINTS,
            line_dash="dot",
            line_color="lightblue",
            annotation_text=f"+{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=tmo_level - TOLERANCE_POINTS,
            line_dash="dot",
            line_color="lightblue",
            annotation_text=f"-{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        
        # Format plot
        fig.update_layout(
            title=f"TMO Touch Example - {date} ({row['weekday']})",
            xaxis_title="Time (NY)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        # Save plot
        fig.write_image(os.path.join(plots_dir, f"tmo_touch_{date}.png"))
        print(f"Created plot for {date}")
    
    # Build plots for snap back cases
    for i, (_, row) in enumerate(snap_back_days.iterrows()):
        if i >= max_examples:
            break
            
        date = row['date']
        tmo_level = row['tmo']
        
        # Get data for this day in analysis window
        day_data = df_5m[df_5m.index.date == date].between_time(WINDOW_START, WINDOW_END)
        
        if len(day_data) == 0:
            continue
        
        # Create plot
        fig = go.Figure(data=[go.Candlestick(
            x=day_data.index,
            open=day_data['open'],
            high=day_data['high'],
            low=day_data['low'],
            close=day_data['close'],
            name='Price'
        )])
        
        # Add TMO line
        fig.add_hline(
            y=tmo_level,
            line_dash="dash",
            line_color="red",
            annotation_text="TMO (Snap Back)",
            annotation_position="right"
        )
        
        # Add tolerance zone around TMO
        fig.add_hline(
            y=tmo_level + TOLERANCE_POINTS,
            line_dash="dot",
            line_color="pink",
            annotation_text=f"+{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=tmo_level - TOLERANCE_POINTS,
            line_dash="dot",
            line_color="pink",
            annotation_text=f"-{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        
        # Format plot
        fig.update_layout(
            title=f"TMO Snap Back Example - {date} ({row['weekday']})",
            xaxis_title="Time (NY)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        # Save plot
        fig.write_image(os.path.join(plots_dir, f"tmo_snap_back_{date}.png"))
        print(f"Created plot for snap back {date}")

def plot_hourly_touch_distribution(hour_stats, market_open_stats):
    """Create plot showing TMO touch distribution by hour"""
    # Extract hours and percentages
    hours = list(hour_stats.keys())
    percentages = [stats['percentage'] for stats in hour_stats.values()]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for hourly percentages
    fig.add_trace(go.Bar(
        x=hours,
        y=percentages,
        text=[f"{p:.2f}%" for p in percentages],
        textposition='auto',
        name='Touch Percentage by Hour',
        marker_color='lightblue'
    ))
    
    # Add horizontal line for market open period percentage
    fig.add_shape(
        type="line",
        x0=7.5, 
        y0=market_open_stats['percentage'],
        x1=10.5, 
        y1=market_open_stats['percentage'],
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for market open period
    fig.add_annotation(
        x=9, 
        y=market_open_stats['percentage'],
        text=f"Market Open Period: {market_open_stats['percentage']:.2f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30
    )
    
    # Update layout
    fig.update_layout(
        title='TMO Touch Percentage by Hour (NY Time)',
        xaxis_title='Hour (NY Time)',
        yaxis_title='Touch Percentage (%)',
        xaxis=dict(
            tickmode='array',
            tickvals=hours,
            ticktext=[f"{h}:00" for h in hours]
        ),
        yaxis=dict(
            range=[0, 100]
        )
    )
    
    # Save plot
    fig.write_image(os.path.join(result_dir, 'hourly_touch_distribution.png'))
    print(f"Created hourly touch distribution plot")
    
    return fig

def main():
    print("Starting Midnight Open Snap analysis...")
    
    # Load data
    print("Loading data...")
    df_1h = load_data(DATA_FILE_1H)
    df_5m = load_data(DATA_FILE_5M)
    df_30m = load_data(DATA_FILE_30M)
    
    if df_1h.empty or df_5m.empty:
        print("Error: Failed to load required data")
        return
    
    # Identify TMO levels
    print("Identifying TMO levels...")
    tmo_data = identify_tmo_levels(df_1h)
    
    # Analyze touches by minute intervals
    print("Analyzing touches by minute intervals...")
    minute_stats, touches_by_minute_df = analyze_touches_by_time(df_5m, tmo_data)
    
    # Analyze touches by hour
    print("Analyzing touches by hour and market open period...")
    hour_stats, market_open_stats, touches_by_hour_df = analyze_touches_by_hour(df_5m, tmo_data)
    
    # Create hourly distribution plot
    print("Creating hourly touch distribution plot...")
    plot_hourly_touch_distribution(hour_stats, market_open_stats)
    
    # Full Midnight Open Snap analysis
    print("Performing full Midnight Open Snap analysis...")
    results_df, snap_back_df, gap_fill_df = analyze_midnight_snap(df_1h, df_5m, df_30m, tmo_data)
    
    # Aggregate statistics
    print("Aggregating statistics...")
    weekday_stats = aggregate_statistics(results_df, snap_back_df, gap_fill_df)
    
    # Build example plots
    print("Building example plots of TMO touches...")
    plot_tmo_examples(df_5m, results_df, tmo_data)
    
    print(f"Analysis complete. Results saved in directory {result_dir}")
    
    # Print statistics to console
    if weekday_stats is not None:
        print("\nStatistics by day of week:")
        print(weekday_stats[['weekday', 'total_days', 'touch_count', 'touch_percentage', 'snap_back_count', 'snap_back_percentage']])
        
        print("\nStatistics by hour:")
        for hour, stats in sorted(hour_stats.items()):
            print(f"{hour:02d}:00: {stats['touches']}/{stats['total']} ({stats['percentage']:.2f}%)")
        
        print("\nMarket Open Period (8:00-10:30 AM):")
        print(f"Touches: {market_open_stats['touches']}/{market_open_stats['total']} ({market_open_stats['percentage']:.2f}%)")
        
        print("\nStatistics by minute intervals:")
        for minute, stats in minute_stats.items():
            print(f"{minute:02d}: {stats['touches']}/{stats['total']} ({stats['percentage']:.2f}%)")

if __name__ == "__main__":
    main() 