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
        
        # Track if we've already found a touch for this day
        day_touched = False
        first_touch_time = None
        
        # For each 5-minute candle, check if there was a touch of TMO
        for idx, candle in window_data.iterrows():
            minute = idx.minute
            minute_group = (minute // 15) * 15  # Group by 15-minute intervals (0, 15, 30, 45)
            
            # Increment counter for corresponding minute interval
            total_by_minute[minute_group] += 1
            
            # Check touch
            touched = (candle['low'] <= tmo_level + TOLERANCE_POINTS) and (candle['high'] >= tmo_level - TOLERANCE_POINTS)
            
            if touched and not day_touched:
                day_touched = True
                first_touch_time = idx
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
                    'touch': True,
                    'first_touch': True
                })
            elif touched:
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
                    'touch': True,
                    'first_touch': False
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

def identify_candlestick_pattern(candle):
    """Identify candlestick patterns in a single candle
    Returns a tuple (pattern_name, pattern_type) where pattern_type is 'bullish' or 'bearish'
    """
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    total_range = candle['high'] - candle['low']
    
    # Pin bar (hammer/shooting star)
    if total_range > 0:
        if candle['close'] > candle['open']:  # Bullish
            if lower_wick > 2 * body and upper_wick < 0.2 * total_range:
                return ('pin_bar', 'bullish')
        else:  # Bearish
            if upper_wick > 2 * body and lower_wick < 0.2 * total_range:
                return ('pin_bar', 'bearish')
    
    # Engulfing pattern (requires previous candle)
    # This will be checked in analyze_snap_back()
    
    return (None, None)

def analyze_volume(df, lookback=20):
    """Analyze volume relative to recent average
    Returns a Series with boolean values where True indicates high volume (>2x average)
    """
    if 'volume' not in df.columns:
        print("Warning: Volume data not available")
        return pd.Series(False, index=df.index)
    
    avg_volume = df['volume'].rolling(window=lookback).mean()
    high_volume = df['volume'] > 2 * avg_volume
    return high_volume

def analyze_snap_back(df_5m, tmo_data, lookback_minutes=30):
    """Enhanced snap back analysis using 5-minute data and pattern recognition"""
    snap_backs = []
    
    for _, row in tmo_data.iterrows():
        date = row['date']
        tmo_level = row['tmo']
        
        # Get data for this day
        day_data = df_5m[df_5m.index.date == date]
        window_data = day_data.between_time(WINDOW_START, WINDOW_END)
        
        # Track if we've already found a snap back for this day
        day_snap_back = False
        
        # Analyze each potential breakout
        for i in range(1, len(window_data)):
            current = window_data.iloc[i]
            previous = window_data.iloc[i-1]
            
            if day_snap_back:
                break
                
            # Check for breakout below TMO
            if current['low'] < tmo_level - TOLERANCE_POINTS:
                # Look forward for snap back
                forward_window = window_data.iloc[i:i+6]  # Next 30 minutes
                if len(forward_window) > 0 and forward_window['high'].max() > tmo_level:
                    # Found potential snap back, check for confirmation
                    snap_candle = forward_window[forward_window['high'] > tmo_level].iloc[0]
                    pattern, pattern_type = identify_candlestick_pattern(snap_candle)
                    
                    # Check if it's an engulfing pattern
                    is_engulfing = (
                        snap_candle['open'] < previous['close'] and
                        snap_candle['close'] > previous['open']
                    )
                    
                    if pattern or is_engulfing:
                        day_snap_back = True
                        snap_backs.append({
                            'date': date,
                            'breakout_time': current.name,
                            'snap_back_time': snap_candle.name,
                            'tmo': tmo_level,
                            'pattern': pattern if pattern else 'engulfing',
                            'pattern_type': pattern_type if pattern else 'bullish',
                            'risk': abs(snap_candle['low'] - tmo_level),
                            'potential_reward': abs(tmo_level - current['low']),
                            'direction': 'long',
                            'breakout_size': abs(current['low'] - tmo_level),
                            'time_to_snap': (snap_candle.name - current.name).total_seconds() / 60
                        })
            
            # Check for breakout above TMO
            elif current['high'] > tmo_level + TOLERANCE_POINTS:
                # Look forward for snap back
                forward_window = window_data.iloc[i:i+6]  # Next 30 minutes
                if len(forward_window) > 0 and forward_window['low'].min() < tmo_level:
                    # Found potential snap back, check for confirmation
                    snap_candle = forward_window[forward_window['low'] < tmo_level].iloc[0]
                    pattern, pattern_type = identify_candlestick_pattern(snap_candle)
                    
                    # Check if it's an engulfing pattern
                    is_engulfing = (
                        snap_candle['open'] > previous['close'] and
                        snap_candle['close'] < previous['open']
                    )
                    
                    if pattern or is_engulfing:
                        day_snap_back = True
                        snap_backs.append({
                            'date': date,
                            'breakout_time': current.name,
                            'snap_back_time': snap_candle.name,
                            'tmo': tmo_level,
                            'pattern': pattern if pattern else 'engulfing',
                            'pattern_type': pattern_type if pattern else 'bearish',
                            'risk': abs(snap_candle['high'] - tmo_level),
                            'potential_reward': abs(tmo_level - current['high']),
                            'direction': 'short',
                            'breakout_size': abs(current['high'] - tmo_level),
                            'time_to_snap': (snap_candle.name - current.name).total_seconds() / 60
                        })
    
    # Convert to DataFrame
    snap_backs_df = pd.DataFrame(snap_backs)
    if not snap_backs_df.empty:
        snap_backs_df.to_csv(os.path.join(result_dir, 'snap_backs.csv'), index=False)
        
        # Calculate additional statistics
        snap_stats = {
            'avg_time_to_snap': snap_backs_df['time_to_snap'].mean(),
            'avg_breakout_size': snap_backs_df['breakout_size'].mean(),
            'direction_breakdown': snap_backs_df['direction'].value_counts().to_dict(),
            'pattern_breakdown': snap_backs_df['pattern'].value_counts().to_dict(),
            'best_patterns': snap_backs_df.groupby('pattern').agg({
                'potential_reward': 'mean',
                'risk': 'mean'
            }).assign(
                avg_rr=lambda x: x['potential_reward'] / x['risk']
            ).sort_values('avg_rr', ascending=False).to_dict('index')
        }
        
        # Save snap back statistics
        with open(os.path.join(result_dir, 'snap_back_stats.txt'), 'w') as f:
            f.write("Snap Back Detailed Statistics\n")
            f.write("===========================\n\n")
            
            f.write(f"Average time to snap back: {snap_stats['avg_time_to_snap']:.1f} minutes\n")
            f.write(f"Average breakout size: {snap_stats['avg_breakout_size']:.1f} points\n\n")
            
            f.write("Direction Breakdown:\n")
            for direction, count in snap_stats['direction_breakdown'].items():
                f.write(f"{direction}: {count} ({count/len(snap_backs_df)*100:.1f}%)\n")
            
            f.write("\nPattern Breakdown:\n")
            for pattern, count in snap_stats['pattern_breakdown'].items():
                f.write(f"{pattern}: {count} ({count/len(snap_backs_df)*100:.1f}%)\n")
            
            f.write("\nPattern Performance:\n")
            for pattern, stats in snap_stats['best_patterns'].items():
                f.write(f"{pattern}:\n")
                f.write(f"  Avg Risk: {stats['risk']:.1f} points\n")
                f.write(f"  Avg Reward: {stats['potential_reward']:.1f} points\n")
                f.write(f"  Avg R:R: {stats['avg_rr']:.2f}\n")
    
    return snap_backs_df

def analyze_gaps(df_1h, tmo_data):
    """Enhanced gap analysis with categorization by size"""
    gaps = []
    
    for _, row in tmo_data.iterrows():
        date = row['date']
        tmo_level = row['tmo']
        
        # Get previous day's close
        prev_day = df_1h[df_1h.index.date < date].iloc[-1]
        prev_close = prev_day['close']
        
        # Calculate gap percentage
        gap_pct = abs(tmo_level - prev_close) / prev_close * 100
        
        if gap_pct >= 0.1:  # Minimum gap size for analysis
            # Categorize gap
            if gap_pct < 0.5:
                gap_category = 'small'
            elif gap_pct < 1.0:
                gap_category = 'medium'
            else:
                gap_category = 'large'
            
            # Get data for gap fill analysis
            day_data = df_1h[df_1h.index.date == date]
            window_data = day_data.between_time(WINDOW_START, WINDOW_END)
            
            # Check if gap was filled
            if tmo_level > prev_close:  # Gap up
                fill_pct = (tmo_level - window_data['low'].min()) / (tmo_level - prev_close) * 100
                direction = 'up'
            else:  # Gap down
                fill_pct = (window_data['high'].max() - tmo_level) / (prev_close - tmo_level) * 100
                direction = 'down'
            
            gaps.append({
                'date': date,
                'tmo': tmo_level,
                'prev_close': prev_close,
                'gap_pct': gap_pct,
                'gap_category': gap_category,
                'direction': direction,
                'fill_percentage': fill_pct,
                'filled': fill_pct >= GAP_FILL_PCT
            })
    
    # Convert to DataFrame
    gaps_df = pd.DataFrame(gaps)
    if not gaps_df.empty:
        gaps_df.to_csv(os.path.join(result_dir, 'gaps.csv'), index=False)
    
    return gaps_df

def calculate_risk_reward_metrics(snap_backs_df):
    """Calculate risk-reward metrics for snap back trades"""
    if snap_backs_df.empty:
        return {}
    
    metrics = {
        'avg_risk_points': snap_backs_df['risk'].mean(),
        'avg_potential_reward_points': snap_backs_df['potential_reward'].mean(),
        'avg_rr_ratio': (snap_backs_df['potential_reward'] / snap_backs_df['risk']).mean(),
        'trades_with_min_2R': len(snap_backs_df[snap_backs_df['potential_reward'] / snap_backs_df['risk'] >= 2]),
        'total_trades': len(snap_backs_df),
        'percent_2R_opportunities': 0
    }
    
    metrics['percent_2R_opportunities'] = (metrics['trades_with_min_2R'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
    
    return metrics

def analyze_midnight_snap(df_1h, df_5m, df_30m, tmo_data):
    """Main analysis function for Midnight Open Snap strategy"""
    # Analyze touches by time intervals
    minute_stats, touches_df = analyze_touches_by_time(df_5m, tmo_data)
    
    # Analyze touches by hour
    hour_stats, market_open_stats, hour_results_df = analyze_touches_by_hour(df_5m, tmo_data)
    
    # Combine touch results
    touches_df = pd.concat([touches_df, hour_results_df]).drop_duplicates()
    
    # Enhanced snap back analysis
    snap_backs_df = analyze_snap_back(df_5m, tmo_data)
    
    # Enhanced gap analysis
    gaps_df = analyze_gaps(df_1h, tmo_data)
    
    # Calculate risk-reward metrics
    rr_metrics = calculate_risk_reward_metrics(snap_backs_df)
    
    # Analyze volume for touch candles if volume data is available
    if 'volume' in df_5m.columns:
        high_volume = analyze_volume(df_5m)
        touches_df['high_volume'] = high_volume[touches_df.index]
    
    # Aggregate all statistics
    stats = aggregate_statistics(touches_df, snap_backs_df, gaps_df)
    
    # Add risk-reward metrics to stats
    stats.update(rr_metrics)
    
    # Save all statistics to file
    with open(os.path.join(result_dir, 'statistics.txt'), 'w') as f:
        f.write("Midnight Open Snap Analysis Results\n")
        f.write("=================================\n\n")
        
        f.write("Touch Statistics:\n")
        f.write(f"Total trading days analyzed: {len(tmo_data)}\n")
        f.write(f"Total touches: {len(touches_df)}\n")
        f.write(f"Overall touch rate: {len(touches_df)/len(tmo_data)*100:.2f}%\n\n")
        
        if 'volume' in df_5m.columns:
            high_vol_touches = touches_df['high_volume'].sum()
            f.write(f"Touches with high volume: {high_vol_touches}\n")
            f.write(f"High volume touch rate: {high_vol_touches/len(touches_df)*100:.2f}%\n\n")
        
        f.write("Snap Back Statistics:\n")
        f.write(f"Total snap backs: {len(snap_backs_df)}\n")
        f.write(f"Snap back rate: {len(snap_backs_df)/len(touches_df)*100:.2f}%\n")
        f.write(f"Average R:R ratio: {rr_metrics['avg_rr_ratio']:.2f}\n")
        f.write(f"Trades with 2:1+ R:R: {rr_metrics['percent_2R_opportunities']:.2f}%\n\n")
        
        f.write("Gap Analysis:\n")
        if not gaps_df.empty:
            for category in ['small', 'medium', 'large']:
                category_gaps = gaps_df[gaps_df['gap_category'] == category]
                if len(category_gaps) > 0:
                    filled_rate = len(category_gaps[category_gaps['filled']]) / len(category_gaps) * 100
                    f.write(f"{category.capitalize()} gaps ({len(category_gaps)}): {filled_rate:.2f}% fill rate\n")
    
    return stats, touches_df, snap_backs_df, gaps_df

def aggregate_statistics(touches_df, snap_backs_df=None, gaps_df=None):
    """Aggregate statistics for touches, snap backs, and gaps"""
    stats = {}
    
    # Touch statistics by weekday
    if not touches_df.empty:
        weekday_stats = touches_df.groupby('weekday').agg(
            total_touches=('date', 'count')
        ).reset_index()
        
        # Calculate total days for each weekday
        total_days_by_weekday = touches_df['weekday'].value_counts()
        weekday_stats['touch_rate'] = weekday_stats['total_touches'] / total_days_by_weekday * 100
        
        stats['weekday_stats'] = weekday_stats.to_dict('records')
    
    # Snap back statistics
    if snap_backs_df is not None and not snap_backs_df.empty:
        snap_stats = {
            'total_snap_backs': len(snap_backs_df),
            'snap_back_rate': len(snap_backs_df) / len(touches_df) * 100 if not touches_df.empty else 0,
            'pattern_breakdown': snap_backs_df['pattern'].value_counts().to_dict(),
            'avg_risk': snap_backs_df['risk'].mean(),
            'avg_reward': snap_backs_df['potential_reward'].mean(),
            'avg_rr': (snap_backs_df['potential_reward'] / snap_backs_df['risk']).mean()
        }
        stats['snap_back_stats'] = snap_stats
    
    # Gap statistics
    if gaps_df is not None and not gaps_df.empty:
        gap_stats = {
            'total_gaps': len(gaps_df),
            'gap_categories': {
                category: {
                    'count': len(category_gaps),
                    'fill_rate': len(category_gaps[category_gaps['filled']]) / len(category_gaps) * 100
                }
                for category, category_gaps in gaps_df.groupby('gap_category')
            }
        }
        stats['gap_stats'] = gap_stats
    
    return stats

def plot_tmo_examples(df_5m, touches_df, tmo_data, max_examples=5):
    """Create example plots of TMO touches and snap backs"""
    if touches_df.empty:
        print("No examples to plot")
        return
    
    # Get unique dates with touches
    touch_dates = touches_df['date'].unique()
    
    # Select random dates for examples (up to max_examples)
    example_dates = np.random.choice(touch_dates, size=min(max_examples, len(touch_dates)), replace=False)
    
    # Create plots directory
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for date in example_dates:
        # Get TMO level for this date
        tmo_level = tmo_data[tmo_data['date'] == date]['tmo'].iloc[0]
        
        # Get data for this day
        day_data = df_5m[df_5m.index.date == date]
        window_data = day_data.between_time(WINDOW_START, WINDOW_END)
        
        if len(window_data) == 0:
            continue
        
        # Create figure
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'TMO Touch Example - {date}'])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=window_data.index,
                open=window_data['open'],
                high=window_data['high'],
                low=window_data['low'],
                close=window_data['close'],
                name='Price'
            )
        )
        
        # Add TMO level line
        fig.add_hline(
            y=tmo_level,
            line_dash="dash",
            line_color="red",
            annotation_text="TMO",
            annotation_position="right"
        )
        
        # Add tolerance bands
        fig.add_hline(
            y=tmo_level + TOLERANCE_POINTS,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"+{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        fig.add_hline(
            y=tmo_level - TOLERANCE_POINTS,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"-{TOLERANCE_POINTS}",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'TMO Touch Example - {date}',
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True,
            height=800
        )
        
        # Save plot
        fig.write_image(os.path.join(plots_dir, f'tmo_touch_example_{date}.png'))
        
        # Close figure to free memory
        fig.data = []
        
    print(f"Created {len(example_dates)} example plots in {plots_dir}")
    return

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
    stats, touches_df, snap_backs_df, gaps_df = analyze_midnight_snap(df_1h, df_5m, df_30m, tmo_data)
    
    # Aggregate statistics
    print("Aggregating statistics...")
    stats = aggregate_statistics(touches_df, snap_backs_df, gaps_df)
    
    # Build example plots
    print("Building example plots of TMO touches...")
    plot_tmo_examples(df_5m, touches_df, tmo_data)
    
    print(f"Analysis complete. Results saved in directory {result_dir}")
    
    # Print statistics to console
    if stats:
        print("\nStatistics by day of week:")
        print(pd.DataFrame(stats['weekday_stats']))
        
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