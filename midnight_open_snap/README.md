# Midnight Open Snap Strategy Analysis

## Overview
This component analyzes the "Midnight Open Snap" strategy (referenced in ICT methodology, video grLk7TGznLA), which tests the hypothesis that price tends to return to the TMO (True Midnight Open) level in the 08:30-12:00 ET trading window with a probability of 60-80%.

## Key Findings
Based on our analysis of TradingView NQ futures data (2021-05-03 to 2025-05-09):
- Overall TMO touch probability: 69.62% (724 touches out of 1040 days)
- Snap back probability: 55.66% (403 cases out of 724 touches)
- Market open period (8:00-10:30 AM) touch probability: 70.87%

### By Hour (NY Time)
- 09:00: 53.37% touch probability
- 10:00: 44.89% touch probability
- 11:00: 31.21% touch probability

### By Day of Week
- Monday: 68.93% touch rate, 54.93% snap back rate
- Tuesday: 68.57% touch rate, 53.47% snap back rate
- Wednesday: 68.75% touch rate, 60.14% snap back rate
- Thursday: 67.14% touch rate, 52.48% snap back rate
- Friday: 74.76% touch rate, 57.14% snap back rate

## Key Concepts
- **TMO (True Midnight Open)**: Price at the opening of the 00:00 ET candle
- **Trading Window**: 08:30-12:00 ET
- **Market Open Period**: 08:00-10:30 ET (highest probability window)
- **Snap Back**: False breakout of TMO level followed by quick reversal within 1-2 candles
- **Gap Fill**: When price fills ≥50% of overnight gap (if gap ≥X%)

## Data
This analysis uses TradingView NQ futures data:
- CME_MINI_DL_NQ1! hourly data

## Usage
```bash
python3 midnight_open_analyzer.py
```

## Results
Analysis results are saved in the `results/` directory with timestamp for:
- Statistical tables of touch probabilities by day, hour, and time
- Market open period (8:00-10:30 AM) specific analysis
- Visualization plots showing price movement near TMO levels
- Hourly touch distribution chart
- CSV files with detailed analysis results 