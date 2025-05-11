# NASDAQ Swing Failure Statistical Analyzer

## 1. Project Overview

Statistical analysis tool for identifying and evaluating Swing Failure Patterns (SFP) in NASDAQ futures (NQ) data across multiple timeframes. It is designed to replicate and refine a statistical model that assists traders in making informed decisions based on historical pattern performance.

## 2. Project Status

This project implements a statistical analyzer for H1 (1-hour), 30-minute, and 5-minute Swing Failure patterns on NASDAQ (NQ) futures data using Python and the Cursor IDE.

**Analysis is currently performed using provided sample data and TradingView data.** Initial Risk:Reward calculations are integrated into individual timeframe analyzers, and preliminary cross-timeframe correlation analysis is implemented.

Further steps include refining R:R calculations, enhancing cross-timeframe analysis, and conducting general statistical correlations.

## 3. Features

- Analysis of swing failure patterns across multiple timeframes (1h, 30m, 5m)
- Integration of TradingView data with Unix timestamp support
- Previous day's distribution analysis for contextual pattern evaluation
- Risk:Reward calculation and tracking
- Cross-timeframe correlation analysis
- Statistical aggregation and results sorting
- Visualization of identified patterns with plotly

## 4. Data Sources

- **FRD Sample Data**: 1-hour, 30-minute, and 5-minute sample data for NQ futures
- **TradingView Data**: Historical hourly data for NQ futures from 2021-04-30 to 2025-05-09

## 5. Project Structure

```
NQ_ANALYZE/
├── analyzers/
│ ├── hourly_swing_failure_analyzer.py
│ ├── thirty_min_swing_failure_analyzer.py
│ ├── five_min_swing_failure_analyzer.py
│ ├── cross_timeframe_analyzer.py
│ ├── tradingview_hourly_swing_failure_analyzer.py
│ └── net_change_analyzer.py
├── midnight_open_snap/
│ ├── midnight_open_analyzer.py
│ ├── results/
│ │ └── results_YYYYMMDD_HHMM/
│ └── README.md
├── scrapers/
│ └── download_ticker_data.py
├── tests/
│ └── test_hourly_swing_failure_analyzer.py
├── frd_sample_futures_NQ/
│ ├── NQ_1hour_sample.csv
│ ├── NQ_30min_sample.csv
│ └── NQ_5min_sample.csv
├── tradingview_futures_NQ/
│ └── CME_MINI_DL_NQ1!, 60, 30 apr 2021 - 11 May 2025.csv
├── analysis_results/
├── analysis_results_30min/
├── analysis_results_5min/
├── analysis_results_tradingview/
├── analyze_results.py
├── README.md
└── PRD.md
```

## 6. Technologies Used

- Python 3.x
- pandas and numpy for data processing
- plotly for visualization
- kaleido for exporting plots
- unittest for testing

## 7. Deliverables

*   A functioning Python environment (`venv_trading_stats`) with necessary libraries (`pandas`, `numpy`, `yfinance`, `plotly`, `kaleido`).
*   Sample data files used for analysis:
    * FRD sample data: (`frd_sample_futures_NQ/NQ_1hour_sample.csv`, `frd_sample_futures_NQ/NQ_30min_sample.csv`, `frd_sample_futures_NQ/NQ_5min_sample.csv`)
    * TradingView data: (`tradingview_futures_NQ/CME_MINI_DL_NQ1!, 60, 30 apr 2021 - 11 May 2025.csv`)
*   Analysis scripts:
    *   `analyzers/hourly_swing_failure_analyzer.py` (includes Previous Day Distribution and initial R:R logic)
    *   `analyzers/thirty_min_swing_failure_analyzer.py` (includes initial R:R logic - faced some issues during implementation)
    *   `analyzers/five_min_swing_failure_analyzer.py` (includes initial R:R logic)
    *   `analyzers/cross_timeframe_analyzer.py` (initial version)
    *   `analyzers/tradingview_hourly_swing_failure_analyzer.py` (TradingView data analyzer with Unix timestamp handling)
    *   `analyze_results.py` (statistical analysis and sorting of results)
*   `analyzers/net_change_analyzer.py` script.
*   Basic test suite in the `tests/` directory.
*   Timestamped analysis results directories:
    *   FRD data results: (`analysis_results/`, `analysis_results_30min/`, `analysis_results_5min/`)
    *   TradingView data results: (`analysis_results_tradingview/`)
    *   Each directory contains:
        *   Statistical summary files (`stats_summary_*.txt`).
        *   Raw pattern results CSVs (`raw_patterns_*.csv`).
        *   Example plot images (`plots/`).
*   Console output from script runs (stats, cross-timeframe correlations).
*   Updated `README.md` (this file) and `PRD.md`.

## 8. How to Run

1.  **Prerequisites:** Ensure you have Python 3.x installed. It is highly recommended to use a virtual environment.
2.  **Virtual Environment:** Navigate to the project root directory in your terminal and create/activate a virtual environment:
    ```bash
    python3 -m venv venv_trading_stats
    source venv_trading_stats/bin/activate
    ```
3.  **Install Dependencies:** With the virtual environment active, install the required libraries:
    ```bash
    pip install pandas numpy yfinance plotly kaleido
    ```
    *(Note: `yfinance` was used for initial data acquisition attempts but analysis currently relies on sample data.)*
4.  **Data:** 
    * Ensure the FRD sample data files (`NQ_1hour_sample.csv`, `NQ_30min_sample.csv`, `NQ_5min_sample.csv`) are placed in the `frd_sample_futures_NQ/` directory.
    * Ensure the TradingView data file(s) are placed in the `tradingview_futures_NQ/` directory.
5.  **Run Analysis Scripts:** Execute the analyzer scripts from the project root directory with the virtual environment active:
    *   FRD Data Analysis:
        *   Hourly Analysis: `python analyzers/hourly_swing_failure_analyzer.py`
        *   30-Minute Analysis: `python analyzers/thirty_min_swing_failure_analyzer.py`
        *   5-Minute Analysis: `python analyzers/five_min_swing_failure_analyzer.py`
        *   Cross-Timeframe Analysis: `python analyzers/cross_timeframe_analyzer.py`
    *   TradingView Data Analysis:
        *   Hourly Analysis: `python analyzers/tradingview_hourly_swing_failure_analyzer.py`
        *   Results Analysis: `python analyze_results.py`

    Each individual timeframe analysis script run will create a new timestamped directory within its respective `analysis_results*` folder containing the statistical summary, raw pattern results (including R:R columns), and example plots. The cross-timeframe analysis script will read the latest raw data from these directories and output correlation findings to the console.

6.  **Run Tests:** Execute the tests from the project root directory with the virtual environment active:
    ```bash
    python -m unittest discover tests
    ```

## 9. Analysis Results and Insights

Running the individual timeframe analysis scripts will output statistical summaries to the console and save detailed results (stats, raw data, plots) to timestamped directories. The `cross_timeframe_analyzer.py` script will output correlation findings to the console, analyzing how nested patterns influence outcomes.

### 9.1 Key Insights from Analysis

*   **Data Sources:** Analysis is performed on both FRD sample data and TradingView data, with the latter spanning from 2021-04-30 to 2025-05-09.
*   **Pattern Refinement:** The swing failure pattern definitions were refined based on specific criteria (C1 close within C0, C1 height, C1 retracement depth), influencing the number and type of patterns identified.
*   **Statistical Aggregation:** Statistical reporting calculates 'Hit%' as a percentage of patterns of that type, providing a clearer picture of pattern frequency and success by time/context.
*   **Previous Day's Distribution:** Analysis categorizes patterns based on the preceding day's price change, showing correlations between market context and pattern outcomes.
*   **Multiple Timeframe Analysis:** Analysis scripts for 30-minute and 5-minute timeframes provide statistical summaries and raw pattern data for these intervals.
*   **Cross-Timeframe Correlations:** Initial cross-timeframe analysis shows that smaller-timeframe patterns within larger-timeframe pattern windows can correlate with the larger pattern's outcome.
*   **Risk to Reward (R:R) Analysis:** Initial R:R calculation and tracking logic has been integrated into the scripts, providing potential Stop Loss, Reward Excursion, and hit flags.

### 9.2 TradingView Data Analysis Results

The TradingView data analysis (1,730 patterns from 2021-04-30 to 2025-05-09) has revealed several important findings:

1. **Pattern Distribution by Hour:**
   - Bearish patterns most frequently form at 15:00 (54), 2:00 (53), and 1:00 (52)
   - Bullish patterns most frequently form at 15:00 (55), 20:00 (53), and 2:00 (51)

2. **Pattern Distribution by Previous Day's Change:**
   - Bearish patterns: Most common after Strongly Up (264) and Slightly Up (244) days
   - Bullish patterns: Most common after Slightly Down (222) and Slightly Up (211) days

3. **Best Success Rate for First Level Breakout (Hit%_1st):**
   - Bearish patterns: 11:00 (52.4%), 12:00 (30.8%), 5:00 (27.0%)
   - Bullish patterns: 3:00 (39.0%), 11:00 (38.1%), 4:00 (33.3%)

4. **Best Combinations by Success Rate:**
   - Bearish: Slightly Up at 11:00 (66.7% first level breakout, 100% target achievement)
   - Bullish: Strongly Down at 5:00 (80.0% first level breakout, 100% target achievement)

5. **Best Risk-Reward Ratio (Target Hit % / Stop Loss Hit %):**
   - Bearish: Strongly Down at 14:00 and Strongly Up at 13:00 (ratio 3.0)
   - Bullish: Strongly Up and Slightly Up at 13:00, Strongly Down at 5:00 (ratio 5.0)

### 9.3 Trading Implications

These findings suggest potentially high-probability setups:

1. **Best Bearish Setups:**
   - Formation at 11:00 after a slightly upward day (Slightly Up) - 67% first level breakout, 100% target achievement
   - Formation at 13:00 after a strongly upward day (Strongly Up) - target/stop ratio = 3.0

2. **Best Bullish Setups:**
   - Formation at 5:00 after a strongly downward day (Strongly Down) - 80% first level breakout, 100% target achievement
   - Formation at 13:00 (after Strongly Up or Slightly Up) - target/stop ratio = 5.0

3. **General Observations:**
   - Bearish patterns tend to form more frequently after upward days
   - Bullish patterns tend to form more frequently after downward days
   - Overall target achievement rates are higher for patterns forming in the morning hours (3:00-11:00)

## 10. Future Development

- Implement TradingView data analysis for 30-minute and 5-minute timeframes
- Enhance cross-timeframe correlation with consistent R:R metrics
- Create a comprehensive testing framework
- Develop a web interface for interactive analysis
- Implement real-time pattern detection capabilities

## 11. Notes & Best Practices

- Keep the virtual environment active when running scripts
- Use timestamped output directories to preserve historical analysis runs
- Set appropriate threshold values for pattern identification
- Consider market context (previous day's movement) when evaluating patterns
- Compare patterns across multiple timeframes for confirmation
