# PRD: NASDAQ H1 Swing Failure Statistical Analyzer with Cursor

Based on https://youtu.be/UJ58fe7g91c?si=9jIu6l1WMElvfSBy video

data to donwload: NT market data https://drive.google.com/drive/folders/130enKE2NMg5HriMu6VjwnZjWzPqprM3a + https://drive.google.com/drive/folders/1ZPmLmaUqk-DH-YXN110gpYxOfn9iiKe3 
+ https://firstratedata.com/i/futures/NQ
+ https://portaracqg.com/futures/int/enq

**Version:** 1.2
**Date:** 2025-05-11
**Author/User:** 

## 1. Project Overview

### 1.1. Objective
To replicate and potentially refine a statistical model for identifying and analyzing "H1 Swing Failure" patterns on historical NASDAQ (QQQ) hourly data. The primary goal is to understand the probabilities of certain price movements following these patterns and to explore how different pattern parameters affect these probabilities. The project will also explore extending the analysis to other timeframes and incorporating historical context and statistical correlations to enhance trading strategy insights.

### 1.2. Scope
*   Attempt to acquire historical NASDAQ (QQQ) hourly data from 2010 to the present. (Note: yfinance limitations encountered, using provided sample data for initial analysis).
*   Implement Python scripts (assisted by Cursor) to:
    *   Define bearish and bullish H1 Swing Failure patterns (3-candle setup).
    *   Iterate through historical data to find these patterns.
    *   Analyze the outcome of the candle following the pattern (C2) relative to key levels of the pattern candles (C0, C1).
    *   Aggregate statistics by the hour of pattern formation (NY Time).
    *   Generate example plots of identified patterns.
*   Use Cursor for code generation, modification, explanation, and debugging.
*   Experiment with different pattern definition parameters.
*   Implement Net Change Analysis as an additional feature.
*   **Analyze the influence of previous days' price distribution on swing failure statistics.**
*   **Perform swing failure analysis on 5-minute and 30-minute timeframes.**
*   **Analyze correlations and explanatory power between 1-hour, 30-minute, and 5-minute swing failure patterns.**
*   **Utilize statistical correlation methods (using libraries like pandas/numpy) to find correlations relevant to the trading strategy and identify potential advantages.**

### 1.3. Target User
A trader/analyst familiar with basic Python and trading concepts, looking to use Cursor to efficiently conduct statistical market studies.

## 2. Prerequisites

### 2.1. Software
*   **Cursor IDE:** Installed and configured.
*   **Python 3.x:** Installed.
*   **Python Virtual Environment:** Recommended (`venv_trading_stats` created and activated).
*   **Required Python Libraries:**
    *   `pandas` (Installed)
    *   `numpy` (Installed)
    *   `yfinance` (Installed - Note: Limitations for historical hourly data)
    *   `plotly` (Installed)
    *   `kaleido` (Installed for plotting)
    *   Ensure these are installed in your active virtual environment: `pip install pandas numpy yfinance plotly kaleido`

### 2.2. Data
*   **Source:** Initially planned: NASDAQ 100 hourly data (using QQQ ETF as a proxy via `yfinance`) from January 1, 2010, to present. **Currently using sample hourly data provided in `frd_sample_futures_NQ/NQ_1hour_sample.csv` due to yfinance historical data limitations.** Future steps will use available sample data: `NQ_5min_sample.csv` and `NQ_30min_sample.csv`.
*   **Timespan:** Sample data covers a limited recent period. Full analysis requires data from January 1, 2010, to present for both 1-hour and potentially 15-minute intervals.
*   **Format:** CSV file (`nasdaq_h1_2010_present.csv` originally planned, now using `frd_sample_futures_NQ/NQ_1hour_sample.csv`) with columns: `Timestamp` (or `timestamp`), `Open`, `High`, `Low`, `Close`, `Volume`. 15-minute data should ideally follow a similar format.

### 2.3. Existing Code Files
You should have the following files:
*   `scrapers/download_ticker_data.py`
*   `analyzers/hourly_swing_failure_analyzer.py` (Modified to use sample data, refined pattern definitions, save results)
*   `analyzers/net_change_analyzer.py` (Created)

## 3. Project Setup & Structure in Cursor

### 3.1. Folder Structure

NQ_ANALYZE/
├── analyzers/
│ └── hourly_swing_failure_analyzer.py
│ └── net_change_analyzer.py
├── scrapers/
│ └── download_ticker_data.py
├── frd_sample_futures_NQ/ (Contains sample data)
│   └── NQ_1hour_sample.csv
│   └── NQ_15min_sample.csv (Placeholder for future 15-minute data) - **Updated to NQ_5min_sample.csv and NQ_30min_sample.csv**
├── analysis_results/ (New folder for timestamped analysis runs)
│   └── height[value]_depth[value]_YYYYMMDD_HHMMSS/
│       ├── configuration.txt
│       ├── swing_failure_stats.txt
│       └── plots/ (Example plots)
├── plots_swing_failures/ (Original plotting output directory - can be deprecated)
├── nasdaq_h1_2010_present.csv (Original target data file - currently not used for analysis)
└── README.md
└── PRD.md (this file)

### 3.2. Opening Project in Cursor
1.  Launch Cursor.
2.  Open the `NQ_ANALYZE` folder as your project workspace.

## 4. Core Analysis Workflow with Cursor (Completed using sample data)

### 4.1. Step 1: Data Acquisition (Running `download_ticker_data.py`)

*   Attempted to use `scrapers/download_ticker_data.py` to acquire historical data from 2010. Encountered limitations with `yfinance` for the requested historical depth of hourly data.
*   Proceeded with using provided sample hourly data located in `frd_sample_futures_NQ/NQ_1hour_sample.csv`.

### 4.2. Step 2: Swing Failure Analysis (Running `analyzers/hourly_swing_failure_analyzer.py`)

*   Successfully ran the `analyzers/hourly_swing_failure_analyzer.py` script using the sample data.
*   The script produced the statistical summary table and generated example plots. The script has been modified to save these results in a timestamped directory within the `analysis_results` folder.

### 4.3. Step 3: Refining Pattern Definitions with Cursor

*   Successfully refined the `is_bearish_swing_failure` and `is_bullish_swing_failure` functions based on the specific conditions mentioned in this section and the user's instructions.
*   Modified the script to use configuration variables for easier experimentation.

### 4.4. Step 4: Refining Statistical Aggregation with Cursor (Optional)

*   Successfully modified the `aggregate_and_print_stats` function to calculate the 'Hit%' as a percentage of the total number of bearish (or bullish) patterns found across all hours.

### 4.5. Step 5: Debugging with Cursor

*   Encountered and successfully resolved a plotting error related to the `kaleido` library.
*   Resolved a linter error in `aggregate_and_print_stats` related to line structure.

## 5. Key Parameters for Experimentation (via Cursor prompts)

Experiment with different values for `C0_C1_RELATIVE_HEIGHT_THRESHOLD` and `C1_RETRACEMENT_DEPTH_THRESHOLD` in `analyzers/hourly_swing_failure_analyzer.py` and re-run the script to see how the statistics change. The results will be saved in timestamped directories.

1.  **`C0_C1_RELATIVE_HEIGHT_THRESHOLD`:** (e.g., 0.9, 0.75, 0.5)
2.  **`C1_RETRACEMENT_DEPTH_THRESHOLD`:** (e.g., 0.2, 0.5, 0.8 for how much C1 can retrace into C0 against the expected direction without invalidating the pattern)
3.  **Strictness of C1 close within C0 range:** Currently `c1['close'] < c0['high'] and c1['close'] > c0['low']`. This can also be refined.

## 6. Advanced Analyses & Future Steps

### 6.1. Net Change Analysis (Completed)

*   **Description:** A script (`analyzers/net_change_analyzer.py`) was created to calculate and analyze the net percentage change for hourly candles, including outlier filtering and hourly aggregation.

### 6.2. Influence of Previous Days' Distribution

*   **Objective:** To determine if the overall price movement (distribution) in the 1 to 3 days preceding a potential swing failure pattern influences the pattern's outcome probabilities.
*   **Tasks:**
    1.  Modify `analyzers/hourly_swing_failure_analyzer.py` (or create a new script) to calculate metrics describing the price distribution of the previous 1, 2, and 3 days for each candle (e.g., total percentage change, number of bullish/bearish days, volatility).
    2.  Incorporate these historical distribution metrics into the analysis.
    3.  Analyze and report how different historical distribution scenarios correlate with the success or failure rates of swing failure patterns.

### 6.3. 5-Minute and 30-Minute Timeframe Analysis

*   **Objective:** To replicate the swing failure analysis on NASDAQ (QQQ) 5-minute and 30-minute data.
*   **Tasks:**
    1.  Use the available sample NASDAQ (QQQ) 5-minute and 30-minute data (`NQ_5min_sample.csv` and `NQ_30min_sample.csv`).
    2.  Create new Python scripts (e.g., `analyzers/thirty_min_swing_failure_analyzer.py` and `analyzers/five_min_swing_failure_analyzer.py`) based on the hourly script.
    3.  Modify the new scripts to load and process the respective timeframe data.
    4.  Run the analysis on both 5-minute and 30-minute data and generate statistical output and plots similar to the 1-hour analysis.

### 6.4. Cross-Timeframe Correlation and Insights

*   **Objective:** To find correlations and gain explanatory insights by comparing swing failure patterns and outcomes between the 1-hour, 30-minute, and 5-minute timeframes.
*   **Tasks:**
    1.  Develop methods to match or relate patterns occurring on the 5-minute and 30-minute timeframes with those on the 1-hour timeframe (e.g., does a 5-min SFP within a 30-min SFP, which is within an H1 SFP, have a different probability?).
    2.  Use statistical methods (e.g., correlation coefficients, contingency tables) to analyze the relationship between patterns and outcomes across timeframes.
    3.  Document findings on how different timeframe interactions correlate with pattern success/failure and provide potential explanations for observed correlations.

- **Risk to Reward (R:R) Analysis**
    - **Objective:** To analyze the potential Risk to Reward ratio associated with swing failure patterns to identify potentially profitable trading opportunities.
    - **Tasks:**
        1.  Modify analyzer scripts to define potential Stop Loss and Target levels based on pattern characteristics (e.g., C1 high/low, C0 high/low). (Partially implemented in 1h and 5min scripts).
        2.  Analyze price action following the pattern (after C2) within a defined number of candles to determine if the Stop Loss or Target was hit first.
        3.  Calculate the achieved Reward and the Risk taken for each pattern occurrence.
        4.  Incorporate R:R metrics (e.g., average R:R ratio, percentage of patterns hitting target/stop loss) into the statistical reporting for each timeframe.
        5.  Include R:R information in the cross-timeframe correlation analysis.

### 6.5. General Statistical Correlation Analysis

*   **Objective:** To use statistical methods to uncover correlations within the data (both raw price data and identified pattern occurrences/outcomes) that can provide insights and potential advantages for the trading strategy.
*   **Tasks:**
    1.  Identify potential variables for correlation analysis (e.g., volume during pattern formation, time of day, preceding price action, volatility, specific candle shapes).
    2.  Apply statistical techniques (e.g., Pearson correlation, Spearman correlation, regression analysis) using libraries like pandas and numpy to quantify relationships.
    3.  Analyze the strength and significance of identified correlations.
    4.  Summarize findings and discuss how these correlations might inform and improve the trading strategy.

## 7. Expected Deliverables
*   A functioning Python environment for the analysis.
*   Sample data files used for analysis (`frd_sample_futures_NQ/NQ_1hour_sample.csv`, potentially `frd_sample_futures_NQ/NQ_15min_sample.csv` or similar).
*   Modified `analyzers/hourly_swing_failure_analyzer.py` script.
*   `analyzers/net_change_analyzer.py` script.
*   (Future) New script(s) for 15-minute analysis and cross-timeframe correlation.
*   Analysis results saved in timestamped directories (`analysis_results/`).
*   Console output of statistical tables.
*   PNG images of example patterns.
*   Documentation of findings from historical distribution analysis, cross-timeframe analysis, and general correlation analysis.

## 8. Notes & Best Practices for using Cursor
*   **Be Specific:** The more specific your prompt to Cursor (Ctrl+K or chat), the better the results.
*   **Iterate:** Don't expect perfect code on the first try. Ask Cursor for small changes, test, then ask for more.
*   **Select Code:** When asking for modifications to existing code, select the relevant lines/function before invoking Cursor.
*   **Use Chat (Ctrl+L):** For broader questions, explanations, or generating new script ideas.
*   **Use "Edit with AI" (Ctrl+K on selection):** For targeted code modifications.
*   **Review AI Output:** Always review code generated or modified by AI to ensure it meets your requirements and is correct.
*   **Version Control (Git):** Highly recommended. Commit changes after each significant successful step.

## 9. Midnight Open Snap Analysis

### 9.1. Objective
To test the hypothesis that price returns to the TMO (True Midnight Open, 00:00 ET) level within the 08:30-12:00 ET trading window with a probability of 60-80%, as mentioned in ICT methodology. The analysis will also identify "snap back" patterns (false breakouts with quick reversals) and analyze their probability of occurrence.

### 9.2. Scope
- Use existing NQ futures data from the frd_sample_futures_NQ directory (1H, 5M, 30M timeframes)
- Identify the TMO level for each trading day
- Analyze price touches to TMO in the specified window
- Calculate touch probabilities by day of week (Monday-Friday)
- Analyze distribution of touches by minute intervals (0, 15, 30, 45 minutes)
- Identify and analyze "snap back" patterns (false breakouts with quick reversals)
- Calculate "gap fill" statistics (when overnight gaps ≥X% are filled by ≥50%)
- Generate visualizations of price movement around TMO levels
- Save statistical results and plots

### 9.3. Prerequisites
- Existing data files: NQ_1hour_sample.csv, NQ_5min_sample.csv, NQ_30min_sample.csv
- Libraries: pandas, numpy, plotly, kaleido
- Timezone conversion support for New York Time (EST/EDT)

### 9.4. Expected Deliverables
- Python script midnight_open_analyzer.py in the midnight_open_snap directory
- Statistical tables showing touch probabilities by day of week
- Distribution of touches by minute intervals
- Gap fill analysis results
- Visualization plots of price action around TMO levels
- CSV output files with detailed results
- Integration with existing Swing Failure Pattern analysis for cross-pattern insights


