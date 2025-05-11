# PRD: NASDAQ H1 Swing Failure Statistical Analyzer with Cursor

Based on https://youtu.be/UJ58fe7g91c?si=9jIu6l1WMElvfSBy video

**Version:** 1.1
**Date:** 2025-05-11
**Author/User:** 

## 1. Project Overview

### 1.1. Objective
To replicate and potentially refine a statistical model for identifying and analyzing "H1 Swing Failure" patterns on historical NASDAQ (QQQ) hourly data. The primary goal is to understand the probabilities of certain price movements following these patterns and to explore how different pattern parameters affect these probabilities.

### 1.2. Scope
*   Attempt to acquire historical NASDAQ (QQQ) hourly data from 2010 to the present. (Note: yfinance limitations encountered, using provided sample data for analysis).
*   Implement Python scripts (assisted by Cursor) to:
    *   Define bearish and bullish H1 Swing Failure patterns (3-candle setup).
    *   Iterate through historical data to find these patterns.
    *   Analyze the outcome of the candle following the pattern (C2) relative to key levels of the pattern candles (C0, C1).
    *   Aggregate statistics by the hour of pattern formation (NY Time).
    *   Generate example plots of identified patterns.
*   Use Cursor for code generation, modification, explanation, and debugging.
*   Experiment with different pattern definition parameters.
*   **Implement Net Change Analysis as an additional feature.**

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
*   **Source:** Initially planned: NASDAQ 100 hourly data (using QQQ ETF as a proxy via `yfinance`) from January 1, 2010, to present. **Currently using sample hourly data provided in `frd_sample_futures_NQ/NQ_1hour_sample.csv` due to yfinance historical data limitations.**
*   **Timespan:** Sample data covers a limited recent period. Full analysis requires data from January 1, 2010, to present.
*   **Format:** CSV file (`nasdaq_h1_2010_present.csv` originally planned, now using `frd_sample_futures_NQ/NQ_1hour_sample.csv`) with columns: `Timestamp` (or `timestamp`), `Open`, `High`, `Low`, `Close`, `Volume`.

### 2.3. Existing Code Files
You should have the following files:
*   `scrapers/download_ticker_data.py`
*   `analyzers/hourly_swing_failure_analyzer.py` (Modified to use sample data and refined pattern definitions)

## 3. Project Setup & Structure in Cursor

### 3.1. Folder Structure

NQ_ANALYZE/
├── analyzers/
│ └── hourly_swing_failure_analyzer.py
├── scrapers/
│ └── download_ticker_data.py
├── frd_sample_futures_NQ/ (Contains sample data)
│   └── NQ_1hour_sample.csv
├── plots_swing_failures/ (Created by the analyzer script)
├── nasdaq_h1_2010_present.csv (Original target data file - currently not used for analysis)
└── README.md
└── PRD.md (this file)


### 3.2. Opening Project in Cursor
1.  Launch Cursor.
2.  Open the `NQ_ANALYZE` folder as your project workspace.

## 4. Core Analysis Workflow with Cursor (Completed using sample data)

### 4.1. Step 1: Data Acquisition (Running `download_ticker_data.py`)

*   Attempted to use `scrapers/download_ticker_data.py` to acquire historical data from 2010. Encountered limitations with `yfinance` for the requested historical depth of hourly data.
*   **Proceeded with using provided sample hourly data located in `frd_sample_futures_NQ/NQ_1hour_sample.csv`.**

### 4.2. Step 2: Swing Failure Analysis (Running `analyzers/hourly_swing_failure_analyzer.py`)

*   Successfully ran the `analyzers/hourly_swing_failure_analyzer.py` script using the sample data.
*   The script produced the statistical summary table and generated example plots in the `plots_swing_failures/` directory.

### 4.3. Step 3: Refining Pattern Definitions with Cursor

*   Successfully refined the `is_bearish_swing_failure` and `is_bullish_swing_failure` functions based on the specific conditions mentioned in this section and the user's instructions.

### 4.4. Step 4: Refining Statistical Aggregation with Cursor (Optional)

*   Successfully modified the `aggregate_and_print_stats` function to calculate the 'Hit%' as a percentage of the total number of bearish (or bullish) patterns found across all hours, as requested.

### 4.5. Step 5: Debugging with Cursor

*   Encountered a plotting error related to the `kaleido` library and successfully installed the required library to resolve the issue.

## 5. Key Parameters for Experimentation (via Cursor prompts)

Use Cursor to easily change and test these values in the pattern definition functions:

1.  **`C0_C1_RELATIVE_HEIGHT_THRESHOLD`:** (e.g., 0.9, 0.75, 0.5)
    *   Prompt: `"In `is_bearish_swing_failure` and `is_bullish_swing_failure`, change the condition where c1_height is compared to c0_height. Instead of `C0_C1_RELATIVE_HEIGHT_THRESHOLD`, use the value 0.75."`
2.  **`C1_RETRACEMENT_DEPTH_THRESHOLD`:** (e.g., 0.2, 0.5, 0.8 for how much C1 can retrace into C0 against the expected direction without invalidating the pattern)
    *   Prompt: `"Change the C1_RETRACEMENT_DEPTH_THRESHOLD to 0.2 in the script."`
3.  **Strictness of C1 close within C0 range:** Currently `c1['close'] < c0['high'] and c1['close'] > c0['low']`. You could ask Cursor to make it stricter, e.g., closing in the upper/lower half of C0.

## 6. Advanced: Asking Cursor for New Features/Analyses

### 6.1. Net Change Analysis (Next Step)

*   **Plan:** Create a new Python script to perform Net Change Analysis as described below.
1.  Create a new Python script.
2.  Load the data (currently using `frd_sample_futures_NQ/NQ_1hour_sample.csv`).
3.  Calculate the net percentage change for each hourly candle ( `(close - open) / open * 100` ).
4.  Filter out extreme outliers (e.g., keep data between the 3rd and 97th percentile of net changes).
5.  Calculate and print overall statistics for the filtered net changes: count, mean, median, standard deviation.
6.  Aggregate and print these net change statistics (mean, median, std_dev) per hour of the day (0-23 NY Time).

## 7. Expected Deliverables
*   A functioning Python environment for the analysis.
*   Sample data file `frd_sample_futures_NQ/NQ_1hour_sample.csv` used for analysis.
*   Modified `analyzers/hourly_swing_failure_analyzer.py` script with refined pattern definitions and statistical aggregation.
*   Console output of the aggregated H1 Swing Failure statistics table.
*   PNG images of example patterns in the `plots_swing_failures/` directory.
*   A new script for "Net Change" analysis and its output.

## 8. Notes & Best Practices for using Cursor
*   **Be Specific:** The more specific your prompt to Cursor (Ctrl+K or chat), the better the results.
*   **Iterate:** Don't expect perfect code on the first try. Ask Cursor for small changes, test, then ask for more.
*   **Select Code:** When asking for modifications to existing code, select the relevant lines/function before invoking Cursor.
*   **Use Chat (Ctrl+L):** For broader questions, explanations, or generating new script ideas.
*   **Use "Edit with AI" (Ctrl+K on selection):** For targeted code modifications.
*   **Review AI Output:** Always review code generated or modified by AI to ensure it meets your requirements and is correct.
*   **Version Control (Git):** Highly recommended. After each significant successful step or modification, commit your changes. Cursor has Git integration. The "U" next to your filenames indicates untracked/modified files.


