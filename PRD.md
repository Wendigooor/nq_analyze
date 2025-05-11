# PRD: NASDAQ H1 Swing Failure Statistical Analyzer with Cursor

**Version:** 1.0
**Date:** 2025-05-10
**Author/User:** 

## 1. Project Overview

### 1.1. Objective
To replicate and potentially refine a statistical model for identifying and analyzing "H1 Swing Failure" patterns on historical NASDAQ (QQQ) hourly data. The primary goal is to understand the probabilities of certain price movements following these patterns and to explore how different pattern parameters affect these probabilities.

### 1.2. Scope
*   Acquire historical NASDAQ (QQQ) hourly data from 2010 to the present.
*   Implement Python scripts (assisted by Cursor) to:
    *   Define bearish and bullish H1 Swing Failure patterns (3-candle setup).
    *   Iterate through historical data to find these patterns.
    *   Analyze the outcome of the candle following the pattern (C2) relative to key levels of the pattern candles (C0, C1).
    *   Aggregate statistics by the hour of pattern formation (NY Time).
    *   Generate example plots of identified patterns.
*   Use Cursor for code generation, modification, explanation, and debugging.
*   Experiment with different pattern definition parameters.

### 1.3. Target User
A trader/analyst familiar with basic Python and trading concepts, looking to use Cursor to efficiently conduct statistical market studies.

## 2. Prerequisites

### 2.1. Software
*   **Cursor IDE:** Installed and configured.
*   **Python 3.x:** Installed.
*   **Python Virtual Environment:** Recommended (e.g., `venv_trading_stats`).
*   **Required Python Libraries:**
    *   `pandas`
    *   `numpy`
    *   `yfinance` (for data scraping)
    *   `plotly` (for plotting, or `matplotlib`)
    *   Ensure these are installed in your active virtual environment: `pip install pandas numpy yfinance plotly`

### 2.2. Data
*   **Source:** NASDAQ 100 hourly data (using QQQ ETF as a proxy via `yfinance`).
*   **Timespan:** January 1, 2010, to present.
*   **Format:** CSV file (`nasdaq_h1_2010_present.csv`) with columns: `Timestamp` (NY Time), `Open`, `High`, `Low`, `Close`, `Volume`.

### 2.3. Existing Code Files
You should have the following files from the previous plan (or similar):
*   `scrapers/download_ticker_data.py`
*   `analyzers/hourly_swing_failure_analyzer.py`

## 3. Project Setup & Structure in Cursor

### 3.1. Folder Structure

NQ_ANALYZE/
├── analyzers/
│ └── hourly_swing_failure_analyzer.py
├── scrapers/
│ └── download_ticker_data.py
├── plots_swing_failures/ (This will be created by the analyzer script)
├── nasdaq_h1_2010_present.csv (This will be created by the scraper script, in the root or a data/ folder)
└── README.md
└── PRD.md (this file)


### 3.2. Opening Project in Cursor
1.  Launch Cursor.
2.  Open the `NQ_ANALYZE` folder as your project workspace.

## 4. Core Analysis Workflow with Cursor

### 4.1. Step 1: Data Acquisition (Running `download_ticker_data.py`)

1.  **Open `scrapers/download_ticker_data.py` in Cursor.**
2.  **Review the Code with Cursor:**
    *   Highlight the `yf.download(...)` line.
    *   Ask Cursor (Ctrl+K or Cmd+K): `"Explain this yfinance download function and its parameters, specifically 'interval'."`
    *   Ask Cursor: `"How can I ensure the downloaded data's timestamp is in America/New_York timezone?"` (The script should already handle this or you can ask Cursor to add it).
3.  **Run the Script using Cursor's Terminal:**
    *   Open the integrated terminal (Ctrl+` or Cmd+`).
    *   Ensure your virtual environment is activated.
    *   Navigate to the `NQ_ANALYZE` root directory if not already there.
    *   Run: `python scrapers/download_ticker_data.py`
4.  **Verify Output:**
    *   Check that `nasdaq_h1_2010_present.csv` is created in the project root (or specified location).
    *   Open the CSV in Cursor or a spreadsheet program to briefly inspect the data.

### 4.2. Step 2: Swing Failure Analysis (Running `analyzers/hourly_swing_failure_analyzer.py`)

1.  **Open `analyzers/hourly_swing_failure_analyzer.py` in Cursor.**
2.  **Understand Key Functions with Cursor:**
    *   Highlight the `is_bearish_swing_failure` function.
    *   Ask Cursor: `"Explain each condition in this function for identifying a bearish swing failure."`
    *   Do the same for `is_bullish_swing_failure`.
    *   Highlight the `analyze_swing_failures` function.
    *   Ask Cursor: `"Explain how this function iterates through the data and what 'swept_mid', 'swept_first', and 'swept_open' represent."`
    *   Highlight the `aggregate_and_print_stats` function.
    *   Ask Cursor: `"Explain how the hourly aggregation works and how the hit rates are calculated."`
3.  **Run the Analysis Script:**
    *   In Cursor's terminal (ensure venv is active):
        `python analyzers/hourly_swing_failure_analyzer.py`
4.  **Review Output:**
    *   **Console Table:** Observe the printed statistical table. Compare its structure to the presenter's.
    *   **Plots:** Check the `plots_swing_failures/` directory for generated PNG images of example patterns. Open them to visually verify the pattern detection.

### 4.3. Step 3: Refining Pattern Definitions with Cursor

This is where you match the presenter's specific nuances or explore your own.

1.  **Focus on `is_bearish_swing_failure` and `is_bullish_swing_failure` functions.**
2.  **Presenter's Conditions (from video):**
    *   **Bearish C1 Low:** `c1["low"] > c0["low"] + 0.2 * c0_height` (C1 low doesn't reach below 20% from C0 low, effectively meaning it doesn't reach the C0 *open* if C0 was purely body).
    *   **Bullish C1 High:** `c1["high"] < c0["high"] - 0.5 * c0_height` (C1 high doesn't reach above 50% from C0 high, if C0 was purely body).
    *   **C1 Height:** `c1_height <= 0.9 * c0_height` (or sometimes mentioned 0.5).
3.  **Example Cursor Prompts for Modification:**
    *   Select the existing condition for C1 low in `is_bearish_swing_failure`.
    *   Ask Cursor (Ctrl+L for chat, or select and Ctrl+K for edit): `"Modify this condition. For a bearish swing failure, I want to ensure that the low of candle c1 (c1['low']) does not go below the open of candle c0 (c0['open']). Also, add a condition that the height of c1 (abs(c1['open'] - c1['close'])) must be less than or equal to 90% of the height of c0 (abs(c0['open'] - c0['close']))." `
    *   Similarly for bullish: `"Modify the conditions for 'is_bullish_swing_failure'. I want to ensure c1['high'] does not go above c0['open']. Also, ensure c1's height is no more than 50% of c0's height."`
    *   After modifications, re-run the analyzer script to see how statistics change.

### 4.4. Step 4: Refining Statistical Aggregation with Cursor (Optional)

*   If the output table's "Hit%" calculation is different from the presenter's.
*   Ask Cursor (referencing the `aggregate_and_print_stats` function): `"Modify the 'bear_hit_rate' and 'bull_hit_rate' calculation. Instead of a percentage of total patterns for that hour, calculate it as a percentage of the total number of bearish (or bullish) patterns found across all hours."` (This might not be what the presenter actually did, his "Hit%" seemed low like a raw count/percentage of overall occurrences). Or: `"How would I calculate the 'Hit%' as the number of SFP occurrences for that hour divided by the total number of 3-candle sequences possible for that hour across the entire dataset?"`

### 4.5. Step 5: Debugging with Cursor

*   **Problem Indication:** Cursor often shows squiggly lines or indicators for potential problems (like the "2 problems in this file" in your `download_ticker_data.py` screenshot).
*   **Using Cursor for Debugging:**
    *   Hover over the problem indicator to see the tooltip.
    *   Select the problematic code.
    *   Ask Cursor: `"What's wrong with this code?"` or `"Fix this error: [paste error message from terminal]"`.
    *   If the script runs but gives unexpected results (e.g., zero patterns found):
        *   Ask Cursor: `"I'm not finding any patterns. Can you help me debug why 'is_bearish_swing_failure' might always be returning False? Let's print the values of c0 and c1 inside the loop when a potential pattern is being checked."`
        *   Use Cursor's "Debug" feature (if available and configured for your Python interpreter) to step through the code.

## 5. Key Parameters for Experimentation (via Cursor prompts)

Use Cursor to easily change and test these values in the pattern definition functions:

1.  **`C0_C1_RELATIVE_HEIGHT_THRESHOLD`:** (e.g., 0.9, 0.75, 0.5)
    *   Prompt: `"In `is_bearish_swing_failure` and `is_bullish_swing_failure`, change the condition where c1_height is compared to c0_height. Instead of `C0_C1_RELATIVE_HEIGHT_THRESHOLD`, use the value 0.75."`
2.  **`C1_RETRACEMENT_DEPTH_THRESHOLD`:** (e.g., 0.2, 0.5, 0.8 for how much C1 can retrace into C0 against the expected direction without invalidating the pattern)
    *   Prompt: `"Change the C1_RETRACEMENT_DEPTH_THRESHOLD to 0.2 in the script."`
3.  **Strictness of C1 close within C0 range:** Currently `c1['close'] < c0['high'] and c1['close'] > c0['low']`. You could ask Cursor to make it stricter, e.g., closing in the upper/lower half of C0.

## 6. Advanced: Asking Cursor for New Features/Analyses

### 6.1. Net Change Analysis (as mentioned by presenter)
1.  Ask Cursor: `"Create a new Python script. This script should:
    1. Load the 'nasdaq_h1_2010_present.csv' data.
    2. Calculate the net percentage change for each hourly candle ( (close - open) / open * 100 ).
    3. Filter out extreme outliers (e.g., keep data between the 3rd and 97th percentile of net changes).
    4. Calculate and print overall statistics for the filtered net changes: count, mean, median, standard deviation.
    5. Calculate and print standard deviation thresholds (e.g., mean +/- 0.5*std, +/- 1.0*std, +/- 1.5*std, +/- 2.0*std).
    6. Aggregate and print these net change statistics (mean, median, std_dev) per hour of the day (0-23 NY Time)." `

## 7. Expected Deliverables
*   A functioning Python environment for the analysis.
*   `nasdaq_h1_2010_present.csv` containing the historical data.
*   Modified `analyzers/hourly_swing_failure_analyzer.py` script.
*   Console output of the aggregated H1 Swing Failure statistics table.
*   PNG images of example patterns in the `plots_swing_failures/` directory.
*   (Optional) A new script for "Net Change" analysis and its output.

## 8. Notes & Best Practices for using Cursor
*   **Be Specific:** The more specific your prompt to Cursor (Ctrl+K or chat), the better the results.
*   **Iterate:** Don't expect perfect code on the first try. Ask Cursor for small changes, test, then ask for more.
*   **Select Code:** When asking for modifications to existing code, select the relevant lines/function before invoking Cursor.
*   **Use Chat (Ctrl+L):** For broader questions, explanations, or generating new script ideas.
*   **Use "Edit with AI" (Ctrl+K on selection):** For targeted code modifications.
*   **Review AI Output:** Always review code generated or modified by AI to ensure it meets your requirements and is correct.
*   **Version Control (Git):** Highly recommended. After each significant successful step or modification, commit your changes. Cursor has Git integration. The "U" next to your filenames indicates untracked/modified files.


