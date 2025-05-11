# PRD: NASDAQ H1 Swing Failure Statistical Analyzer with Cursor

# nq_analyze

## 7. Expected Deliverables
*   A functioning Python environment for the analysis.
*   Sample data files used for analysis (`frd_sample_futures_NQ/NQ_1hour_sample.csv`, `frd_sample_futures_NQ/NQ_30min_sample.csv`, `frd_sample_futures_NQ/NQ_5min_sample.csv`).
*   Modified `analyzers/hourly_swing_failure_analyzer.py` script.
*   `analyzers/net_change_analyzer.py` script.
*   New scripts for 30-minute (`analyzers/thirty_min_swing_failure_analyzer.py`) and 5-minute (`analyzers/five_min_swing_failure_analyzer.py`) analysis.
*   New script for cross-timeframe correlation (`analyzers/cross_timeframe_analyzer.py`).
*   Basic test suite in the `tests/` directory.
*   Analysis results saved in timestamped directories (`analysis_results/`, `analysis_results_30min/`, `analysis_results_5min/`).
*   Console output of statistical tables.
*   PNG images of example patterns.
*   Documentation of findings from historical distribution analysis, cross-timeframe analysis, and general correlation analysis (part of PRD updates and potential separate documentation).

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
    *(Note: `yfinance` is used for data acquisition attempts but sample data is currently used for analysis.)*
4.  **Data:** Ensure the sample data files (`NQ_1hour_sample.csv`, `NQ_30min_sample.csv`, `NQ_5min_sample.csv`) are placed in the `frd_sample_futures_NQ/` directory.
5.  **Run Analysis Scripts:** Execute the analyzer scripts from the project root directory with the virtual environment active:
    *   Hourly Analysis: `python analyzers/hourly_swing_failure_analyzer.py`
    *   30-Minute Analysis: `python analyzers/thirty_min_swing_failure_analyzer.py`
    *   5-Minute Analysis: `python analyzers/five_min_swing_failure_analyzer.py`
    *   Cross-Timeframe Analysis: `python analyzers/cross_timeframe_analyzer.py`

    Each analysis script run will create a new timestamped directory within the respective `analysis_results*` folder containing the statistical summary (`swing_failure_stats.txt`), configuration (`configuration.txt`), raw pattern results (`raw_pattern_results*.csv`), and example plots (`plots/`).

6.  **Run Tests:** Execute the tests from the project root directory with the virtual environment active:
    ```bash
    python -m unittest discover tests
    ```

## 9. Expected Output and Key Insights

Running the analysis scripts will output statistical summaries to the console and save detailed results (stats, raw data, plots) to timestamped directories under `analysis_results/`, `analysis_results_30min/`, and `analysis_results_5min/`. The `cross_timeframe_analyzer.py` script will output correlation findings to the console.

**Key Insights from Analysis Runs (based on sample data):**

*   **Data Limitations:** Initial attempts to acquire historical hourly QQQ data from 2010 using `yfinance` were unsuccessful; analysis is currently based on provided sample data for 1-hour, 30-minute, and 5-minute intervals.
*   **Pattern Refinement:** The swing failure pattern definitions were refined based on specific criteria (C1 close within C0, C1 height, C1 retracement depth).
*   **Statistical Aggregation:** Statistical reporting was updated to calculate 'Hit%' relative to the total number of patterns of that type.
*   **Previous Day's Distribution (Hourly):** Analysis was implemented to categorize patterns based on the preceding day's price change and group statistics accordingly. (Note: This is currently commented out for lower timeframes).
*   **5-Minute and 30-Minute Analysis:** Analysis scripts for 5-minute and 30-minute timeframes have been created and run, generating statistical summaries and raw pattern data for these timeframes.
*   **Cross-Timeframe Correlations:** The `cross_timeframe_analyzer.py` script has been created and run to analyze correlations between patterns found in different timeframes, focusing on how the presence and type of nested smaller-timeframe patterns within larger-timeframe patterns relate to the larger pattern's outcome (specifically Hit%). Initial findings suggest some correlation, which requires further detailed analysis.
*   **Risk to Reward (R:R) Analysis:** Initial R:R calculation logic has been integrated into the 5-minute analysis script. This includes calculating potential Stop Loss, Reward Excursion, and tracking if target or stop loss levels are hit within a defined window after C2. Full R:R integration into the 1-hour and 30-minute scripts, and incorporating R:R metrics into the cross-timeframe analysis, are planned for future steps.

Further analysis, particularly completing the Risk to Reward calculations across all timeframes and refining the cross-timeframe analysis with R:R data, is needed to identify high-probability, low-risk trading setups.

## 10. Notes & Best Practices for using Cursor
