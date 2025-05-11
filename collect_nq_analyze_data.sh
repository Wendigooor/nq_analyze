#!/bin/bash

# Output file name
OUTPUT_FILE="nq_analyze_summary.txt"

# Create the output file and add a header
echo "NQ Analyze Project Summary for LLM" > "$OUTPUT_FILE"
echo "=================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 1. Collect PRD.md
echo "Collecting PRD.md..."
echo "===== PRD.md =====" >> "$OUTPUT_FILE"
if [ -f "PRD.md" ]; then
    cat "PRD.md" >> "$OUTPUT_FILE"
else
    echo "PRD.md not found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 2. Collect download_ticker_data.py
echo "Collecting download_ticker_data.py..."
echo "===== download_ticker_data.py =====" >> "$OUTPUT_FILE"
if [ -f "scrapers/download_ticker_data.py" ]; then
    cat "scrapers/download_ticker_data.py" >> "$OUTPUT_FILE"
else
    echo "download_ticker_data.py not found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 3. Collect hourly_swing_failure_analyzer.py
echo "Collecting hourly_swing_failure_analyzer.py..."
echo "===== hourly_swing_failure_analyzer.py =====" >> "$OUTPUT_FILE"
if [ -f "analyzers/hourly_swing_failure_analyzer.py" ]; then
    cat "analyzers/hourly_swing_failure_analyzer.py" >> "$OUTPUT_FILE"
else
    echo "hourly_swing_failure_analyzer.py not found." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 4. Add a note about the data
echo "===== Data Note =====" >> "$OUTPUT_FILE"
echo "Historical NASDAQ futures (NQ) data (1H, 5M, 30M OHLC) is available in frd_sample_futures_NQ/." >> "$OUTPUT_FILE"
echo "The data was sourced from TradingView and covers a limited recent period." >> "$OUTPUT_FILE"
echo "For a more comprehensive analysis, obtain full data from 2010-present (e.g., from FirstRateData or PortaraCQG)." >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 5. Example result (based on a typical output for Swing Failure or Midnight Open Snap analysis)
echo "===== Example Result =====" >> "$OUTPUT_FILE"
echo "Below is an example statistical output from the analysis (assumed from hourly_swing_failure_analyzer.py or midnight_open_analyzer.py)." >> "$OUTPUT_FILE"
echo "This example reflects the format expected for Swing Failure or Midnight Open Snap analysis:" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Example output for Swing Failure analysis (if the image was for that)
echo "Aggregated 3-candle swing-failure stats by Previous Day's Distribution (NY time):" >> "$OUTPUT_FILE"
echo "Prev Day      | Hours    | Bear Nr. | Hit%_Mid | Hit%_1st | Hit%_Opn | Avg_SL | Avg_RE | Hit_T% | Hit_SL% | Bull Nr. | Hit%_Mid_bull | Hit%_1st_bull | Hit%_Opn_bull | Avg_SL_bull | Avg_RE_bull | Hit_T%_bull | Hit_SL%_bull" >> "$OUTPUT_FILE"
echo "------------------------------------------------------------------------------------------------------------------------------" >> "$OUTPUT_FILE"
echo "Flat          | 09,10,11 | 85       | 62.35    | 48.24    | 41.18    | 12.50  | 25.30  | 52.94  | 35.29   | 70       | 58.57         | 45.71         | 38.57         | 11.80       | 22.40       | 50.00       | 32.86" >> "$OUTPUT_FILE"
echo "Slightly Up   | 10,11,12 | 92       | 65.22    | 50.00    | 43.48    | 13.20  | 27.10  | 55.43  | 34.78   | 65       | 60.00         | 46.15         | 40.00         | 12.10       | 23.90       | 53.85       | 30.77" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Example output for Midnight Open Snap analysis (if the image was for that)
echo "Midnight Open Snap Analysis (08:30-12:00 ET, NQ 1H data):" >> "$OUTPUT_FILE"
echo "Weekday | Total Sessions | Touches | Touch Probability | Snap Back | Minute 0 | Minute 15 | Minute 30 | Minute 45 | Gap Fill Probability" >> "$OUTPUT_FILE"
echo "-------------------------------------------------------------------------------------------------------------" >> "$OUTPUT_FILE"
echo "Monday  | 50             | 32      | 64.00%           | 25        | 10       | 8         | 9         | 5         | 52.00%" >> "$OUTPUT_FILE"
echo "Tuesday | 52             | 35      | 67.31%           | 28        | 12       | 9         | 8         | 6         | 55.77%" >> "$OUTPUT_FILE"
echo "Wednesday | 51           | 34      | 66.67%           | 27        | 11       | 9         | 7         | 7         | 54.90%" >> "$OUTPUT_FILE"
echo "Thursday | 50            | 33      | 66.00%           | 26        | 10       | 8         | 8         | 7         | 50.00%" >> "$OUTPUT_FILE"
echo "Friday  | 48             | 28      | 58.33%           | 20        | 8        | 7         | 6         | 7         | 47.92%" >> "$OUTPUT_FILE"
echo "-------------------------------------------------------------------------------------------------------------" >> "$OUTPUT_FILE"
echo "Overall Touch Probability: 64.58%" >> "$OUTPUT_FILE"
echo "Overall Gap Fill Probability: 52.12%" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Data collected into $OUTPUT_FILE."