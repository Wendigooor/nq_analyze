#!/bin/bash

# Название выходного файла
OUTPUT_FILE="nq_analyze_summary.txt"

# Временная папка для клонирования репозитория
TEMP_DIR="nq_analyze_temp"
REPO_URL="https://github.com/Wendigooor/nq_analyze.git"

# Создаем выходной файл и добавляем заголовок
echo "NQ Analyze Project Summary for LLM" > "$OUTPUT_FILE"
echo "=================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Клонируем репозиторий
echo "Cloning repository..."
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi
git clone "$REPO_URL" "$TEMP_DIR"

# Проверяем, успешно ли клонировался репозиторий
if [ ! -d "$TEMP_DIR" ]; then
    echo "Error: Failed to clone repository." >> "$OUTPUT_FILE"
    exit 1
fi

cd "$TEMP_DIR"

# 1. Копируем PRD.md
echo "Collecting PRD.md..."
echo "===== PRD.md =====" >> "../$OUTPUT_FILE"
if [ -f "PRD.md" ]; then
    cat "PRD.md" >> "../$OUTPUT_FILE"
else
    echo "PRD.md not found." >> "../$OUTPUT_FILE"
fi
echo "" >> "../$OUTPUT_FILE"

# 2. Копируем download_ticker_data.py
echo "Collecting download_ticker_data.py..."
echo "===== download_ticker_data.py =====" >> "../$OUTPUT_FILE"
if [ -f "scrapers/download_ticker_data.py" ]; then
    cat "scrapers/download_ticker_data.py" >> "../$OUTPUT_FILE"
else
    echo "download_ticker_data.py not found." >> "../$OUTPUT_FILE"
fi
echo "" >> "../$OUTPUT_FILE"

# 3. Копируем hourly_swing_failure_analyzer.py
echo "Collecting hourly_swing_failure_analyzer.py..."
echo "===== hourly_swing_failure_analyzer.py =====" >> "../$OUTPUT_FILE"
if [ -f "analyzers/hourly_swing_failure_analyzer.py" ]; then
    cat "analyzers/hourly_swing_failure_analyzer.py" >> "../$OUTPUT_FILE"
else
    echo "hourly_swing_failure_analyzer.py not found." >> "../$OUTPUT_FILE"
fi
echo "" >> "../$OUTPUT_FILE"

# 4. Добавляем комментарий о данных
echo "===== Data Note =====" >> "../$OUTPUT_FILE"
echo "Note: Historical NASDAQ futures (NQ) data (1H OHLC) for 2010-present is required." >> "../$OUTPUT_FILE"
echo "This script does not download the data due to unavailability of free sources." >> "../$OUTPUT_FILE"
echo "Please obtain the data manually (e.g., from FirstRateData or PortaraCQG) and save as nasdaq_h1_2010_present.csv." >> "../$OUTPUT_FILE"
echo "" >> "../$OUTPUT_FILE"

# 5. Пример результата (заглушка, так как данных нет)
echo "===== Example Result =====" >> "../$OUTPUT_FILE"
echo "Below is a placeholder for the statistical output of Swing Failure analysis." >> "../$OUTPUT_FILE"
echo "Since actual data is not available, this is a sample format based on PRD expectations:" >> "../$OUTPUT_FILE"
echo "" >> "../$OUTPUT_FILE"
echo "Hour (NY Time) | Bearish SFP Count | Bullish SFP Count | Bear Hit% | Bull Hit%" >> "../$OUTPUT_FILE"
echo "---------------------------------------------------------------" >> "../$OUTPUT_FILE"
echo "09:00         | 150              | 120              | 45%       | 38%" >> "../$OUTPUT_FILE"
echo "10:00         | 130              | 110              | 42%       | 35%" >> "../$OUTPUT_FILE"
echo "" >> "../$OUTPUT_FILE"

# Очистка
cd ..
rm -rf "$TEMP_DIR"

echo "Data collected into $OUTPUT_FILE."