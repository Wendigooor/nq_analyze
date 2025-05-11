import pandas as pd
import os
import glob
from datetime import datetime

# Найти самую последнюю директорию с результатами
results_dirs = glob.glob("analysis_results_tradingview/*/")
latest_dir = max(results_dirs, key=os.path.getmtime)
print(f"Анализ результатов из: {latest_dir}")

# Загрузить сырые данные о паттернах
raw_file = os.path.join(latest_dir, "raw_patterns_tradingview_hourly.csv")
df = pd.read_csv(raw_file)

print(f"Загружено {len(df)} паттернов")
print(f"Медвежьих паттернов: {len(df[df['type'] == 'bearish'])}")
print(f"Бычьих паттернов: {len(df[df['type'] == 'bullish'])}")

# Анализ по часу дня
print("\n=== ЧАСТОТА ПАТТЕРНОВ ПО ЧАСАМ (сортировка по убыванию) ===")
hour_counts = df.groupby(['hour_c0', 'type']).size().reset_index(name='count')
for pattern_type in ['bearish', 'bullish']:
    pattern_hours = hour_counts[hour_counts['type'] == pattern_type].sort_values('count', ascending=False)
    print(f"\n{pattern_type.upper()} ПАТТЕРНЫ:")
    print(pattern_hours[['hour_c0', 'count']].to_string(index=False))

# Анализ по предыдущему дневному изменению
print("\n=== ЧАСТОТА ПАТТЕРНОВ ПО ПРЕДЫДУЩЕМУ ДНЕВНОМУ ИЗМЕНЕНИЮ (сортировка по убыванию) ===")
prev_day_counts = df.groupby(['prev_1d_cat', 'type']).size().reset_index(name='count')
for pattern_type in ['bearish', 'bullish']:
    pattern_prevday = prev_day_counts[prev_day_counts['type'] == pattern_type].sort_values('count', ascending=False)
    print(f"\n{pattern_type.upper()} ПАТТЕРНЫ:")
    print(pattern_prevday[['prev_1d_cat', 'count']].to_string(index=False))

# Расчет процента успеха для медвежьих паттернов (swept_mid, swept_first, swept_open)
print("\n=== ПРОЦЕНТ УСПЕХА МЕДВЕЖЬИХ ПАТТЕРНОВ ПО ЧАСАМ (сортировка по Hit%_1st) ===")
bear_df = df[df['type'] == 'bearish']
bear_success = bear_df.groupby('hour_c0').agg({
    'swept_mid': ['count', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_first': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_open': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'hit_target': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0]
}).reset_index()

# Переименовать колонки для лучшей читабельности
bear_success.columns = ['hour', 'count', 'hit_mid_pct', 'swept_first_sum', 'hit_first_pct', 
                        'swept_open_sum', 'hit_open_pct', 'hit_target_sum', 'hit_target_pct']

# Отсортировать по убыванию процента hit_first_pct
bear_success_sorted = bear_success.sort_values('hit_first_pct', ascending=False)
print(bear_success_sorted[['hour', 'count', 'hit_mid_pct', 'hit_first_pct', 'hit_open_pct', 'hit_target_pct']].to_string(index=False))

# Расчет процента успеха для бычьих паттернов
print("\n=== ПРОЦЕНТ УСПЕХА БЫЧЬИХ ПАТТЕРНОВ ПО ЧАСАМ (сортировка по Hit%_1st) ===")
bull_df = df[df['type'] == 'bullish']
bull_success = bull_df.groupby('hour_c0').agg({
    'swept_mid': ['count', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_first': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_open': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'hit_target': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0]
}).reset_index()

# Переименовать колонки для лучшей читабельности
bull_success.columns = ['hour', 'count', 'hit_mid_pct', 'swept_first_sum', 'hit_first_pct', 
                        'swept_open_sum', 'hit_open_pct', 'hit_target_sum', 'hit_target_pct']

# Отсортировать по убыванию процента hit_first_pct
bull_success_sorted = bull_success.sort_values('hit_first_pct', ascending=False)
print(bull_success_sorted[['hour', 'count', 'hit_mid_pct', 'hit_first_pct', 'hit_open_pct', 'hit_target_pct']].to_string(index=False))

# Анализ по предыдущему дневному изменению и проценту успеха
print("\n=== ТОП-10 КОМБИНАЦИЙ ПАТТЕРНОВ ПО ЧАСАМ И ПРЕДЫДУЩЕМУ ДНЕВНОМУ ИЗМЕНЕНИЮ (сортировка по hit_first_pct) ===")
combined_success = df.groupby(['type', 'prev_1d_cat', 'hour_c0']).agg({
    'swept_mid': ['count', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_first': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'swept_open': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'hit_target': ['sum', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0]
}).reset_index()

# Переименовать колонки
combined_success.columns = ['type', 'prev_day', 'hour', 'count', 'hit_mid_pct', 
                           'swept_first_sum', 'hit_first_pct', 'swept_open_sum', 
                           'hit_open_pct', 'hit_target_sum', 'hit_target_pct']

# Минимальное количество паттернов для статистической значимости
min_count = 5

# Фильтр и сортировка
filtered_combined = combined_success[combined_success['count'] >= min_count]

# Для медвежьих паттернов
bear_combined = filtered_combined[filtered_combined['type'] == 'bearish'].sort_values('hit_first_pct', ascending=False)
print("\nМЕДВЕЖЬИ ПАТТЕРНЫ (мин. частота {}):".format(min_count))
print(bear_combined[['prev_day', 'hour', 'count', 'hit_mid_pct', 'hit_first_pct', 'hit_open_pct', 'hit_target_pct']].head(10).to_string(index=False))

# Для бычьих паттернов
bull_combined = filtered_combined[filtered_combined['type'] == 'bullish'].sort_values('hit_first_pct', ascending=False)
print("\nБЫЧЬИ ПАТТЕРНЫ (мин. частота {}):".format(min_count))
print(bull_combined[['prev_day', 'hour', 'count', 'hit_mid_pct', 'hit_first_pct', 'hit_open_pct', 'hit_target_pct']].head(10).to_string(index=False))

# Вывод комбинаций с наилучшим соотношением Hit Target % к Hit Stop Loss %
print("\n=== ТОП-10 КОМБИНАЦИЙ С ЛУЧШИМ СООТНОШЕНИЕМ HIT TARGET/HIT STOP LOSS ===")

# Добавим расчет отношения target/stop loss
ratio_df = df.groupby(['type', 'prev_1d_cat', 'hour_c0']).agg({
    'hit_target': ['count', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0],
    'hit_stop_loss': [lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0]
}).reset_index()

ratio_df.columns = ['type', 'prev_day', 'hour', 'count', 'hit_target_pct', 'hit_sl_pct']
ratio_df['target_sl_ratio'] = ratio_df['hit_target_pct'] / ratio_df['hit_sl_pct'].replace(0, float('inf'))

# Фильтр по минимальному количеству
ratio_filtered = ratio_df[ratio_df['count'] >= min_count]

# Сортировка по отношению target/stop loss
ratio_sorted = ratio_filtered.sort_values('target_sl_ratio', ascending=False)

# Вывод для медвежьих паттернов
bear_ratio = ratio_sorted[ratio_sorted['type'] == 'bearish']
print("\nМЕДВЕЖЬИ ПАТТЕРНЫ (мин. частота {}):".format(min_count))
print(bear_ratio[['prev_day', 'hour', 'count', 'hit_target_pct', 'hit_sl_pct', 'target_sl_ratio']].head(10).to_string(index=False))

# Вывод для бычьих паттернов
bull_ratio = ratio_sorted[ratio_sorted['type'] == 'bullish']
print("\nБЫЧЬИ ПАТТЕРНЫ (мин. частота {}):".format(min_count))
print(bull_ratio[['prev_day', 'hour', 'count', 'hit_target_pct', 'hit_sl_pct', 'target_sl_ratio']].head(10).to_string(index=False)) 