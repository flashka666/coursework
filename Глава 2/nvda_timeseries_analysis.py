"""
Курсовой проект – Глава 2: Первичный анализ набора данных с временными рядами
Датасет: NVIDIA Daily Stock Prices Dataset (Kaggle, Ibrahim Shahrukh)
https://www.kaggle.com/datasets/ibrahimshahrukh/nvidia-daily-stock-prices-20162026-dataset

Запуск:
    pip install pandas numpy matplotlib seaborn statsmodels scipy
    python nvda_timeseries_analysis.py

Все рисунки сохраняются в папку ./figures_ch2/

Соответствие рисунков и подписей в документе (Глава 2):
  fig01 -> Рисунок 1.  Многомерный временной ряд NVDA (2016–2025) по всем пяти каналам
  fig02 -> Рисунок 2.  Цена закрытия и объём торгов с разделением на обучающую/тестовую выборки
  fig03 -> Рисунок 3.  Среднее ± ст. отклонение ценовых каналов и динамика среднегодовой цены
  fig04 -> Рисунок 4.  Доля пропущенных значений и количество выбросов (3σ) по каналам
  fig05 -> Рисунок 5.  Диаграммы размаха (boxplot) по каналам для выявления выбросов
  fig06 -> Рисунок 6.  Сравнение диапазонов значений каналов NVDA без нормирования
  fig07 -> Рисунок 7.  Тепловая карта матрицы корреляции Пирсона между каналами NVDA
  fig08 -> Рисунок 8.  Аддитивная декомпозиция ряда Close: тренд, сезонность, остатки
  fig09 -> Рисунок 9.  Сигнал vs Шум (SNR = 15.86 дБ) и распределение остатков
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # для сохранения без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

# ── Настройки ──────────────────────────────────────────────────────────────
DATA_PATH  = 'NVDA_yfinance_clean.csv'   # путь к файлу датасета
OUT_DIR    = 'figures_ch2'
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

# ── Цвета ──────────────────────────────────────────────────────────────────
BLUE   = '#2196F3'
RED    = '#E53935'
GREEN  = '#43A047'
ORANGE = '#FB8C00'
PURPLE = '#8E24AA'
TEAL   = '#26C6DA'

# ── Загрузка и базовая предобработка ──────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

channels      = ['Open', 'High', 'Low', 'Close', 'Volume']
price_ch      = ['Open', 'High', 'Low', 'Close']
colors_ch     = [BLUE, TEAL, GREEN, ORANGE, PURPLE]
split_date    = pd.Timestamp('2024-01-01')
train         = df[df.index <  split_date]
test          = df[df.index >= split_date]
labels_ch     = ['Open (цена открытия)', 'High (макс. цена)',
                 'Low (мин. цена)', 'Close (цена закрытия)',
                 'Volume (объём торгов)']

print(f"Датасет загружен: {df.shape[0]} строк × {df.shape[1]} каналов")
print(f"Период: {df.index.min().date()} → {df.index.max().date()}")
print(f"Пропуски: {df.isnull().sum().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 1. Все пять каналов многомерного ряда
# Подпись: Рисунок 1. Многомерный временной ряд NVDA (2016–2025) по всем каналам
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
for ax, ch, col, lbl in zip(axes, channels, colors_ch, labels_ch):
    ax.plot(df.index, df[ch], color=col, lw=0.8, label=lbl)
    ax.set_ylabel(lbl, fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    ax.tick_params(axis='x', rotation=30)
    ax.axvline(split_date, color=RED, lw=1.5, ls='--', alpha=0.7)
axes[0].text(split_date, axes[0].get_ylim()[1] * 0.75,
             ' Граница\n обуч./тест.', color=RED, fontsize=8, va='top')
plt.suptitle(
    'Многомерный временной ряд NVDA (2016–2025)\n'
    'Вертикальная линия — граница обучающей и тестовой выборок',
    fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_raw_channels.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig01 saved  →  Рисунок 1. Многомерный временной ряд NVDA')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 2. Close + Volume с разделением обучение/тест
# Подпись: Рисунок 2. Цена закрытия и объём торгов с разделением на выборки
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(train.index, train['Close'], color=BLUE,   lw=1.2, label='Обучение (2016–2023)')
ax1.plot(test.index,  test['Close'],  color=ORANGE, lw=1.2, label='Тест (2024–2025)')
ax1.axvline(split_date, color=RED, lw=1.5, ls='--', alpha=0.8)
ax1.set_ylabel('Цена закрытия (USD)')
ax1.legend(); ax1.grid(alpha=0.25)
ax1.set_title('Цена закрытия (Close)', fontweight='bold')

ax2.bar(train.index, train['Volume'], color=BLUE,   alpha=0.5, width=1, label='Обучение')
ax2.bar(test.index,  test['Volume'],  color=ORANGE, alpha=0.5, width=1, label='Тест')
ax2.axvline(split_date, color=RED, lw=1.5, ls='--', alpha=0.8)
ax2.set_ylabel('Объём торгов (шт.)')
ax2.legend(); ax2.grid(alpha=0.25)
ax2.set_title('Объём торгов (Volume)', fontweight='bold')
ax2.tick_params(axis='x', rotation=30)

plt.suptitle('Временной ряд NVDA: цена закрытия и объём торгов',
             fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_close_volume_split.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig02 saved  →  Рисунок 2. Цена закрытия и объём торгов с разделением')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 3. Статистика: среднее ± std для ценовых каналов + среднегодовая цена
# Подпись: Рисунок 3. Среднее ± ст. отклонение ценовых каналов и динамика средн. цены
# ══════════════════════════════════════════════════════════════════════════════
stats    = df[channels].describe().T
means_p  = stats.loc[price_ch, 'mean']
stds_p   = stats.loc[price_ch, 'std']
yearly   = df['Close'].resample('YE').mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x_p  = np.arange(len(price_ch))
bars = axes[0].bar(x_p, means_p.values,
                   color=[BLUE, TEAL, GREEN, ORANGE], alpha=0.85,
                   yerr=stds_p.values, capsize=5,
                   error_kw=dict(ecolor='gray', lw=1.5))
axes[0].set_xticks(list(x_p)); axes[0].set_xticklabels(price_ch)
axes[0].set_title('Среднее ± ст. откл. ценовых каналов (USD)', fontweight='bold')
axes[0].set_ylabel('USD'); axes[0].grid(axis='y', alpha=0.3)
for bar, v in zip(bars, means_p.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{v:.1f}', ha='center', fontsize=8)

axes[1].bar(yearly.index.year, yearly.values,
            color=BLUE, alpha=0.85, edgecolor='white')
axes[1].set_title('Среднегодовая цена закрытия NVDA', fontweight='bold')
axes[1].set_xlabel('Год'); axes[1].set_ylabel('Средняя цена (USD)')
axes[1].grid(axis='y', alpha=0.3); axes[1].tick_params(axis='x', rotation=30)
for x, y in zip(yearly.index.year, yearly.values):
    axes[1].text(x, y + 1, f'{y:.1f}', ha='center', fontsize=7, rotation=45)

plt.suptitle('Статистические характеристики временного ряда NVDA',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_statistics.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig03 saved  →  Рисунок 3. Статистика ценовых каналов')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 4. Пропуски и выбросы 3σ (столбчатые диаграммы)
# Подпись: Рисунок 4. Доля пропущенных значений и количество выбросов (3σ) по каналам
# ══════════════════════════════════════════════════════════════════════════════
miss        = (df[channels].isnull().sum() / len(df) * 100).values
out_counts  = []
for ch in channels:
    s  = df[ch].dropna(); mu, sig = s.mean(), s.std()
    out_counts.append(int(((s < mu - 3 * sig) | (s > mu + 3 * sig)).sum()))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(channels, miss, color=colors_ch, alpha=0.85)
axes[0].set_title('Доля пропущенных значений по каналам (%)', fontweight='bold')
axes[0].set_ylabel('%'); axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(miss):
    axes[0].text(i, v + 0.02, f'{v:.3f}%', ha='center', fontsize=9)

axes[1].bar(channels, out_counts, color=colors_ch, alpha=0.85)
axes[1].set_title('Количество выбросов по правилу 3σ', fontweight='bold')
axes[1].set_ylabel('Количество'); axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(out_counts):
    axes[1].text(i, v + 0.3, str(v), ha='center', fontsize=10)

plt.suptitle('Анализ пропусков и выбросов', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig04_missing_outliers.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig04 saved  →  Рисунок 4. Пропуски и выбросы 3σ')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 5. Boxplot по каждому каналу
# Подпись: Рисунок 5. Диаграммы размаха (boxplot) по каналам для выявления выбросов
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(16, 5))
for ax, ch, col, n_out in zip(axes, channels, colors_ch, out_counts):
    s = df[ch].dropna()
    ax.boxplot(s, vert=True, patch_artist=True,
               boxprops=dict(facecolor=col, alpha=0.5),
               medianprops=dict(color=RED, lw=2),
               flierprops=dict(marker='o', ms=2, alpha=0.3, markerfacecolor=RED))
    ax.set_title(f'{ch}\n({n_out} выбр. 3σ)', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Диаграммы размаха (boxplot) по каналам для обнаружения выбросов',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig05_boxplots.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig05 saved  →  Рисунок 5. Boxplot по каналам')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 6. Диапазоны значений без нормирования
# Подпись: Рисунок 6. Сравнение диапазонов значений каналов временного ряда NVDA без нормирования
# Примечание: используется логарифмическая шкала Y, чтобы одновременно были видны
# ценовые каналы и Volume, так как Volume отличается по масштабу на несколько порядков.
# ══════════════════════════════════════════════════════════════════════════════
bp_data = [df[ch].dropna().values for ch in channels]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    bp_data,
    labels=channels,
    patch_artist=True,
    notch=False,
    medianprops=dict(color=RED, lw=2),
    flierprops=dict(marker='o', ms=2, alpha=0.3, markerfacecolor=RED)
)

for patch, col in zip(bp['boxes'], colors_ch):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)

ax.set_title(
    'Сравнение диапазонов значений каналов NVDA\n'
    'без нормирования, логарифмическая шкала',
    fontweight='bold'
)
ax.set_ylabel('Исходное значение (логарифмическая шкала)')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig06_ranges_raw_logscale.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig06 saved  →  Рисунок 6. Диапазоны значений без нормирования')

# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 7. Тепловая карта корреляции Пирсона
# Подпись: Рисунок 7. Тепловая карта матрицы корреляции Пирсона между каналами NVDA
# ══════════════════════════════════════════════════════════════════════════════
corr = df[channels].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, annot_kws={'size': 11},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Коэффициент Пирсона', 'shrink': 0.8})
ax.set_title('Матрица корреляции Пирсона каналов NVDA', fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig07_corr_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig07 saved  →  Рисунок 7. Тепловая карта корреляции')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 8. Аддитивная декомпозиция Close (недельная агрегация, period=52)
# Подпись: Рисунок 8. Аддитивная декомпозиция ряда Close: тренд, сезонность, остатки
# ══════════════════════════════════════════════════════════════════════════════
close_weekly = df['Close'].resample('W').mean().dropna()
result       = seasonal_decompose(close_weekly, model='additive', period=52)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(close_weekly.index, close_weekly.values, color=BLUE, lw=0.9)
axes[0].set_title('Исходный ряд (Close, недельная агрегация)', fontweight='bold')
axes[0].set_ylabel('USD'); axes[0].grid(alpha=0.3)

axes[1].plot(result.trend.index,    result.trend.values,    color=ORANGE, lw=1.5)
axes[1].set_title('Тренд', fontweight='bold')
axes[1].set_ylabel('USD'); axes[1].grid(alpha=0.3)

axes[2].plot(result.seasonal.index, result.seasonal.values, color=GREEN, lw=0.9)
axes[2].set_title('Сезонная компонента (период = 52 нед., ~1 год)', fontweight='bold')
axes[2].set_ylabel('USD'); axes[2].grid(alpha=0.3)

axes[3].plot(result.resid.index,    result.resid.values,    color=RED, lw=0.7, alpha=0.7)
axes[3].axhline(0, color='k', lw=0.5, ls='--')
axes[3].set_title('Остатки (шум)', fontweight='bold')
axes[3].set_ylabel('USD'); axes[3].grid(alpha=0.3)
axes[3].tick_params(axis='x', rotation=30)

plt.suptitle('Аддитивная декомпозиция временного ряда Close (NVDA)',
             fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig08_decomposition.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig08 saved  →  Рисунок 8. Аддитивная декомпозиция Close')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 9. SNR + гистограмма остатков
# Подпись: Рисунок 9. Сигнал vs Шум (SNR = 15.86 дБ) и распределение остатков
# ══════════════════════════════════════════════════════════════════════════════
resid    = result.resid.dropna()
trend    = result.trend.dropna()
common   = trend.index.intersection(resid.index)
signal_v = trend.loc[common] + result.seasonal.loc[common]
resid_v  = resid.loc[common]

snr   = 10 * np.log10(np.var(signal_v) / np.var(resid_v))
mu_r  = float(resid_v.mean())
std_r = float(resid_v.std())
skew  = float(sp_stats.skew(resid_v))
kurt  = float(sp_stats.kurtosis(resid_v))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(signal_v.index, signal_v.values, color=ORANGE, lw=1.2,
         label='Сигнал (тренд + сезонность)')
ax1.plot(resid_v.index,  resid_v.values,  color=RED,    lw=0.6, alpha=0.6,
         label='Шум (остатки)')
ax1.set_title(f'Сигнал vs Шум  |  SNR = {snr:.2f} дБ', fontweight='bold')
ax1.set_ylabel('USD'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.tick_params(axis='x', rotation=30)

ax2.hist(resid_v.values, bins=55, color=BLUE, edgecolor='white', alpha=0.85, density=True)
x_r = np.linspace(resid_v.min(), resid_v.max(), 300)
ax2.plot(x_r, sp_stats.norm.pdf(x_r, mu_r, std_r), 'r-', lw=2, label='Норм. распр.')
ax2.set_title('Распределение остатков (шума)', fontweight='bold')
ax2.set_xlabel('Остатки (USD)'); ax2.set_ylabel('Плотность')
ax2.legend(); ax2.grid(alpha=0.3)
ax2.text(0.97, 0.95,
         f'Среднее:  {mu_r:.3f}\nСт. откл.: {std_r:.3f}\nАсим.:     {skew:.3f}\nЭксцесс:  {kurt:.3f}',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('Оценка шума и отношение сигнал/шум (SNR)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig09_snr_residuals.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig09 saved  →  Рисунок 9. SNR и распределение остатков')

print(f'\n✓ Все 9 рисунков сохранены в папку ./{OUT_DIR}/')
print(f'   SNR = {snr:.2f} дБ | Асим. = {skew:.3f} | Эксцесс = {kurt:.3f}')
print("""
Таблица соответствия файлов и подписей (Глава 2):
  figures_ch2/fig01_raw_channels.png       → Рисунок 1.  Многомерный ряд NVDA по всем каналам
  figures_ch2/fig02_close_volume_split.png → Рисунок 2.  Close и Volume с границей выборок
  figures_ch2/fig03_statistics.png         → Рисунок 3.  Среднее ± ст. откл. + среднегодовая цена
  figures_ch2/fig04_missing_outliers.png   → Рисунок 4.  Пропуски и выбросы 3σ по каналам
  figures_ch2/fig05_boxplots.png           → Рисунок 5.  Boxplot по каналам
  figures_ch2/fig06_ranges_raw_logscale.png→ Рисунок 6.  Диапазоны значений без нормирования
  figures_ch2/fig07_corr_heatmap.png       → Рисунок 7.  Тепловая карта корреляции Пирсона
  figures_ch2/fig08_decomposition.png      → Рисунок 8.  Аддитивная декомпозиция Close
  figures_ch2/fig09_snr_residuals.png      → Рисунок 9.  SNR и гистограмма остатков
""")
