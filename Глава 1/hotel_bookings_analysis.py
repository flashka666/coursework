"""
Курсовой проект – Глава 1: Первичный анализ набора табличных данных
Датасет: Hotel Booking Demand (Kaggle, Jesse Mostipak, 2020)
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Запуск:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly
    python hotel_bookings_analysis.py

Все рисунки сохраняются в папку ./figures/
Соответствие рисунков и подписей в документе:
  fig01 -> Рисунок 1.  Гистограммы распределения числовых признаков
  fig02 -> Рисунок 2.  Зависимость ADR от срока бронирования (Seaborn scatterplot)
  fig03 -> Рисунок 3.  Распределение ADR по месяцам (Seaborn boxplot)
  fig04 -> Рисунок 4.  Попарная диаграмма числовых признаков (Seaborn pairplot)
  fig05 -> Рисунок 5.  Динамика ADR и lead_time по типам отелей (Pyplot)
  fig06 -> Рисунок 6.  Тепловая карта пропущенных значений
  fig07 -> Рисунок 7.  Распределение категориальных признаков
  fig08 -> Рисунок 8.  Анализ выбросов числовых признаков (IQR)
  fig09 -> Рисунок 9.  Три фильтра условной фильтрации
  fig10 -> Рисунок 10. Добавление гауссовского шума к признакам
  fig11 -> Рисунок 11. Преобразование числовых признаков в категориальные
  fig12 -> Рисунок 12. Распределение категориальных признаков
  fig13 -> Рисунок 13. Кодирование категориальных данных и агрегация
  fig14 -> Рисунок 14. Изменения данных до и после обработки
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Настройки ──────────────────────────────────────────────────────────────
DATA_PATH = 'hotel_bookings.csv'   # путь к файлу датасета
OUT_DIR   = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

# ── Загрузка данных ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Пропущенные значения:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
print(f"Дублирующихся строк: {df.duplicated().sum()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 1. Распределения числовых признаков (Matplotlib)
# Подпись: Рисунок 1. Гистограммы распределения числовых признаков датасета
# ══════════════════════════════════════════════════════════════════════════════
num_cols = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'adr', 'total_of_special_requests', 'days_in_waiting_list']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(num_cols)))

for i, col in enumerate(num_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=40, color=colors[i], edgecolor='white', linewidth=0.4)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
    axes[i].grid(axis='y', alpha=0.3)
    mu, med = data.mean(), data.median()
    axes[i].axvline(mu,  color='red',    lw=1.5, linestyle='--', label=f'Среднее: {mu:.1f}')
    axes[i].axvline(med, color='orange', lw=1.5, linestyle=':',  label=f'Медиана: {med:.1f}')
    axes[i].legend(fontsize=7)

plt.suptitle('Распределения числовых признаков датасета Hotel Bookings',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_numeric_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig01 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 2. Seaborn scatterplot: lead_time vs ADR по типу отеля
# Подпись: Рисунок 2. Зависимость ADR от срока бронирования по типу отеля (Seaborn)
# ══════════════════════════════════════════════════════════════════════════════
sample = df[df['adr'].between(0, 600)].sample(5000, random_state=42)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=sample, x='lead_time', y='adr', hue='hotel',
                alpha=0.4, s=20, palette='Set2', ax=ax)
ax.set_title('Зависимость среднесуточной стоимости (ADR) от срока бронирования',
             fontweight='bold')
ax.set_xlabel('Срок бронирования (дней)')
ax.set_ylabel('ADR (средняя стоимость за ночь, €)')
ax.legend(title='Тип отеля')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_seaborn_leadtime_adr.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig02 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 3. Seaborn boxplot: ADR по месяцам
# Подпись: Рисунок 3. Распределение ADR по месяцам прибытия (Seaborn boxplot)
# ══════════════════════════════════════════════════════════════════════════════
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
data_box = df[df['adr'].between(0, 400)]

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=data_box, x='arrival_date_month', y='adr',
            order=month_order, palette='coolwarm', ax=ax,
            showfliers=False, width=0.6)
ax.set_title('Распределение ADR по месяцам прибытия', fontweight='bold')
ax.set_xlabel('Месяц прибытия')
ax.set_ylabel('ADR (€)')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_seaborn_adr_monthly.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig03 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 4. Seaborn pairplot
# Подпись: Рисунок 4. Попарная диаграмма числовых признаков по типу отеля (pairplot)
# ══════════════════════════════════════════════════════════════════════════════
pair_cols = ['lead_time', 'adr', 'stays_in_week_nights',
             'total_of_special_requests', 'is_canceled']
s = df[pair_cols + ['hotel']].dropna().sample(2000, random_state=42)
s = s[s['adr'].between(0, 500)]

fig = sns.pairplot(s, hue='hotel', vars=pair_cols, palette='Set2',
                   plot_kws={'alpha': 0.3, 's': 10}, diag_kind='kde')
fig.fig.suptitle('Попарная диаграмма числовых признаков (по типу отеля)',
                 y=1.01, fontweight='bold')
fig.savefig(f'{OUT_DIR}/fig04_seaborn_pairplot.png', bbox_inches='tight', dpi=130)
plt.close()
print("fig04 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 5. Pyplot: динамика ADR по месяцам + распределение lead_time
# Подпись: Рисунок 5. Динамика ADR по месяцам и распределение lead_time (Pyplot)
# ══════════════════════════════════════════════════════════════════════════════
monthly_adr = (df[df['adr'].between(0, 500)]
               .groupby(['arrival_date_month', 'hotel'])['adr']
               .mean().reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for hotel, grp in monthly_adr.groupby('hotel'):
    grp_sorted = grp.set_index('arrival_date_month').reindex(month_order)
    axes[0].plot(range(12), grp_sorted['adr'], marker='o', label=hotel, linewidth=2)
axes[0].set_xticks(range(12))
axes[0].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[0].set_title('Динамика среднего ADR по месяцам', fontweight='bold')
axes[0].set_xlabel('Месяц прибытия')
axes[0].set_ylabel('Средний ADR (€)')
axes[0].legend(title='Тип отеля')
axes[0].grid(alpha=0.3)

bins = np.arange(0, 400, 20)
for hotel, grp in df.groupby('hotel'):
    axes[1].hist(grp['lead_time'], bins=bins, alpha=0.6, label=hotel, density=True)
axes[1].set_title('Распределение срока бронирования (lead_time)', fontweight='bold')
axes[1].set_xlabel('Срок бронирования (дней)')
axes[1].set_ylabel('Плотность')
axes[1].legend(title='Тип отеля')
axes[1].grid(alpha=0.3)

plt.suptitle('Визуализация признаков средствами Matplotlib/Pyplot',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig05_pyplot_dynamic.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig05 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 6. Тепловая карта пропущенных значений
# Подпись: Рисунок 6. Тепловая карта пропущенных значений (выборка 500 записей)
# ══════════════════════════════════════════════════════════════════════════════
missing_cols = ['children', 'country', 'agent', 'company']
sample_miss = df[missing_cols].sample(500, random_state=42)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(sample_miss.isnull(), cbar=True, yticklabels=False,
            cmap='Blues', ax=ax, cbar_kws={'label': '1 = пропуск, 0 = значение'})
ax.set_title('Тепловая карта пропущенных значений (500 наблюдений)', fontweight='bold')
ax.set_xticklabels(missing_cols, rotation=30)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig06_heatmap_missing.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig06 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 7. Распределение категориальных признаков
# Подпись: Рисунок 7. Распределение категориальных признаков датасета
# ══════════════════════════════════════════════════════════════════════════════
cat_plots = [
    ('hotel',              'Тип отеля'),
    ('meal',               'Тип питания'),
    ('market_segment',     'Сегмент рынка'),
    ('deposit_type',       'Тип депозита'),
    ('customer_type',      'Тип клиента'),
    ('reservation_status', 'Статус бронирования'),
]
palette = sns.color_palette('Set3', 8)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (col, title) in zip(axes.flatten(), cat_plots):
    vc = df[col].value_counts()
    bars = ax.bar(range(len(vc)), vc.values, color=palette[:len(vc)])
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=30, ha='right', fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Количество записей')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{v:,}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Распределение категориальных признаков', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig07_categorical_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig07 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 8. Анализ выбросов (boxplot, метод IQR)
# Подпись: Рисунок 8. Анализ выбросов числовых признаков методом IQR
# ══════════════════════════════════════════════════════════════════════════════
outlier_cols = ['lead_time', 'adr', 'stays_in_week_nights', 'adults', 'days_in_waiting_list']

fig, axes = plt.subplots(1, 5, figsize=(16, 5))
for ax, col in zip(axes, outlier_cols):
    data = df[col].dropna()
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor='#A8D8EA', color='#1F618D'),
               medianprops=dict(color='#E74C3C', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=2, alpha=0.3))
    ax.set_title(f'{col}\n({len(outliers)} выбр.)', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Анализ выбросов числовых признаков (метод IQR)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig08_outliers_boxplot.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig08 saved")


# ══════════════════════════════════════════════════════════════════════════════
# ПРЕДОБРАБОТКА (для дальнейших шагов)
# ══════════════════════════════════════════════════════════════════════════════
df_clean = df.copy()
df_clean['children'] = df_clean['children'].fillna(0)
df_clean['country']  = df_clean['country'].fillna('Unknown')
df_clean['agent']    = df_clean['agent'].fillna(0)
df_clean['company']  = df_clean['company'].fillna(0)
df_clean = df_clean.drop_duplicates()
print(f"\nПосле удаления дубликатов: {len(df_clean)} строк")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 9. Три варианта условной фильтрации
# Подпись: Рисунок 9. Результаты трёх вариантов условной фильтрации данных
# ══════════════════════════════════════════════════════════════════════════════

# Фильтр 1: не отменённые + ADR ∈ [50, 300]
f1 = df_clean[(df_clean['is_canceled'] == 0) & df_clean['adr'].between(50, 300)]

# Фильтр 2: City Hotel, летние месяцы
f2 = df_clean[(df_clean['arrival_date_month'].isin(['June', 'July', 'August'])) &
              (df_clean['hotel'] == 'City Hotel')]

# Фильтр 3: семейные (дети > 0 или взрослых >= 3)
f3 = df_clean[(df_clean['children'] > 0) | (df_clean['adults'] >= 3)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(f1['adr'], bins=40, color='#3498DB', edgecolor='white', alpha=0.8)
axes[0].set_title(f'Ф1: Не отмен. + ADR ∈ [50,300]\n(n={len(f1):,})', fontweight='bold')
axes[0].set_xlabel('ADR (€)')
axes[0].set_ylabel('Частота')
axes[0].grid(alpha=0.3)

top_seg = f2['market_segment'].value_counts().head(5).index
f2_top  = f2[f2['market_segment'].isin(top_seg)]
axes[1].boxplot([f2_top[f2_top['market_segment'] == s]['lead_time'].values for s in top_seg],
                labels=top_seg, patch_artist=True)
axes[1].set_title(f'Ф2: Лето, City Hotel\n(n={len(f2):,})', fontweight='bold')
axes[1].set_xlabel('Сегмент рынка')
axes[1].set_ylabel('lead_time (дней)')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(alpha=0.3)

vc3 = f3['meal'].value_counts()
axes[2].bar(vc3.index.tolist(), vc3.values,
            color=sns.color_palette('Pastel1', len(vc3)))
axes[2].set_title(f'Ф3: Семейные группы\n(n={len(f3):,})', fontweight='bold')
axes[2].set_xlabel('Тип питания')
axes[2].set_ylabel('Количество')
axes[2].grid(alpha=0.3)

plt.suptitle('Результаты условной фильтрации данных', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig09_filtering.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig09 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 10. Добавление гауссовского шума
# Подпись: Рисунок 10. Результаты добавления гауссовского шума к признакам lead_time и adr
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
df_noise = df_clean[['lead_time', 'adr']].copy().dropna()
df_noise['lead_time_noisy'] = df_noise['lead_time'] + np.random.normal(0, 5, len(df_noise))
df_noise['adr_noisy']       = df_noise['adr'].clip(0, 500) + np.random.normal(0, 3, len(df_noise))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
s = df_noise.sample(3000, random_state=0)

for (ax_h, ax_s), orig_col, noise_col, label in [
    ((axes[0, 0], axes[0, 1]), 'lead_time', 'lead_time_noisy', 'lead_time'),
    ((axes[1, 0], axes[1, 1]), 'adr',       'adr_noisy',       'adr'),
]:
    orig  = s[orig_col]
    noisy = s[noise_col]
    if label == 'adr':
        orig  = orig.clip(0, 500)
        noisy = noisy.clip(0, 500)
    ax_h.hist(orig,  bins=40, alpha=0.6, color='#3498DB', label='Исходные', density=True)
    ax_h.hist(noisy, bins=40, alpha=0.6, color='#E74C3C', label='С шумом',   density=True)
    ax_h.set_title(f'Распределение: {label}', fontweight='bold')
    ax_h.set_xlabel('Значение'); ax_h.set_ylabel('Плотность')
    ax_h.legend(); ax_h.grid(alpha=0.3)

    ax_s.scatter(orig, noisy, alpha=0.1, s=5, color='#8E44AD')
    mn = min(orig.min(), noisy.min()); mx = max(orig.max(), noisy.max())
    ax_s.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='y=x')
    ax_s.set_title(f'Исходные vs Зашумлённые: {label}', fontweight='bold')
    ax_s.set_xlabel('Исходное'); ax_s.set_ylabel('С шумом')
    ax_s.legend(); ax_s.grid(alpha=0.3)

plt.suptitle('Результаты добавления гауссовского шума', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_noise.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig10 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 11. Числовые признаки → категориальные + новая категория «Сезон»
# Подпись: Рисунок 11. Преобразование числовых признаков в категориальные
# ══════════════════════════════════════════════════════════════════════════════
df_cat = df_clean.copy()

df_cat['adr_category'] = pd.cut(
    df_cat['adr'].clip(0, 500),
    bins=[0, 50, 100, 150, 200, 500],
    labels=['Очень низкий', 'Низкий', 'Средний', 'Высокий', 'Премиум'])

df_cat['lead_time_category'] = pd.cut(
    df_cat['lead_time'],
    bins=[-1, 7, 30, 90, 365, 800],
    labels=['Очень ранний', 'Ранний', 'Заблаговременный', 'Поздний', 'Очень поздний'])

season_map = {
    'December': 'Зима', 'January': 'Зима', 'February': 'Зима',
    'March': 'Весна',   'April': 'Весна',   'May': 'Весна',
    'June': 'Лето',     'July': 'Лето',     'August': 'Лето',
    'September': 'Осень', 'October': 'Осень', 'November': 'Осень',
}
df_cat['season'] = df_cat['arrival_date_month'].map(season_map)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vc_adr = df_cat['adr_category'].value_counts().sort_index()
axes[0].bar(vc_adr.index.tolist(), vc_adr.values,
            color=sns.color_palette('YlOrRd', len(vc_adr)))
axes[0].set_title('Категории ценового диапазона ADR', fontweight='bold')
axes[0].set_xlabel('Категория'); axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=20); axes[0].grid(alpha=0.3)

vc_lt = df_cat['lead_time_category'].value_counts().sort_index()
axes[1].bar(vc_lt.index.tolist(), vc_lt.values,
            color=sns.color_palette('Blues', len(vc_lt)))
axes[1].set_title('Категории срока бронирования', fontweight='bold')
axes[1].set_xlabel('Категория'); axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=20); axes[1].grid(alpha=0.3)

vc_s = df_cat['season'].value_counts()
axes[2].pie(vc_s.values, labels=vc_s.index,
            colors=['#3498DB', '#E67E22', '#27AE60', '#95A5A6'],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
axes[2].set_title('Распределение по сезонам\n(новая категория)', fontweight='bold')

plt.suptitle('Преобразование числовых данных в категориальные',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig11_num_to_cat.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig11 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 12. Тепловая карта корреляции числовых признаков
# Подпись: Рисунок 12. Тепловая карта корреляции числовых признаков (Пирсон)
# ══════════════════════════════════════════════════════════════════════════════
num_all = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
           'adults', 'children', 'babies', 'is_canceled', 'adr',
           'total_of_special_requests', 'days_in_waiting_list',
           'previous_cancellations', 'booking_changes', 'is_repeated_guest']
corr = df[num_all].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8},
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Коэффициент корреляции Пирсона'})
ax.set_title('Тепловая карта корреляции числовых признаков', fontweight='bold', pad=10)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig12_heatmap_corr.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig12 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 13. Кодирование категориальных данных и агрегация
# Подпись: Рисунок 13. Кодирование категориальных данных и агрегация по сегментам рынка
# ══════════════════════════════════════════════════════════════════════════════
agg = (df_clean
       .groupby('market_segment')
       .agg(avg_adr=('adr', 'mean'), cancel_rate=('is_canceled', 'mean'), count=('is_canceled', 'count'))
       .reset_index()
       .sort_values('count', ascending=False)
       .reset_index(drop=True))

deposit_counts = df_clean['deposit_type'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(agg))
axes[0].bar(x, agg['avg_adr'].values, color='#3498DB', alpha=0.8, label='Средний ADR')
ax2 = axes[0].twinx()
ax2.plot(x, (agg['cancel_rate'] * 100).values, 'ro-', label='% отмен', linewidth=2)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(agg['market_segment'].tolist(), rotation=30, ha='right', fontsize=8)
axes[0].set_ylabel('Средний ADR (€)', color='#3498DB')
ax2.set_ylabel('Доля отмен (%)', color='red')
axes[0].set_title('Агрегация по сегментам рынка', fontweight='bold')
axes[0].legend(loc='upper left'); ax2.legend(loc='upper right')
axes[0].grid(alpha=0.3)

axes[1].bar(deposit_counts.index.tolist(), deposit_counts.values.tolist(),
            color=['#3498DB', '#E74C3C', '#2ECC71'])
axes[1].set_title('Распределение типов депозита\n(Label Encoding)', fontweight='bold')
axes[1].set_xlabel('Тип депозита'); axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=15); axes[1].grid(alpha=0.3)
for i, v in enumerate(deposit_counts.values):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=8)

plt.suptitle('Кодирование категориальных данных и агрегация',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig13_encoding_aggregation.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig13 saved")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК 14. Оценка изменений до/после обработки
# Подпись: Рисунок 14. Изменения в данных до и после предобработки
# ══════════════════════════════════════════════════════════════════════════════
df_after = df_clean[df_clean['adr'].between(0, 500)].copy()
q1_a, q3_a = df_after['adr'].quantile(0.25), df_after['adr'].quantile(0.75)
iqr_a = q3_a - q1_a
df_after = df_after[(df_after['adr'] >= q1_a - 1.5*iqr_a) & (df_after['adr'] <= q3_a + 1.5*iqr_a)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df['adr'].dropna().clip(0, 500).values, bins=50, alpha=0.6,
             color='#E74C3C', label=f'До (n={len(df):,})', density=True)
axes[0].hist(df_after['adr'].values, bins=50, alpha=0.6,
             color='#3498DB', label=f'После (n={len(df_after):,})', density=True)
axes[0].set_title('ADR: до и после обработки', fontweight='bold')
axes[0].set_xlabel('ADR (€)'); axes[0].set_ylabel('Плотность')
axes[0].legend(); axes[0].grid(alpha=0.3)

hotels = ['City Hotel', 'Resort Hotel']
c_before = [df[df['hotel'] == h]['is_canceled'].mean() * 100 for h in hotels]
c_after  = [df_after[df_after['hotel'] == h]['is_canceled'].mean() * 100 for h in hotels]
x_bar = np.arange(len(hotels)); w = 0.35
axes[1].bar(x_bar - w/2, c_before, width=w, label='До',    color='#E74C3C', alpha=0.8)
axes[1].bar(x_bar + w/2, c_after,  width=w, label='После', color='#3498DB', alpha=0.8)
axes[1].set_xticks(list(x_bar)); axes[1].set_xticklabels(hotels)
axes[1].set_title('Доля отмен по типу отеля', fontweight='bold')
axes[1].set_ylabel('%'); axes[1].legend(); axes[1].grid(alpha=0.3)

labels = ['Исходный', 'После очистки\nдубликатов', 'После\nфильтрации']
sizes  = [len(df), len(df_clean), len(df_after)]
bars   = axes[2].bar(labels, sizes, color=['#E74C3C', '#F39C12', '#3498DB'], alpha=0.85)
for bar, sz in zip(bars, sizes):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{sz:,}', ha='center', fontweight='bold', fontsize=9)
axes[2].set_title('Изменение объёма датасета', fontweight='bold')
axes[2].set_ylabel('Количество записей'); axes[2].grid(alpha=0.3)

plt.suptitle('Оценка изменений в данных после обработки',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig14_before_after.png', bbox_inches='tight', dpi=150)
plt.close()
print("fig14 saved")

print("\n✓ Все 14 рисунков сохранены в папку ./figures/")
print("""
Соответствие файлов и подписей:
  figures/fig01_numeric_distributions.png  → Рисунок 1.  Гистограммы распределения числовых признаков
  figures/fig02_seaborn_leadtime_adr.png   → Рисунок 2.  Зависимость ADR от срока бронирования (Seaborn)
  figures/fig03_seaborn_adr_monthly.png    → Рисунок 3.  Распределение ADR по месяцам (Seaborn boxplot)
  figures/fig04_seaborn_pairplot.png       → Рисунок 4.  Попарная диаграмма числовых признаков (pairplot)
  figures/fig05_pyplot_dynamic.png         → Рисунок 5.  Динамика ADR и lead_time по типам отелей (Pyplot)
  figures/fig06_heatmap_missing.png        → Рисунок 6.  Тепловая карта пропущенных значений
  figures/fig07_categorical_distributions  → Рисунок 7.  Распределение категориальных признаков
  figures/fig08_outliers_boxplot.png       → Рисунок 8.  Анализ выбросов числовых признаков (IQR)
  figures/fig09_filtering.png              → Рисунок 9.  Три варианта условной фильтрации
  figures/fig10_noise.png                  → Рисунок 10. Добавление гауссовского шума
  figures/fig11_num_to_cat.png             → Рисунок 11. Преобразование числовых признаков в категориальные
  figures/fig12_heatmap_corr.png           → Рисунок 12. Тепловая карта корреляции числовых признаков
  figures/fig13_encoding_aggregation.png   → Рисунок 13. Кодирование категориальных данных и агрегация
  figures/fig14_before_after.png           → Рисунок 14. Изменения в данных до и после обработки
""")
