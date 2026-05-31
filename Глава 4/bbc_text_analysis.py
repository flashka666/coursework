"""
Курсовой проект – Глава 4: Первичный анализ набора текстовых данных
Датасет: BBC News Archive (Kaggle, H. Gültekin)
https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

Запуск:
    pip install pandas numpy matplotlib seaborn
    python bbc_text_analysis.py

Выходные файлы:
  figures_ch4/           — папка с 10 рисунками
  bbc-news-data-clean.csv — очищенный датасет (2 061 статья, TSV, UTF-8)
  bbc_train.csv          — обучающая выборка (1 648 статей)
  bbc_val.csv            — валидационная выборка (206 статей)
  bbc_test.csv           — тестовая выборка (207 статей)

Соответствие рисунков и подписей в документе (Глава 4):
  fig10_csv_structure.png  → Рисунок 1.  Структура и примеры данных BBC News
  fig01_class_balance.png  → Рисунок 2.  Баланс классов (после дедупликации)
  fig03_dedup.png          → Рисунок 3.  Обнаружение и удаление дубликатов
  fig02_text_length.png    → Рисунок 4.  Распределение длин статей (слова)
  fig05_stats_heatmap.png  → Рисунок 5.  Тепловая карта статистик по категориям
  fig06_sentences.png      → Рисунок 6.  Распределение числа предложений
  fig04_top_words.png      → Рисунок 7.  Топ-15 слов по категориям
  fig08_scatter.png        → Рисунок 8.  Слова × предложения (scatter)
  fig07_titles.png         → Рисунок 9.  Длины заголовков по категориям
  fig09_split.png          → Рисунок 10. Распределение классов по выборкам
"""

import os
import re
import json
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Настройки ──────────────────────────────────────────────────────────────
DATA_PATH = 'bbc-news-data.csv'   # входной TSV-файл
OUT_DIR   = 'figures_ch4'
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

BLUE   = '#2196F3'
ORANGE = '#FB8C00'
GREEN  = '#43A047'
RED    = '#E53935'
PURPLE = '#8E24AA'
CAT_COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE]

CATS       = ['business', 'entertainment', 'politics', 'sport', 'tech']
CAT_LABELS = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

# ── Стоп-слова (базовый набор) ─────────────────────────────────────────────
STOP = set([
    'the','a','an','in','on','of','to','and','is','are','was','were','it',
    'its','be','that','this','for','with','as','at','by','from','has','have',
    'had','he','she','they','we','his','her','their','but','or','not','which',
    'will','would','said','also','been','can','more','about','after','than',
    'other','when','up','out','one','all','there','who','mr','ms','if','so',
    'no','do','may','new','year','says','say','us','uk','told','two','first',
    'last','could','should','just','get','make','made','take','taken','some',
    'such','over','been','into','been','their','only','time','people','three',
])


# ══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ОЧИСТКА
# ══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(DATA_PATH, sep='\t', on_bad_lines='skip')
print(f"Raw shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print(f"Categories:\n{df_raw['category'].value_counts().to_string()}")

# Удаление дубликатов
df = df_raw.drop_duplicates(subset='content').drop_duplicates(subset='title').copy()
df = df.reset_index(drop=True)
print(f"\nAfter deduplication: {len(df)} rows (removed {len(df_raw) - len(df)})")

# Производные признаки
df['word_count']  = df['content'].str.split().str.len()
df['char_count']  = df['content'].str.len()
df['title_words'] = df['title'].str.split().str.len()
df['sent_count']  = df['content'].str.count(r'[.!?]+')

cat_counts = [df[df['category'] == c].shape[0] for c in CATS]


# ══════════════════════════════════════════════════════════════════════════════
# 2. РАЗБИЕНИЕ НА TRAIN/VAL/TEST (80/10/10)
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
df_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)
n       = len(df_shuffle)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
df_train = df_shuffle[:n_train]
df_val   = df_shuffle[n_train:n_train + n_val]
df_test  = df_shuffle[n_train + n_val:]
print(f"\nSplit: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig10. Структура CSV-файла (схема)
# → Рисунок 1 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
ax.axis('off')
cols      = ['category', 'filename', 'title', 'content']
xs        = [0.01, 0.17, 0.33, 0.60]
widths    = [0.14, 0.14, 0.27, 0.38]
header_st = dict(boxstyle='round,pad=0.4', facecolor='#1565C0', edgecolor='#1565C0')
cell_sts  = [
    dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#90CAF9'),
    dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#CE93D8'),
    dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#A5D6A7'),
    dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#FFF176'),
]
example_rows = [
    ['business', '001.txt', 'Ad sales boost\nTime Warner profit',
     'Quarterly profits at US media\ngiant TimeWarner jumped 76%...'],
    ['politics', '045.txt', 'Labour plans\nmaternity pay rise',
     'The government plans to extend\nmaternity pay from six months...'],
    ['tech',     '012.txt', 'China net cafe\nculture crackdown',
     'China is clamping down on its\nburgeoning internet cafe culture...'],
]
for j, (col, x, w) in enumerate(zip(cols, xs, widths)):
    ax.text(x + w/2, 0.87, col, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', bbox=header_st)
for i, row in enumerate(example_rows):
    y = 0.60 - i * 0.22
    for j, (val, x, w) in enumerate(zip(row, xs, widths)):
        ax.text(x + w/2, y, val, transform=ax.transAxes, ha='center', va='center',
                fontsize=8.5, fontfamily='monospace', bbox=cell_sts[j], wrap=True)
ax.set_title('Структура и примеры данных BBC News Archive (bbc-news-data.csv)',
             fontweight='bold', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_csv_structure.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig10 saved  →  Рисунок 1 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig01. Баланс классов
# → Рисунок 2 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
wedges, texts, autotexts = ax1.pie(
    cat_counts, labels=CAT_LABELS, colors=CAT_COLORS,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
for at in autotexts:
    at.set_fontsize(10)
ax1.set_title('Доля классов', fontweight='bold')

bars = ax2.bar(CAT_LABELS, cat_counts, color=CAT_COLORS, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, cat_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             f'{v}', ha='center', fontweight='bold', fontsize=10)
ax2.set_title('Количество статей по категориям', fontweight='bold')
ax2.set_ylabel('Количество статей')
ax2.set_ylim(0, max(cat_counts) * 1.15)
ax2.grid(axis='y', alpha=0.3)
plt.suptitle('Баланс классов BBC News Archive (после дедупликации)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_class_balance.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig01 saved  →  Рисунок 2 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig03. Дубликаты до/после
# → Рисунок 3 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_dup = [f'Исходный\n({len(df_raw)} ст.)', f'После дедупл.\n({len(df)} ст.)']
bars = axes[0].bar(labels_dup, [len(df_raw), len(df)],
                   color=[RED, GREEN], alpha=0.85, width=0.4)
for bar, v in zip(bars, [len(df_raw), len(df)]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{v}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('Размер датасета до/после удаления дубликатов', fontweight='bold')
axes[0].set_ylabel('Количество статей')
axes[0].set_ylim(0, 2500)
axes[0].grid(axis='y', alpha=0.3)

raw_counts   = [df_raw[df_raw['category'] == c].shape[0] for c in CATS]
clean_counts = cat_counts
x = np.arange(len(CATS)); w = 0.35
axes[1].bar(x - w/2, raw_counts,   w, label='До',    color=RED,   alpha=0.7)
axes[1].bar(x + w/2, clean_counts, w, label='После', color=GREEN, alpha=0.7)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(CAT_LABELS)
axes[1].set_title('Статьи по категориям до/после дедупликации', fontweight='bold')
axes[1].set_ylabel('Количество')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
plt.suptitle('Обнаружение и удаление дубликатов', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_dedup.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig03 saved  →  Рисунок 3 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig02. Распределение длин статей
# → Рисунок 4 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for cat, col in zip(CATS, CAT_COLORS):
    data = df[df['category'] == cat]['word_count']
    axes[0].hist(data, bins=30, alpha=0.5, color=col,
                 label=f'{cat.capitalize()} (μ={data.mean():.0f})', density=True)
axes[0].set_title('Гистограмма количества слов', fontweight='bold')
axes[0].set_xlabel('Количество слов'); axes[0].set_ylabel('Плотность')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

bp_data = [df[df['category'] == c]['word_count'].values for c in CATS]
bp = axes[1].boxplot(bp_data, labels=CAT_LABELS, patch_artist=True,
                     medianprops=dict(color=RED, lw=2),
                     flierprops=dict(marker='o', ms=2, alpha=0.3, markerfacecolor=RED))
for patch, col in zip(bp['boxes'], CAT_COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.6)
axes[1].set_title('Boxplot количества слов по категориям', fontweight='bold')
axes[1].set_ylabel('Количество слов'); axes[1].grid(axis='y', alpha=0.3)
plt.suptitle('Распределение длин статей (в словах)', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_text_length.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig02 saved  →  Рисунок 4 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig05. Тепловая карта статистик
# → Рисунок 5 в документе
# ══════════════════════════════════════════════════════════════════════════════
stat_df = df.groupby('category').agg(
    mean_words  = ('word_count', 'mean'),
    median_words= ('word_count', 'median'),
    std_words   = ('word_count', 'std'),
    mean_sents  = ('sent_count', 'mean'),
    mean_title  = ('title_words','mean'),
).round(1).reindex(CATS)
stat_df.index = CAT_LABELS

norm = (stat_df - stat_df.min()) / (stat_df.max() - stat_df.min())
col_labels = ['Ср. слов', 'Мед. слов', 'Ст. откл. слов', 'Ср. предл.', 'Ср. слов в заголовке']

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(norm, annot=stat_df.values, fmt='.0f', cmap='YlOrRd', ax=ax,
            xticklabels=col_labels, linewidths=0.5,
            cbar_kws={'label': 'Норм. значение'})
ax.set_title('Статистические характеристики текстов по категориям', fontweight='bold')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig05_stats_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig05 saved  →  Рисунок 5 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig06. Распределение числа предложений
# → Рисунок 6 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
for cat, col in zip(CATS, CAT_COLORS):
    data = df[df['category'] == cat]['sent_count']
    ax.hist(data, bins=25, alpha=0.5, color=col,
            label=f'{cat.capitalize()} (μ={data.mean():.1f})', density=True)
ax.set_title('Распределение количества предложений в статье', fontweight='bold')
ax.set_xlabel('Количество предложений'); ax.set_ylabel('Плотность')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig06_sentences.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig06 saved  →  Рисунок 6 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig04. Топ-15 слов по категориям
# → Рисунок 7 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 7))
for ax, cat, col in zip(axes, CATS, CAT_COLORS):
    texts = ' '.join(df[df['category'] == cat]['content']).lower()
    words = [w for w in re.findall(r'\b[a-z]{3,}\b', texts) if w not in STOP]
    top   = Counter(words).most_common(15)
    words_t, counts_t = zip(*top)
    ax.barh(list(words_t)[::-1], list(counts_t)[::-1], color=col, alpha=0.85)
    ax.set_title(cat.capitalize(), fontweight='bold', fontsize=10)
    ax.tick_params(axis='y', labelsize=8); ax.grid(axis='x', alpha=0.3)
plt.suptitle('Топ-15 наиболее частых слов по категориям (стоп-слова удалены)',
             fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig04_top_words.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig04 saved  →  Рисунок 7 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig08. Scatter: слова × предложения
# → Рисунок 8 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
for cat, col in zip(CATS, CAT_COLORS):
    sub = df[df['category'] == cat]
    ax.scatter(sub['sent_count'], sub['word_count'],
               alpha=0.3, s=10, color=col, label=cat.capitalize())
ax.set_xlabel('Количество предложений'); ax.set_ylabel('Количество слов')
ax.set_title('Зависимость числа слов от числа предложений', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig08_scatter.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig08 saved  →  Рисунок 8 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig07. Длины заголовков
# → Рисунок 9 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
bp2 = ax1.boxplot([df[df['category'] == c]['title_words'].values for c in CATS],
                  labels=CAT_LABELS, patch_artist=True,
                  medianprops=dict(color=RED, lw=2),
                  flierprops=dict(marker='o', ms=2, alpha=0.3, markerfacecolor=RED))
for patch, col in zip(bp2['boxes'], CAT_COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax1.set_title('Длина заголовков (слова) по категориям', fontweight='bold')
ax1.set_ylabel('Слов в заголовке'); ax1.grid(axis='y', alpha=0.3)

ax2.hist(df['title_words'], bins=20, color=BLUE, alpha=0.8, edgecolor='white')
ax2.axvline(df['title_words'].mean(),   color=RED,   lw=2, ls='--',
            label=f"Среднее: {df['title_words'].mean():.1f}")
ax2.axvline(df['title_words'].median(), color=ORANGE, lw=2, ls=':',
            label=f"Медиана: {df['title_words'].median():.1f}")
ax2.set_title('Гистограмма длин заголовков (все статьи)', fontweight='bold')
ax2.set_xlabel('Слов в заголовке'); ax2.set_ylabel('Количество')
ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle('Анализ длин заголовков статей', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig07_titles.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig07 saved  →  Рисунок 9 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig09. Распределение классов по выборкам
# → Рисунок 10 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sub, title in [
    (axes[0], df_train, f'Обучающая (train)\nn={len(df_train)}'),
    (axes[1], df_val,   f'Валидационная (val)\nn={len(df_val)}'),
    (axes[2], df_test,  f'Тестовая (test)\nn={len(df_test)}'),
]:
    counts = [sub[sub['category'] == c].shape[0] for c in CATS]
    bars = ax.bar(CAT_LABELS, counts, color=CAT_COLORS, alpha=0.85)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha='center', fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('Количество статей')
    ax.tick_params(axis='x', rotation=20)
plt.suptitle('Распределение классов по выборкам (split 80/10/10)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig09_split.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig09 saved  →  Рисунок 10 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# 3. СОХРАНЕНИЕ ОЧИЩЕННЫХ ДАТАСЕТОВ
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv('bbc-news-data-clean.csv',      index=False, sep='\t', encoding='utf-8')
df_train.to_csv('bbc_train.csv',          index=False, sep='\t', encoding='utf-8')
df_val.to_csv('bbc_val.csv',              index=False, sep='\t', encoding='utf-8')
df_test.to_csv('bbc_test.csv',            index=False, sep='\t', encoding='utf-8')
print(f'\n✓ Очищенный датасет: bbc-news-data-clean.csv ({len(df)} строк)')
print(f'✓ Train: bbc_train.csv ({len(df_train)}), Val: bbc_val.csv ({len(df_val)}), Test: bbc_test.csv ({len(df_test)})')


# ══════════════════════════════════════════════════════════════════════════════
# 4. ПРИМЕРЫ ДАННЫХ (5 статей с разметкой)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 5 примеров с разметкой (1 на категорию) ===')
for cat in CATS:
    row = df[df['category'] == cat].iloc[0]
    print(f"\n[{cat.upper()}]")
    print(f"  Заголовок : {row['title']}")
    print(f"  Начало    : {row['content'][:120].strip()}...")
    print(f"  Метка     : {row['category']}")
    print(f"  Слов      : {row['word_count']}")

print(f"""
Таблица соответствия файлов рисунков и подписей в Главе 4:
  figures_ch4/fig10_csv_structure.png → Рисунок 1.  Структура CSV-файла
  figures_ch4/fig01_class_balance.png → Рисунок 2.  Баланс классов
  figures_ch4/fig03_dedup.png         → Рисунок 3.  Удаление дубликатов
  figures_ch4/fig02_text_length.png   → Рисунок 4.  Длины статей (слова)
  figures_ch4/fig05_stats_heatmap.png → Рисунок 5.  Тепловая карта статистик
  figures_ch4/fig06_sentences.png     → Рисунок 6.  Количество предложений
  figures_ch4/fig04_top_words.png     → Рисунок 7.  Топ-15 слов по категориям
  figures_ch4/fig08_scatter.png       → Рисунок 8.  Слова × предложения
  figures_ch4/fig07_titles.png        → Рисунок 9.  Длины заголовков
  figures_ch4/fig09_split.png         → Рисунок 10. Разбиение train/val/test
""")
