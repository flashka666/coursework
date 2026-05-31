"""
Курсовой проект – Глава 3: Первичный анализ набора данных с изображениями
Датасет: Cat and Dog (Kaggle, Tong Python)
https://www.kaggle.com/datasets/tongpython/cat-and-dog

Структура датасета после распаковки:
    Cat and Dog/
        cats/
            cat.4001.jpg  ...  cat.5000.jpg   (1 011 файл)
        dogs/
            dog.4001.jpg  ...  dog.5000.jpg   (1 012 файлов)

Запуск:
    pip install pillow matplotlib seaborn numpy
    python catdog_analysis.py

Все рисунки сохраняются в папку ./figures_ch3/

Соответствие файлов рисунков и подписей в документе (Глава 3):
  fig01 -> Рисунок 3.  Баланс классов (круговая + столбчатая диаграммы)
  fig02 -> Рисунок 4.  Распределение ширины и высоты изображений по классам
  fig03 -> Рисунок 5.  Диаграмма рассеяния ширина × высота
  fig04 -> Рисунок 6.  Размеры файлов (boxplot + гистограмма)
  fig05 -> Рисунок 8.  Распределение средних RGB-значений по каналам и классам
  fig06 -> Рисунок 7.  Соотношение сторон (гистограмма + категории)
  fig07 -> Рисунок 2.  Примеры изображений (6 кошек × 6 собак)
  fig08 -> Рисунок 9.  Средние изображения классов + разностное изображение
  fig09 -> Рисунок 1.  Структура директорий датасета (схема)
  fig10 -> Рисунок 10. Распределение средней яркости по классам

  ВСТАВКА ФОТОГРАФИЙ В ДОКУМЕНТ:
  ─────────────────────────────────────────────────────────────────────
  Рисунок 1  ← fig09_dir_structure.png   (структура каталогов)
  Рисунок 2  ← fig07_examples.png        (примеры фото кошек и собак)
  Рисунок 3  ← fig01_class_balance.png   (баланс классов)
  Рисунок 4  ← fig02_resolution_hist.png (гистограммы разрешений)
  Рисунок 5  ← fig03_wh_scatter.png      (scatter ширина × высота)
  Рисунок 6  ← fig04_file_sizes.png      (размеры файлов)
  Рисунок 7  ← fig06_aspect_ratio.png    (соотношение сторон)
  Рисунок 8  ← fig05_rgb_channels.png    (RGB-каналы)
  Рисунок 9  ← fig08_mean_images.png     (средние изображения)
  Рисунок 10 ← fig10_brightness.png      (яркость)
  ─────────────────────────────────────────────────────────────────────
"""

import os
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Настройки ──────────────────────────────────────────────────────────────
CAT_DIR  = 'Cat and Dog/cats'   # путь к папке с кошками
DOG_DIR  = 'Cat and Dog/dogs'   # путь к папке с собаками
OUT_DIR  = 'figures_ch3'
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

BLUE   = '#2196F3'
ORANGE = '#FB8C00'
RED    = '#E53935'
GREEN  = '#43A047'

# ── Вспомогательные функции ────────────────────────────────────────────────
def get_files(d):
    return sorted([f for f in os.listdir(d) if f.lower().endswith('.jpg')])

def load_stats(d, files):
    """Возвращает списки ширин, высот и размеров файлов (KB)."""
    ws, hs, ss = [], [], []
    for f in files:
        img = Image.open(os.path.join(d, f))
        ws.append(img.width); hs.append(img.height)
        ss.append(os.path.getsize(os.path.join(d, f)) // 1024)
        img.close()
    return ws, hs, ss

cat_files = get_files(CAT_DIR)
dog_files = get_files(DOG_DIR)
print(f"Кошки: {len(cat_files)} файлов  |  Собаки: {len(dog_files)} файлов")

cat_ws, cat_hs, cat_ss = load_stats(CAT_DIR, cat_files)
dog_ws, dog_hs, dog_ss = load_stats(DOG_DIR, dog_files)


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig09. Структура директорий (схема)
# → Рисунок 1 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

nodes = {
    'Cat and Dog/':                                  (0.50, 0.90),
    'cats/':                                         (0.25, 0.65),
    'dogs/':                                         (0.75, 0.65),
    'cat.4001.jpg\ncat.4002.jpg\n...\ncat.5000.jpg': (0.25, 0.28),
    'dog.4001.jpg\ndog.4002.jpg\n...\ndog.5000.jpg': (0.75, 0.28),
}
root_style = dict(boxstyle='round,pad=0.6', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
dir_style  = dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
file_style = dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
style_map  = {
    'Cat and Dog/': root_style,
    'cats/': dir_style, 'dogs/': dir_style,
    'cat.4001.jpg\ncat.4002.jpg\n...\ncat.5000.jpg': file_style,
    'dog.4001.jpg\ndog.4002.jpg\n...\ndog.5000.jpg': file_style,
}
for text, (x, y) in nodes.items():
    ax.text(x, y, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontfamily='monospace', bbox=style_map[text])

for (px, py), (cx, cy) in [
    ((0.50, 0.85), (0.25, 0.73)), ((0.50, 0.85), (0.75, 0.73)),
    ((0.25, 0.57), (0.25, 0.42)), ((0.75, 0.57), (0.75, 0.42)),
]:
    ax.annotate('', xy=(cx, cy), xytext=(px, py),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

ax.text(0.25, 0.10, '1 011 изображений\nmax 500×500 px',
        ha='center', va='center', transform=ax.transAxes,
        fontsize=9, color='#1565C0')
ax.text(0.75, 0.10, '1 012 изображений\nmax 500×500 px',
        ha='center', va='center', transform=ax.transAxes,
        fontsize=9, color='#E65100')
ax.set_title('Структура директорий датасета Cat and Dog',
             fontweight='bold', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig09_dir_structure.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig09 saved  →  Рисунок 1 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig07. Примеры изображений (grid 2×6)
# → Рисунок 2 в документе
# ══════════════════════════════════════════════════════════════════════════════
random.seed(7)
sample_cats = random.sample(cat_files, 6)
sample_dogs = random.sample(dog_files, 6)

fig = plt.figure(figsize=(14, 6))
for i, f in enumerate(sample_cats):
    ax = fig.add_subplot(2, 6, i + 1)
    img = Image.open(os.path.join(CAT_DIR, f)).convert('RGB')
    ax.imshow(img); ax.axis('off')
    ax.set_title(f'{img.width}×{img.height}', fontsize=7)
    if i == 0:
        ax.set_ylabel('Кошки', fontsize=10, fontweight='bold', rotation=90, labelpad=10)
for i, f in enumerate(sample_dogs):
    ax = fig.add_subplot(2, 6, i + 7)
    img = Image.open(os.path.join(DOG_DIR, f)).convert('RGB')
    ax.imshow(img); ax.axis('off')
    ax.set_title(f'{img.width}×{img.height}', fontsize=7)
    if i == 0:
        ax.set_ylabel('Собаки', fontsize=10, fontweight='bold', rotation=90, labelpad=10)
plt.suptitle('Примеры изображений датасета Cat and Dog\n'
             '(по 6 случайных изображений каждого класса)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig07_examples.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig07 saved  →  Рисунок 2 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig01. Баланс классов
# → Рисунок 3 в документе
# ══════════════════════════════════════════════════════════════════════════════
labels  = ['Кошки (cats)', 'Собаки (dogs)']
counts  = [len(cat_files), len(dog_files)]
colors  = [BLUE, ORANGE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
wedges, texts, autotexts = ax1.pie(
    counts, labels=labels, colors=colors, autopct='%1.1f%%',
    startangle=90, textprops={'fontsize': 11})
for at in autotexts:
    at.set_fontsize(11)
ax1.set_title('Соотношение классов', fontweight='bold')

bars = ax2.bar(labels, counts, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{v}', ha='center', fontweight='bold', fontsize=12)
ax2.set_title('Количество изображений по классам', fontweight='bold')
ax2.set_ylabel('Количество изображений')
ax2.set_ylim(0, max(counts) * 1.15)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Баланс классов датасета Cat and Dog', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_class_balance.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig01 saved  →  Рисунок 3 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig02. Распределение разрешений (ширина + высота)
# → Рисунок 4 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, data, label, color in [
    (axes[0, 0], cat_ws, 'Ширина (кошки)',  BLUE),
    (axes[0, 1], dog_ws, 'Ширина (собаки)', ORANGE),
    (axes[1, 0], cat_hs, 'Высота (кошки)',  BLUE),
    (axes[1, 1], dog_hs, 'Высота (собаки)', ORANGE),
]:
    ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(np.mean(data),   color=RED,   lw=2, ls='--',
               label=f'Среднее: {np.mean(data):.0f} px')
    ax.axvline(np.median(data), color=GREEN, lw=2, ls=':',
               label=f'Медиана: {np.median(data):.0f} px')
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Пиксели'); ax.set_ylabel('Количество')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Распределение ширины и высоты изображений по классам',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_resolution_hist.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig02 saved  →  Рисунок 4 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig03. Scatter ширина × высота
# → Рисунок 5 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(cat_ws, cat_hs, alpha=0.3, s=8, color=BLUE,
           label=f'Кошки (n={len(cat_files)})')
ax.scatter(dog_ws, dog_hs, alpha=0.3, s=8, color=ORANGE,
           label=f'Собаки (n={len(dog_files)})')
ax.axline((0, 0), slope=1, color='gray', lw=1, ls='--', label='w = h')
ax.set_xlabel('Ширина (пиксели)'); ax.set_ylabel('Высота (пиксели)')
ax.set_title('Диаграмма рассеяния: ширина × высота изображений', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_wh_scatter.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig03 saved  →  Рисунок 5 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig04. Размеры файлов
# → Рисунок 6 в документе
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bp = ax1.boxplot([cat_ss, dog_ss], labels=['Кошки', 'Собаки'],
                 patch_artist=True,
                 medianprops=dict(color=RED, lw=2),
                 flierprops=dict(marker='o', ms=3, alpha=0.4,
                                 markerfacecolor=RED))
for patch, col in zip(bp['boxes'], [BLUE, ORANGE]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax1.set_title('Размеры файлов (boxplot)', fontweight='bold')
ax1.set_ylabel('Размер файла (КБ)'); ax1.grid(axis='y', alpha=0.3)

ax2.hist(cat_ss, bins=30, alpha=0.6, color=BLUE,
         label=f'Кошки (μ={np.mean(cat_ss):.0f} КБ)', density=True)
ax2.hist(dog_ss, bins=30, alpha=0.6, color=ORANGE,
         label=f'Собаки (μ={np.mean(dog_ss):.0f} КБ)', density=True)
ax2.set_title('Гистограмма размеров файлов', fontweight='bold')
ax2.set_xlabel('Размер файла (КБ)'); ax2.set_ylabel('Плотность')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('Анализ размеров файлов изображений', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig04_file_sizes.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig04 saved  →  Рисунок 6 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig06. Соотношение сторон
# → Рисунок 7 в документе
# ══════════════════════════════════════════════════════════════════════════════
cat_asp = [w / h for w, h in zip(cat_ws, cat_hs)]
dog_asp = [w / h for w, h in zip(dog_ws, dog_hs)]

def asp_categories(asp_list):
    sq   = sum(1 for a in asp_list if 0.9 <= a <= 1.1)
    land = sum(1 for a in asp_list if a > 1.1)
    port = sum(1 for a in asp_list if a < 0.9)
    return [sq, land, port]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(cat_asp, bins=30, alpha=0.7, color=BLUE,   label='Кошки', density=True)
ax1.hist(dog_asp, bins=30, alpha=0.7, color=ORANGE, label='Собаки', density=True)
ax1.axvline(1.0, color='gray', lw=1.5, ls='--', label='Квадрат (1:1)')
ax1.set_title('Гистограмма соотношений сторон (ш/в)', fontweight='bold')
ax1.set_xlabel('Соотношение сторон'); ax1.set_ylabel('Плотность')
ax1.legend(); ax1.grid(alpha=0.3)

cat_c = asp_categories(cat_asp)
dog_c = asp_categories(dog_asp)
x = np.arange(3); w = 0.35
ax2.bar(x - w/2, cat_c, w, label='Кошки', color=BLUE,   alpha=0.85)
ax2.bar(x + w/2, dog_c, w, label='Собаки', color=ORANGE, alpha=0.85)
ax2.set_xticks(list(x))
ax2.set_xticklabels(['Квадрат\n(≈1:1)', 'Альбомная\n(>1.1)', 'Портретная\n(<0.9)'])
ax2.set_title('Категории соотношения сторон', fontweight='bold')
ax2.set_ylabel('Количество изображений')
ax2.legend(); ax2.grid(alpha=0.3)
for rect in ax2.patches:
    ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 3,
             str(int(rect.get_height())), ha='center', fontsize=8)

plt.suptitle('Анализ соотношения сторон изображений', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig06_aspect_ratio.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig06 saved  →  Рисунок 7 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig05. Средние RGB-значения по каналам
# → Рисунок 8 в документе
# ══════════════════════════════════════════════════════════════════════════════
def mean_rgb(d, files, n=200):
    random.seed(42)
    sample = random.sample(files, min(n, len(files)))
    rs, gs, bs = [], [], []
    for f in sample:
        arr = np.array(Image.open(os.path.join(d, f)).convert('RGB'))
        rs.append(arr[:, :, 0].mean())
        gs.append(arr[:, :, 1].mean())
        bs.append(arr[:, :, 2].mean())
    return np.array(rs), np.array(gs), np.array(bs)

cat_r, cat_g, cat_b = mean_rgb(CAT_DIR, cat_files)
dog_r, dog_g, dog_b = mean_rgb(DOG_DIR, dog_files)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, cr, dr, ch, col_c, col_d in [
    (axes[0], cat_r, dog_r, 'Красный канал (R)', '#EF5350', '#FF8A65'),
    (axes[1], cat_g, dog_g, 'Зелёный канал (G)', '#66BB6A', '#AED581'),
    (axes[2], cat_b, dog_b, 'Синий канал (B)',   '#42A5F5', '#90CAF9'),
]:
    ax.hist(cr, bins=25, alpha=0.7, color=col_c, label='Кошки', density=True)
    ax.hist(dr, bins=25, alpha=0.7, color=col_d, label='Собаки', density=True)
    ax.axvline(cr.mean(), color='darkred',  lw=1.5, ls='--')
    ax.axvline(dr.mean(), color='darkblue', lw=1.5, ls=':')
    ax.set_title(ch, fontweight='bold')
    ax.set_xlabel('Среднее значение канала (0–255)')
    ax.set_ylabel('Плотность')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Распределение средних значений RGB-каналов (выборка 200 изобр.)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig05_rgb_channels.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig05 saved  →  Рисунок 8 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig08. Средние изображения + разностное
# → Рисунок 9 в документе
# ══════════════════════════════════════════════════════════════════════════════
def mean_image(d, files, n=300, size=(200, 200)):
    random.seed(42)
    sample = random.sample(files, min(n, len(files)))
    stack  = []
    for f in sample:
        arr = np.array(
            Image.open(os.path.join(d, f)).convert('RGB').resize(size, Image.LANCZOS),
            dtype=np.float32)
        stack.append(arr)
    return np.mean(stack, axis=0).astype(np.uint8)

cat_mean = mean_image(CAT_DIR, cat_files)
dog_mean = mean_image(DOG_DIR, dog_files)
diff     = np.abs(cat_mean.astype(int) - dog_mean.astype(int)).clip(0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
axes[0].imshow(cat_mean); axes[0].axis('off')
axes[0].set_title('Среднее изображение\nкласса «Кошка»', fontweight='bold')
axes[1].imshow(dog_mean); axes[1].axis('off')
axes[1].set_title('Среднее изображение\nкласса «Собака»', fontweight='bold')
im = axes[2].imshow(diff, cmap='hot'); axes[2].axis('off')
axes[2].set_title('Разностное изображение\n(|кошка − собака|)', fontweight='bold')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Различие по пикселям')
plt.suptitle('Средние изображения по классам и их разность (выборка 300 изобр.)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig08_mean_images.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig08 saved  →  Рисунок 9 в документе')


# ══════════════════════════════════════════════════════════════════════════════
# РИСУНОК fig10. Яркость изображений
# → Рисунок 10 в документе
# ══════════════════════════════════════════════════════════════════════════════
def brightness_sample(d, files, n=300):
    random.seed(42)
    sample = random.sample(files, min(n, len(files)))
    return [np.array(Image.open(os.path.join(d, f)).convert('L'),
                     dtype=float).mean()
            for f in sample]

cat_br = brightness_sample(CAT_DIR, cat_files)
dog_br = brightness_sample(DOG_DIR, dog_files)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(cat_br, bins=30, alpha=0.7, color=BLUE,
        label=f'Кошки (μ={np.mean(cat_br):.1f})', density=True)
ax.hist(dog_br, bins=30, alpha=0.7, color=ORANGE,
        label=f'Собаки (μ={np.mean(dog_br):.1f})', density=True)
ax.axvline(np.mean(cat_br), color=BLUE,   lw=2, ls='--')
ax.axvline(np.mean(dog_br), color=ORANGE, lw=2, ls='--')
ax.set_title('Распределение средней яркости изображений (выборка 300)',
             fontweight='bold')
ax.set_xlabel('Средняя яркость (0–255)'); ax.set_ylabel('Плотность')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_brightness.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig10 saved  →  Рисунок 10 в документе')


# ── Дополнительно: генерация CSV с метаданными ─────────────────────────────
import csv
with open(f'{OUT_DIR}/metadata.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label', 'width', 'height',
                     'size_kb', 'aspect_ratio', 'filepath'])
    for fname, w, h, s in zip(cat_files, cat_ws, cat_hs, cat_ss):
        writer.writerow([fname, 'cat', w, h, s, round(w/h, 3),
                         os.path.join(CAT_DIR, fname)])
    for fname, w, h, s in zip(dog_files, dog_ws, dog_hs, dog_ss):
        writer.writerow([fname, 'dog', w, h, s, round(w/h, 3),
                         os.path.join(DOG_DIR, fname)])
print('metadata.csv saved')

print(f'\n✓ Все 10 рисунков и CSV сохранены в папку ./{OUT_DIR}/')
print("""
Таблица соответствия файлов рисунков и подписей в Главе 3:
  figures_ch3/fig09_dir_structure.png  → Рисунок 1.  Структура директорий датасета
  figures_ch3/fig07_examples.png       → Рисунок 2.  Примеры изображений (6+6)
  figures_ch3/fig01_class_balance.png  → Рисунок 3.  Баланс классов
  figures_ch3/fig02_resolution_hist.png→ Рисунок 4.  Гистограммы разрешений
  figures_ch3/fig03_wh_scatter.png     → Рисунок 5.  Scatter ширина × высота
  figures_ch3/fig04_file_sizes.png     → Рисунок 6.  Размеры файлов
  figures_ch3/fig06_aspect_ratio.png   → Рисунок 7.  Соотношение сторон
  figures_ch3/fig05_rgb_channels.png   → Рисунок 8.  RGB-каналы по классам
  figures_ch3/fig08_mean_images.png    → Рисунок 9.  Средние изображения классов
  figures_ch3/fig10_brightness.png     → Рисунок 10. Яркость изображений
""")
