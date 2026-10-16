#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Типовой расчет по математической статистике
Часть 1, Задание 1
Вариант 11: Биномиальное распределение B(n=19, p=0.478)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb

# ==================== Параметры ====================
V = 11  # номер варианта
N = 200  # объем выборки
n = 19   # параметр биномиального распределения
p = 0.478  # параметр биномиального распределения
q = 1 - p  # вероятность неудачи

# Установка seed = V + 10 = 21 (согласно указанию)
np.random.seed(V + 10)  # seed = 21

print("=" * 80)
print(f"Задание 1. Биномиальное распределение B({n}, {p})")
print(f"Объем выборки N = {N}")
print(f"Seed = {V + 10}")
print("=" * 80)

# ==================== 1. Генерация выборки ====================
# Используем numpy.random.Generator.binomial для генерации
rng = np.random.default_rng(V + 10)
binom_data = rng.binomial(n, p, N)

# ==================== 2. Исходная выборка (таблица 20×10) ====================
print("\n" + "=" * 80)
print("Исходная выборка {x1, x2, ..., xN} (20 строк × 10 столбцов):")
print("=" * 80)

# Разбиваем на строки по 10 элементов
for i in range(0, N, 10):
    row = binom_data[i:i+10]
    print(" ".join(f"{val:4d}" for val in row))

# ==================== 3. Упорядоченная выборка ====================
sorted_data = np.sort(binom_data)

print("\n" + "=" * 80)
print("Упорядоченная выборка по возрастанию {x(1), x(2), ..., x(N)} (20 строк × 10 столбцов):")
print("=" * 80)

for i in range(0, N, 10):
    row = sorted_data[i:i+10]
    print(" ".join(f"{val:4d}" for val in row))

# ==================== 4. Статистический ряд ====================
# Находим уникальные значения и их частоты
values, counts = np.unique(binom_data, return_counts=True)
freq = counts / N  # относительные частоты w_i
cum_freq = np.cumsum(freq)  # накопленные относительные частоты s_i

print("\n" + "=" * 80)
print("Статистический ряд:")
print("=" * 80)
print(f"{'x_i*':>8} {'n_i':>8} {'w_i':>12} {'s_i':>12}")
print("-" * 45)
for v, n_i, w_i, s_i in zip(values, counts, freq, cum_freq):
    print(f"{v:8d} {n_i:8d} {w_i:12.6f} {s_i:12.6f}")
print("-" * 45)
print(f"{'Сумма':>8} {sum(counts):8d} {sum(freq):12.6f}")
print("=" * 80)

# ==================== 5. Эмпирическая функция распределения ====================
# Строим график эмпирической функции распределения (только горизонтальные полоски, без вертикальных линий)
plt.figure(figsize=(12, 7))

# Рисуем горизонтальные полоски для каждого интервала
for i in range(len(values)):
    # Левая граница интервала
    x_left = values[i]
    # Правая граница интервала
    if i < len(values) - 1:
        x_right = values[i + 1]
    else:
        x_right = values[i] + 1
    # Значение функции на этом интервале
    y_val = cum_freq[i]

    # Только горизонтальная линия (полочка), без вертикальных
    plt.hlines(y=y_val, xmin=x_left, xmax=x_right, linewidth=2, colors='blue')

# Полоска от -∞ до первого значения (F=0)
plt.hlines(y=0, xmin=values[0] - 1, xmax=values[0], linewidth=2, colors='blue')

# Полоска после последнего значения (F=1)
plt.hlines(y=1, xmin=values[-1], xmax=values[-1] + 1, linewidth=2, colors='blue')

plt.xlabel('x', fontsize=12)
plt.ylabel('F_N(x)', fontsize=12)
plt.title(f'Эмпирическая функция распределения\nБиномиальное распределение B({n}, {p}), N={N}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MS_11_1_ecdf.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 6. Полигон относительных частот ====================
# Точки для полигона относительных частот
x_poly = values
w_poly = freq

# Теоретические вероятности для биномиального распределения
# Вычисляем для всех k от 0 до n
k_all = np.arange(0, n + 1)
p_theor_all = comb(n, k_all) * (p ** k_all) * (q ** (n - k_all))

# Обрезаем до максимального значения в выборке (M = max(values))
M = max(values)
p_theor_for_plot = p_theor_all[:M + 1]
k_for_plot = k_all[:M + 1]

plt.figure(figsize=(12, 7))
# Полигон относительных частот (синий)
plt.plot(x_poly, w_poly, 'bo-', linewidth=2, markersize=8, label='Полигон относительных частот')
# Полигон теоретических вероятностей (красный)
plt.plot(k_for_plot, p_theor_for_plot, 'ro-', linewidth=2, markersize=5, label='Полигон теоретических вероятностей')
plt.xlabel('x', fontsize=12)
plt.ylabel('Вероятность / относительная частота', fontsize=12)
plt.title(f'Полигон относительных частот и теоретических вероятностей\nБиномиальное распределение B({n}, {p}), N={N}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MS_11_1_polygon.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. Выборочные характеристики ====================
# Выборочное среднее
x_bar = np.mean(binom_data)

# Выборочная дисперсия
D_B = np.var(binom_data, ddof=0)  # ddof=0 для деления на N

# Выборочное СКО
sigma = np.sqrt(D_B)

# Выборочная мода
max_count = np.max(counts)
modes = values[counts == max_count]
if len(modes) == 1:
    M0 = modes[0]
else:
    # Если несколько значений с одинаковой максимальной частотой
    # Проверяем, идут ли они подряд
    modes_sorted = np.sort(modes)
    if np.all(np.diff(modes_sorted) == 1):
        M0 = (modes_sorted[0] + modes_sorted[-1]) / 2
    else:
        M0 = None  # мода не существует

# Выборочная медиана
# Находим значение, для которого накопленная частота впервые >= 0.5
cum_sum = 0
Me = None
for i, (v, cnt) in enumerate(zip(values, counts)):
    cum_sum += cnt / N
    if cum_sum >= 0.5:
        Me = v
        break

# Центральные моменты для асимметрии и эксцесса
mu1 = x_bar
mu2 = np.mean((binom_data - mu1) ** 2)
mu3 = np.mean((binom_data - mu1) ** 3)
mu4 = np.mean((binom_data - mu1) ** 4)

# Коэффициент асимметрии
gamma1 = mu3 / (sigma ** 3)

# Коэффициент эксцесса
gamma2 = mu4 / (sigma ** 4) - 3

print("\n" + "=" * 80)
print("Выборочные характеристики:")
print("=" * 80)
print(f"1) Выборочное среднее x̄                    = {x_bar:.10f}")
print(f"2) Выборочная дисперсия D_B                = {D_B:.10f}")
print(f"3) Выборочное СКО σ                        = {sigma:.10f}")
print(f"4) Выборочная мода M0                      = {M0 if M0 is not None else 'не существует'}")
print(f"5) Выборочная медиана Me                   = {Me:.10f}")
print(f"6) Выборочный коэффициент асимметрии γ1    = {gamma1:.10f}")
print(f"7) Выборочный коэффициент эксцесса γ2      = {gamma2:.10f}")

# ==================== 8. Теоретические характеристики ====================
theor_mean = n * p
theor_var = n * p * q
theor_sigma = np.sqrt(theor_var)

# Теоретическая мода: floor((n+1)p) или два значения
mode_candidate = (n + 1) * p
if mode_candidate.is_integer():
    # Два модальных значения
    theor_M0 = f"{int(mode_candidate - 1)} и {int(mode_candidate)}"
else:
    theor_M0 = int(np.floor(mode_candidate))

# Теоретическая медиана (приближённо)
theor_Me = int(np.round(n * p))

# Теоретическая асимметрия
theor_gamma1 = (q - p) / theor_sigma

# Теоретический эксцесс
theor_gamma2 = (1 - 6 * p * q) / theor_var

print("\n" + "=" * 80)
print("Теоретические характеристики биномиального распределения:")
print("=" * 80)
print(f"Математическое ожидание M[X]          = {theor_mean:.10f}")
print(f"Дисперсия D[X]                         = {theor_var:.10f}")
print(f"Среднее квадратическое отклонение σ[X] = {theor_sigma:.10f}")
print(f"Мода M0                                = {theor_M0}")
print(f"Медиана Me                             = {theor_Me}")
print(f"Коэффициент асимметрии γ1              = {theor_gamma1:.10f}")
print(f"Коэффициент эксцесса γ2                = {theor_gamma2:.10f}")

# ==================== 9. Таблица сравнения относительных частот и теоретических вероятностей ====================
print("\n" + "=" * 80)
print("Таблица сравнения относительных частот и теоретических вероятностей")
print("Биномиальное распределение B(19, 0.478)")
print("=" * 80)

# Вычисляем теоретические вероятности для каждого встретившегося значения
p_theor_for_values = comb(n, values) * (p ** values) * (q ** (n - values))

print(f"{'x_i*':>8} {'w_i':>12} {'p̃_i':>12} {'|w_i - p̃_i|':>15}")
print("-" * 55)

diffs = []
for v, w, p_th in zip(values, freq, p_theor_for_values):
    diff = abs(w - p_th)
    diffs.append(diff)
    print(f"{v:8d} {w:12.6f} {p_th:12.6f} {diff:15.6f}")

print("-" * 55)
print(f"{'Сумма':>8} {sum(freq):12.6f} {sum(p_theor_for_values):12.6f}")
print(f"\nΔ_max = {max(diffs):.10f}")

# ==================== 10. Таблица сравнения характеристик ====================
print("\n" + "=" * 80)
print("Таблица сравнения рассчитанных характеристик с теоретическими значениями")
print("Биномиальное распределение B(19, 0.478)")
print("=" * 80)

# Подготовка данных для таблицы
characteristics = [
    "Среднее значение",
    "Дисперсия",
    "Среднее квадратичное отклонение",
    "Мода",
    "Медиана",
    "Коэффициент асимметрии",
    "Коэффициент эксцесса"
]

sample_values = [
    f"{x_bar:.10f}",
    f"{D_B:.10f}",
    f"{sigma:.10f}",
    f"{M0}" if isinstance(M0, (int, float)) else str(M0),
    f"{Me:.10f}",
    f"{gamma1:.10f}",
    f"{gamma2:.10f}"
]

theor_values = [
    f"{theor_mean:.10f}",
    f"{theor_var:.10f}",
    f"{theor_sigma:.10f}",
    str(theor_M0),
    str(theor_Me),
    f"{theor_gamma1:.10f}",
    f"{theor_gamma2:.10f}"
]

abs_dev = [
    f"{abs(x_bar - theor_mean):.10f}",
    f"{abs(D_B - theor_var):.10f}",
    f"{abs(sigma - theor_sigma):.10f}",
    "—" if M0 is None else str(abs(float(M0) - (int(theor_M0) if isinstance(theor_M0, int) else float(theor_M0.split()[0])))),
    f"{abs(Me - theor_Me):.10f}",
    f"{abs(gamma1 - theor_gamma1):.10f}",
    f"{abs(gamma2 - theor_gamma2):.10f}"
]

rel_dev = []
for i, (th, samp) in enumerate(zip(theor_values, sample_values)):
    try:
        th_val = float(th)
        samp_val = float(samp)
        if th_val == 0:
            rel_dev.append("—")
        else:
            rel_dev.append(f"{abs(samp_val - th_val) / abs(th_val):.10f}")
    except (ValueError, TypeError):
        rel_dev.append("—")

print(f"{'Название показателя':<35} {'Выборочное значение':<25} {'Теоретическое значение':<25} {'Абсолютное отклонение':<25} {'Относительное отклонение':<25}")
print("-" * 135)
for name, samp, theor, abs_d, rel_d in zip(characteristics, sample_values, theor_values, abs_dev, rel_dev):
    print(f"{name:<35} {samp:<25} {theor:<25} {abs_d:<25} {rel_d:<25}")

print("\n" + "=" * 80)
print("Все расчеты выполнены. Графики сохранены в файлы:")
print("  - MS_11_1_ecdf.png (эмпирическая функция распределения)")
print("  - MS_11_1_polygon.png (полигон относительных частот и теоретических вероятностей)")
print("=" * 80)