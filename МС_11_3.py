#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Типовой расчет по математической статистике
Часть 1, Задание 3
Вариант 11: Показательное распределение с параметром λ = 0.967
Студент: 3 курс ПМИ
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Параметры ====================
V = 11  # номер варианта
N = 200  # объем выборки
lambda_param = 0.967  # параметр показательного распределения (из таблицы для варианта 11)

# Установка seed = V + 10 = 21 (согласно указанию)
np.random.seed(V + 10)  # seed = 21

print("=" * 80)
print(f"Задание 3. Показательное распределение с параметром λ = {lambda_param}")
print(f"Объем выборки N = {N}")
print(f"Seed = {V + 10}")
print("=" * 80)

# ==================== 1. Генерация выборки ====================
# Используем numpy.random.Generator.exponential для генерации
rng = np.random.default_rng(V + 10)
# Для показательного распределения scale = 1/λ
exp_data = rng.exponential(scale=1 / lambda_param, size=N)

# ==================== 2. Исходная выборка (таблица 20×10) ====================
print("\n" + "=" * 80)
print("Исходная выборка {x1, x2, ..., xN} (20 строк × 10 столбцов):")
print("=" * 80)

# Разбиваем на строки по 10 элементов
for i in range(0, N, 10):
    row = exp_data[i:i + 10]
    print(" ".join(f"{val:8.5f}" for val in row))

# ==================== 3. Упорядоченная выборка ====================
sorted_data = np.sort(exp_data)

print("\n" + "=" * 80)
print("Упорядоченная выборка по возрастанию {x(1), x(2), ..., x(N)} (20 строк × 10 столбцов):")
print("=" * 80)

for i in range(0, N, 10):
    row = sorted_data[i:i + 10]
    print(" ".join(f"{val:8.5f}" for val in row))

# ==================== 4. Определение интервалов по формуле Стерджеса ====================
m = 1 + int(np.log2(N))  # число интервалов по формуле Стерджеса
a0 = 0  # левая граница (по условию a0 = 0)
am = np.max(exp_data)  # правая граница = максимальное значение выборки
h = (am - a0) / m  # длина интервала

print("\n" + "=" * 80)
print("Параметры интервального ряда:")
print("=" * 80)
print(f"Число интервалов (по формуле Стерджеса): m = 1 + [log2({N})] = {m}")
print(f"Левая граница: a0 = {a0}")
print(f"Правая граница: am = {am:.6f}")
print(f"Длина интервала: h = {h:.6f}")

# ==================== 5. Интервальный ряд и ассоциированный статистический ряд ====================
# Формируем интервалы
intervals = []
midpoints = []
frequencies = []
relative_freqs = []

for i in range(m):
    left = a0 + i * h
    right = a0 + (i + 1) * h
    if i == 0:
        interval_str = f"[{left:.4f}, {right:.4f}]"
    else:
        interval_str = f"({left:.4f}, {right:.4f}]"
    intervals.append(interval_str)

    # Середина интервала
    midpoint = (left + right) / 2
    midpoints.append(midpoint)

    # Частота попадания в интервал
    if i == m - 1:
        # Для последнего интервала включаем правую границу
        freq = np.sum((exp_data >= left) & (exp_data <= right))
    else:
        freq = np.sum((exp_data >= left) & (exp_data < right))
    frequencies.append(freq)

    # Относительная частота
    relative_freqs.append(freq / N)

print("\n" + "=" * 80)
print("Интервальный ряд:")
print("=" * 80)
print(f"{'Интервал':<20} {'n_i':>8} {'w_i':>12}")
print("-" * 45)
for interval, n_i, w_i in zip(intervals, frequencies, relative_freqs):
    print(f"{interval:<20} {n_i:8d} {w_i:12.6f}")
print("-" * 45)
print(f"{'Сумма':<20} {sum(frequencies):8d} {sum(relative_freqs):12.6f}")

print("\n" + "=" * 80)
print("Ассоциированный статистический ряд:")
print("=" * 80)
print(f"{'x_i* (середина)':>20} {'n_i':>8} {'w_i':>12}")
print("-" * 45)
for x_star, n_i, w_i in zip(midpoints, frequencies, relative_freqs):
    print(f"{x_star:20.6f} {n_i:8d} {w_i:12.6f}")
print("-" * 45)

# ==================== 6. Эмпирическая функция распределения ====================
# Строим график эмпирической функции распределения (только горизонтальные полоски, без вертикальных линий)
plt.figure(figsize=(12, 7))

# Получаем все уникальные значения выборки
unique_vals = np.unique(exp_data)
# Вычисляем накопленные частоты для этих значений
ecdf_y = np.array([np.sum(exp_data <= val) / N for val in unique_vals])

# Рисуем горизонтальные полоски для каждого интервала
for i in range(len(unique_vals)):
    # Левая граница интервала
    x_left = unique_vals[i]
    # Правая граница интервала
    if i < len(unique_vals) - 1:
        x_right = unique_vals[i + 1]
    else:
        x_right = unique_vals[i] + 0.1
    # Значение функции на этом интервале
    y_val = ecdf_y[i]

    # Только горизонтальная линия (полочка), без вертикальных
    plt.hlines(y=y_val, xmin=x_left, xmax=x_right, linewidth=2, colors='blue')

# Полоска от -∞ до первого значения (F=0)
plt.hlines(y=0, xmin=unique_vals[0] - 0.1, xmax=unique_vals[0], linewidth=2, colors='blue')

# Полоска после последнего значения (F=1)
plt.hlines(y=1, xmin=unique_vals[-1], xmax=unique_vals[-1] + 0.1, linewidth=2, colors='blue')

plt.xlabel('x', fontsize=12)
plt.ylabel('F_N(x)', fontsize=12)
plt.title(f'Эмпирическая функция распределения\nПоказательное распределение, λ = {lambda_param}, N = {N}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MS_11_3_ecdf.png', dpi=150, bbox_inches='tight')
plt.show()
# ==================== 7. Гистограмма относительных частот с наложенной плотностью ====================
plt.figure(figsize=(12, 7))

# Строим гистограмму относительных частот
# Высота столбцов = w_i / h, чтобы площадь каждого столбца = w_i
heights = [w_i / h for w_i in relative_freqs]

# Создаем границы интервалов для гистограммы
bins = [a0 + i * h for i in range(m + 1)]

plt.bar(bins[:-1], heights, width=h, align='edge', alpha=0.7,
        edgecolor='black', label='Гистограмма относительных частот')

# Строим график плотности показательного распределения (красный)
x_density = np.linspace(0, am, 1000)
density = lambda_param * np.exp(-lambda_param * x_density)
plt.plot(x_density, density, 'r-', linewidth=3, label='Плотность показательного распределения')

plt.xlabel('x', fontsize=12)
plt.ylabel('Плотность / высота', fontsize=12)
plt.title(
    f'Гистограмма относительных частот и плотность распределения\nПоказательное распределение, λ = {lambda_param}, N = {N}',
    fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MS_11_3_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 8. Выборочные характеристики ====================
# Выборочное среднее
x_bar = np.mean(exp_data)

# Выборочная дисперсия с поправкой Шеппарда
# Сначала вычисляем дисперсию по сгруппированным данным
mu2_grouped = np.sum((np.array(midpoints) ** 2) * np.array(relative_freqs))
mu1_grouped = x_bar  # выборочное среднее (можно и по сгруппированным)
D_grouped = mu2_grouped - mu1_grouped ** 2
# Поправка Шеппарда
D_B = D_grouped - (h ** 2) / 12

# Выборочное СКО
sigma = np.sqrt(D_B)

# Выборочная мода (по гистограмме)
# Находим модальный интервал (с максимальной относительной частотой)
max_freq_index = np.argmax(relative_freqs)
max_freq = relative_freqs[max_freq_index]

# Проверяем, один ли модальный интервал
max_indices = np.where(np.array(relative_freqs) == max_freq)[0]

if len(max_indices) == 1:
    # Один модальный интервал
    k = max_indices[0]
    w_k = relative_freqs[k]
    w_kminus1 = relative_freqs[k - 1] if k > 0 else 0
    w_kplus1 = relative_freqs[k + 1] if k < m - 1 else 0

    if 2 * w_k - w_kminus1 - w_kplus1 != 0:
        M0 = a0 + k * h + h * (w_k - w_kminus1) / (2 * w_k - w_kminus1 - w_kplus1)
    else:
        M0 = a0 + k * h + h / 2  # середина интервала как приближение
elif len(max_indices) > 1:
    # Проверяем, идут ли модальные интервалы подряд
    consecutive = np.all(np.diff(max_indices) == 1)
    if consecutive:
        # Несколько модальных интервалов подряд
        k = max_indices[0]
        l = len(max_indices)
        w_k = relative_freqs[k]
        w_kminus1 = relative_freqs[k - 1] if k > 0 else 0
        w_kplusl = relative_freqs[k + l] if k + l < m else 0

        if 2 * w_k - w_kminus1 - w_kplusl != 0:
            M0 = a0 + k * h + l * h * (w_k - w_kminus1) / (2 * w_k - w_kminus1 - w_kplusl)
        else:
            M0 = a0 + k * h + l * h / 2
    else:
        M0 = None  # мода не существует
else:
    M0 = None

# Выборочная медиана
cum_sum = 0
Me = None
for i in range(m):
    cum_sum += relative_freqs[i]
    if cum_sum >= 0.5:
        if cum_sum == 0.5 and i < m - 1:
            # Если накопленная частота точно равна 0.5, медиана = правая граница интервала
            Me = a0 + (i + 1) * h
        else:
            # Линейная интерполяция
            cum_prev = cum_sum - relative_freqs[i]
            Me = a0 + i * h + h * (0.5 - cum_prev) / relative_freqs[i]
        break

# Центральные моменты для асимметрии и эксцесса
mu1 = x_bar
mu2_grouped = D_grouped  # используем сгруппированную дисперсию без поправки для моментов
mu3_grouped = np.sum(((np.array(midpoints) - mu1) ** 3) * np.array(relative_freqs))
mu4_grouped = np.sum(((np.array(midpoints) - mu1) ** 4) * np.array(relative_freqs))

# Коэффициент асимметрии
gamma1 = mu3_grouped / (sigma ** 3)

# Коэффициент эксцесса
gamma2 = mu4_grouped / (sigma ** 4) - 3

print("\n" + "=" * 80)
print("Выборочные характеристики:")
print("=" * 80)
print(f"1) Выборочное среднее x̄                              = {x_bar:.10f}")
print(f"2) Выборочная дисперсия с поправкой Шеппарда D_B     = {D_B:.10f}")
print(f"3) Выборочное СКО σ                                  = {sigma:.10f}")
print(
    f"4) Выборочная мода M0                                = {M0:.6f}" if M0 is not None else "4) Выборочная мода M0                                = не существует")
print(f"5) Выборочная медиана Me                             = {Me:.10f}")
print(f"6) Выборочный коэффициент асимметрии γ1              = {gamma1:.10f}")
print(f"7) Выборочный коэффициент эксцесса γ2                = {gamma2:.10f}")

# ==================== 9. Теоретические характеристики ====================
theor_mean = 1 / lambda_param
theor_var = 1 / (lambda_param ** 2)
theor_sigma = 1 / lambda_param
theor_M0 = 0
theor_Me = np.log(2) / lambda_param
theor_gamma1 = 2
theor_gamma2 = 6

print("\n" + "=" * 80)
print("Теоретические характеристики показательного распределения:")
print("=" * 80)
print(f"Математическое ожидание M[X]          = {theor_mean:.10f}")
print(f"Дисперсия D[X]                         = {theor_var:.10f}")
print(f"Среднее квадратическое отклонение σ[X] = {theor_sigma:.10f}")
print(f"Мода M0                                = {theor_M0}")
print(f"Медиана Me                             = {theor_Me:.10f}")
print(f"Коэффициент асимметрии γ1              = {theor_gamma1:.10f}")
print(f"Коэффициент эксцесса γ2                = {theor_gamma2:.10f}")

# ==================== 10. Таблица сравнения относительных частот и теоретических вероятностей попадания в интервалы ====================
print("\n" + "=" * 80)
print("Таблица сравнения относительных частот и теоретических вероятностей попадания в интервалы")
print(f"Показательное распределение, λ = {lambda_param}")
print("=" * 80)

# Вычисляем теоретические вероятности попадания в каждый интервал
theor_probs = []
for i in range(m):
    left = a0 + i * h
    right = a0 + (i + 1) * h
    # P(left < X ≤ right) = F(right) - F(left)
    prob = (1 - np.exp(-lambda_param * right)) - (1 - np.exp(-lambda_param * left))
    theor_probs.append(prob)

print(f"{'Интервал':<20} {'w_i':>12} {'p_i':>12} {'|w_i - p_i|':>15}")
print("-" * 65)

diffs = []
for interval, w_i, p_i in zip(intervals, relative_freqs, theor_probs):
    diff = abs(w_i - p_i)
    diffs.append(diff)
    print(f"{interval:<20} {w_i:12.6f} {p_i:12.6f} {diff:15.6f}")

print("-" * 65)
print(f"{'Сумма':<20} {sum(relative_freqs):12.6f} {sum(theor_probs):12.6f}")
print(f"\nΔ_max = {max(diffs):.10f}")

# ==================== 11. Таблица сравнения характеристик ====================
print("\n" + "=" * 80)
print("Таблица сравнения рассчитанных характеристик с теоретическими значениями")
print(f"Показательное распределение, λ = {lambda_param}")
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
    f"{M0:.6f}" if M0 is not None else "не существует",
    f"{Me:.10f}",
    f"{gamma1:.10f}",
    f"{gamma2:.10f}"
]

theor_values = [
    f"{theor_mean:.10f}",
    f"{theor_var:.10f}",
    f"{theor_sigma:.10f}",
    f"{theor_M0}",
    f"{theor_Me:.10f}",
    f"{theor_gamma1:.10f}",
    f"{theor_gamma2:.10f}"
]

abs_dev = [
    f"{abs(x_bar - theor_mean):.10f}",
    f"{abs(D_B - theor_var):.10f}",
    f"{abs(sigma - theor_sigma):.10f}",
    "—" if M0 is None else f"{abs(M0 - theor_M0):.10f}",
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

print(
    f"{'Название показателя':<35} {'Выборочное значение':<25} {'Теоретическое значение':<25} {'Абсолютное отклонение':<25} {'Относительное отклонение':<25}")
print("-" * 135)
for name, samp, theor, abs_d, rel_d in zip(characteristics, sample_values, theor_values, abs_dev, rel_dev):
    print(f"{name:<35} {samp:<25} {theor:<25} {abs_d:<25} {rel_d:<25}")

print("\n" + "=" * 80)
print("Все расчеты выполнены. Графики сохранены в файлы:")
print("  - MS_11_3_ecdf.png (эмпирическая функция распределения)")
print("  - MS_11_3_histogram.png (гистограмма относительных частот и плотность распределения)")
print("=" * 80)