# main.py — точка входа (ОБНОВЛЁННАЯ ВЕРСИЯ)
import numpy as np
from methods import rectangles, trapezoid, simpson, three_eighths, adaptive_integration
from plots import plot_error_vs_epsilon, plot_n_vs_epsilon

# ===== 1. Задаём функцию и точное значение интеграла =====
# Пример 1: гладкая функция
def f_smooth(x):
    return np.exp(x)  # ∫₀¹ eˣ dx = e - 1 ≈ 1.71828

a, b = 0, 1
exact_smooth = np.e - 1

# Пример 2: функция с изломом (менее гладкая)
def f_rough(x):
    return np.abs(x - 0.5)  # ∫₀¹ |x-0.5| dx = 0.25

exact_rough = 0.25

# ===== 2. Словарь методов =====
methods = {
    'Прямоугольники': rectangles,
    'Трапеции': trapezoid,
    'Симпсон': simpson,
    '3/8': three_eighths
}

# ===== 3. Набор точностей для исследования =====
epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# ===== 4. Исследование на гладкой функции =====
print("🔍 Исследование: гладкая функция f(x)=eˣ")
plot_error_vs_epsilon(methods, f_smooth, a, b, exact_smooth, epsilons, 'graph_smooth_error.png')
plot_n_vs_epsilon(methods, f_smooth, a, b, epsilons, 'graph_smooth_n.png')

# ===== 5. Исследование на негладкой функции =====
print("🔍 Исследование: негладкая функция f(x)=|x-0.5|")
plot_error_vs_epsilon(methods, f_rough, a, b, exact_rough, epsilons, 'graph_rough_error.png')
plot_n_vs_epsilon(methods, f_rough, a, b, epsilons, 'graph_rough_n.png')

# ===== 6. Печать таблицы результатов (для отчёта) =====
print("\n📊 Таблица результатов (гладкая функция):")
print(f"{'Метод':<15} {'ε':<10} {'N':<8} {'Погрешность':<15}")
print("-" * 50)
for name, method in methods.items():
    for eps in [1e-3, 1e-5]:
        result, n = adaptive_integration(method, f_smooth, a, b, eps)
        error = abs(result - exact_smooth)
        print(f"{name:<15} {eps:<10.1e} {n:<8} {error:<15.3e}")

# ===== 7. ГРАФИК СХОДИМОСТИ (НОВЫЙ БЛОК!) =====
print("\n📈 Построение графика сходимости...")

import matplotlib.pyplot as plt

# Набор значений N для исследования сходимости
N_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

plt.figure(figsize=(10, 7))

for name, method in methods.items():
    errors = []
    for n in N_values:
        result = method(f_smooth, a, b, n)
        error = abs(result - exact_smooth)
        errors.append(error)
    
    plt.loglog(N_values, errors, marker='o', label=name, linewidth=2)

# Добавим теоретическую линию сходимости O(N^-2)
plt.loglog(N_values, [1e-2 * (N_values[0]/n)**2 for n in N_values], 
          '--', color='gray', alpha=0.5, label='O(N⁻²)')

# Добавим теоретическую линию сходимости O(N^-4)
plt.loglog(N_values, [1e-2 * (N_values[0]/n)**4 for n in N_values], 
          '--', color='red', alpha=0.5, label='O(N⁻⁴)')

plt.xlabel('Число разбиений N')
plt.ylabel('Фактическая погрешность')
plt.title('Сходимость методов численного интегрирования')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ График сходимости сохранён: convergence_plot.png")

print("\n✅ Все 5 графиков сохранены в папке проекта!")
