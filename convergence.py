# convergence.py — исследование сходимости методов
import numpy as np
import matplotlib.pyplot as plt
from methods import rectangles, trapezoid, simpson, three_eighths

def f(x):
    return np.exp(x)

a, b = 0, 1
exact = np.e - 1

# Набор значений N
N_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

methods = {
    'Прямоугольники': rectangles,
    'Трапеции': trapezoid,
    'Симпсон': simpson,
    '3/8': three_eighths
}

plt.figure(figsize=(10, 7))

for name, method in methods.items():
    errors = []
    for n in N_values:
        result = method(f, a, b, n)
        error = abs(result - exact)
        errors.append(error)
    
    plt.loglog(N_values, errors, marker='o', label=name, linewidth=2)
    
    # Добавим теоретическую линию сходимости
    if 'Симпсон' in name or '3/8' in name:
        # O(N^-4)
        plt.loglog(N_values, [1e-2 * (N_values[0]/n)**4 for n in N_values], 
                  '--', alpha=0.3)
    else:
        # O(N^-2)
        plt.loglog(N_values, [1e-2 * (N_values[0]/n)**2 for n in N_values], 
                  '--', alpha=0.3)

plt.xlabel('Число разбиений N')
plt.ylabel('Фактическая погрешность')
plt.title('Сходимость методов численного интегрирования')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("График сходимости сохранён: convergence_plot.png")
