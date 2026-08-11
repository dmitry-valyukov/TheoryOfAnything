import numpy as np
import matplotlib.pyplot as plt

# Параметры
omega_p = 1e-13  # плазменная частота (реалистичное значение)
c = 3e8  # скорость света
omega_0 = 1e15  # центральная частота фотона (1 мкм)
L = 9.46e24  # расстояние 1 млрд световых лет в метрах
tau_0 = 1e-15  # начальная длительность (1 фс)
z = 1  # красное смещение для 1 млрд световых лет

# 1. Космологическое красное смещение
tau_cosmo = tau_0 * (1 + z)

# 2. Плазменная дисперсия
k2 = -omega_p**2 / (c * omega_0**3)

# Дисперсионное уширение с учетом космологического растяжения
def dispersion_broadening(L, k2, tau_0, z):
    return tau_0 * (1 + z) * np.sqrt(1 + (k2 * L / (tau_0 * (1 + z))**2))

tau_disp = dispersion_broadening(L, k2, tau_0, z)

# 3. Общая длительность (с учетом обоих эффектов)
tau_total = tau_disp

# Визуализация
plt.figure(figsize=(10, 6))

plt.barh([1, 2, 3], [tau_0, tau_cosmo, tau_total], color=['gray', 'blue', 'green'])
plt.yticks([1, 2, 3], ['Исходная', 'Космология', 'Общая'])
plt.xlabel('Длительность (с)', fontsize=12)
plt.title('Эволюция длительности фотона', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()