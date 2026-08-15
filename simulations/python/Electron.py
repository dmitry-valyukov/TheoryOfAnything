"""Электрон: самосопряжённый рог-тор.

Что показано и каков статус каждого слоя (по состоянию модели, 2.7-2.8):

  лента (hsv)      — структура ФАЗЫ волнового фронта: обход 4pi, спин 1/2
                     из репера среды (pi_1(SO(3)) = Z_2, подглава 2.4).
                     Тор — не формфактор заряда (4.12).
  золотые стрелки  — репер Картана: нечётное число полуоборотов вдоль
                     орбиты. Именно так теперь читается условие
                     phi = theta/2 (2.7.7, нечётный фрейминг).
  алые стрелки     — ГИРАЦИЯ: физическое смещение среды u = Re[W e^{-iwt}].
                     Каждая точка трубки описывает окружность в
                     полоидальной плоскости, фаза гирации бежит по тору
                     с намоткой n = 1. По расчёту минимайзера сектора
                     (TorusBVP.py, 2.8.5) гирация несёт ~40% момента
                     (спин), намотка ~60% (орбита).

  R = r (рог-тор)  — больше НЕ постулат: цена спиральности kappa(A)
                     минимальна на границе A = R/r = 1 (2.8.3).
  О ЗНАКЕ: «+» у электрона — конвенция сцены, а не вывод. Среда зеркально
  симметрична, оба зеркала вырождены по энергии (TorusBVP: H = +-10.071);
  физична только ПРОТИВОПОЛОЖНОСТЬ пары (C = зеркало). Абсолютный знак
  стал бы выводимым только в слабом секторе (T-38).
"""

import lib.plot as plot
from lib.plot import rgba
from lib.torus import Torus

import numpy as np

# --- Параметры модели электрона ---
num_points = 400  # Количество точек на волновой нити
num_frames = 120  # Количество кадров для полного зацикленного периода
FPS = 25

# Рог-тор R = r: минимум цены спиральности kappa(A) при A = 1 (2.8.3)
tor = Torus(R = 1.0, r = 1.0)

# --- Инициализация 3D сцены ---
ax = plot.init("Электрон в ОТО: Динамика самосопряженного Horn Torus",
               zoom = 4, elev = 30, azim = 45,
               theme = 'Dark')

# Отрисовка сетки рог-тора
tor.plot(ax, 32)

CONST_PI = np.pi
CONST_2PI = CONST_PI * 2

# 2. Инициализация нити волнового фронта
theta_wave = np.linspace(0, 4 * CONST_PI, num_points)

# Спин 1/2, обход 4*pi
theta_wave2 = 0.5 * theta_wave

# Шаг для отрисовки векторов Картана (каждая 15-я точка)
step = 15
theta_quiver = theta_wave[::step]
num_arrows = len(theta_quiver)

# Длина стрелки Картана (векторы (tx,ty,tz) уже единичной нормы)
ARROW_LEN = 0.25

# Создаем пустые графические объекты, которые будут обновляться в анимации
#wave_line = plot.Segments(ax)
num_ribbon_lines = 5  # Количество линий в ленте
wave_ribbon = []
offsets = np.linspace(-0.1, 0.1, num_ribbon_lines) # Сдвиги фазы

for i in range(num_ribbon_lines):
    line = plot.Segments(ax, cmap='hsv', linewidths=4.0, alpha=0.5)
    wave_ribbon.append(line)

# Одна переиспользуемая коллекция вместо 27 отдельных ax.quiver() на каждом кадре
quiver_arrows = plot.Arrows(ax, num_arrows, rgba('gold', 0.9))

# --- Слой гирации (2.8.5): смещение среды в точках кернового кольца ---
NUM_GYRO = 20                      # точки на керновом кольце (rho = R, z = 0)
GYRO_AMP = 0.45                    # видимая амплитуда смещения
theta_gyro = np.linspace(0, CONST_2PI, NUM_GYRO, endpoint=False)
gyro_arrows = plot.Arrows(ax, NUM_GYRO, rgba('#CC3311', 0.95))

ax.text2D(0.02, 0.03,
          "гирация (алые): ~40% момента — спин\n"
          "намотка фазы: ~60% — орбита   (TorusBVP.py, 2.8.5)",
          transform=ax.transAxes, color=plot.FG_COLOR, fontsize=10)

# Приращение фазы на фрейм
dphi = CONST_2PI / num_frames

colors0 = np.linspace(0, 1.5, num_points) % 1.0

# 4. Функция обновления кадра анимации
def update(frame):
    phase = dphi * frame

    # Смещение фазы волнового фронта во времени
    #phi_wave = theta_wave2 - phase

    R = tor.R
    r = tor.r

    phi_base = theta_wave2 - phase

    for idx, offset in enumerate(offsets):
        # Добавляем небольшой сдвиг к каждой линии ленты
        phi_current = phi_base + offset

        rcos_phi = R + r *np.cos(phi_current)
        sin_phi = np.sin(phi_current)

        x_w = rcos_phi * np.cos(theta_wave)
        y_w = rcos_phi * np.sin(theta_wave)
        z_w = r * sin_phi

        points = np.array([x_w, y_w, z_w]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Устанавливаем сегменты
        wave_ribbon[idx].set_segments(segments)

        # Циклический сдвиг в зависимости от фазы
        # Преобразуем фазу в количество позиций для сдвига
        shift_amount = int(((CONST_PI + phase - offset) / CONST_2PI) * num_points) % num_points

        # Сдвигаем массив
        colors_shifted = np.roll(colors0, shift_amount)

        wave_ribbon[idx].set_array(colors_shifted)

    # Новые позиции векторов Картана
    phi_quiver = 0.5 * theta_quiver - phase
    x_q = (R + r * np.cos(phi_quiver)) * np.cos(theta_quiver)
    y_q = (R + r * np.cos(phi_quiver)) * np.sin(theta_quiver)
    z_q = r * np.sin(phi_quiver)

    # Направления векторов кручения (норма тождественно равна 1)
    u_q = ARROW_LEN * np.cos(phi_quiver) * np.cos(theta_quiver)
    v_q = ARROW_LEN * np.cos(phi_quiver) * np.sin(theta_quiver)
    w_q = ARROW_LEN * np.sin(phi_quiver)

    # Обновляем геометрию стрелок in-place — без remove()/пересоздания
    quiver_arrows.recalc(x_q, y_q, z_q, u_q, v_q, w_q)

    # Гирация: u = A [cos(Phi) e_rho + sin(Phi) e_z], Phi = theta - w t.
    # Знаки согласованы с лентой: у ленты phi_c = theta/2 - w t, то есть
    # d(phi_c)/d(theta) > 0 и d(phi_c)/dt < 0 — у гирации те же знаки.
    # Намотка гирации целочисленная (n = 1): векторное поле обязано быть
    # однозначным; полуоборот на обход (theta/2) — свойство РЕПЕРА (ленты),
    # а не физического смещения.
    Phi = theta_gyro - phase
    ox = R * np.cos(theta_gyro)
    oy = R * np.sin(theta_gyro)
    oz = np.zeros_like(theta_gyro)
    gx = GYRO_AMP * np.cos(Phi) * np.cos(theta_gyro)
    gy = GYRO_AMP * np.cos(Phi) * np.sin(theta_gyro)
    gz = GYRO_AMP * np.sin(Phi)
    gyro_arrows.recalc(ox, oy, oz, gx, gy, gz)

    return wave_ribbon + [quiver_arrows, gyro_arrows]

# Запуск зацикленной анимации
plot.animate(update, num_frames, 1000/FPS)
