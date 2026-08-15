"""Электрон и позитрон: две хиральности одного рог-тора.

Статус слоёв по состоянию модели (2.7-2.8):

  канат (cool)     — фаза волнового фронта, обход 4pi (спин 1/2 из
                     pi_1(SO(3)) = Z_2, подглава 2.4); эффект Гуи в фокусе.
  жёлтые стрелки   — репер Картана: нечётное число полуоборотов вдоль
                     орбиты; условие phi = theta/2 читается как нечётный
                     фрейминг, а не выбор конкретной пары (p, q) (2.7.7).
  алые стрелки     — гирация: физическое смещение среды, окружность в
                     полоидальной плоскости; ~40% момента — спин, ~60% —
                     орбитальная намотка (TorusBVP.py, 2.8.5). У позитрона
                     пространственная намотка гирации инвертирована вместе
                     с хиральностью.

  R = r            — рог-тор больше не постулат: kappa(A) минимальна
                     на границе A = R/r = 1 (2.8.3).
  О ЗНАКЕ: «+» у электрона — конвенция сцены, а не вывод. Среда зеркально
  симметрична, оба зеркала вырождены по энергии (TorusBVP: H = +-10.071);
  физична только ПРОТИВОПОЛОЖНОСТЬ пары (C = зеркало). Абсолютный знак
  стал бы выводимым только в слабом секторе (T-38).
"""

import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import adaptive_indices, alpha_profile, to_segments
from lib.torus import Torus

# --- Параметры самосопряжённого тора ---
r = 1.0       # Радиус "шланга"
R = 1 * r     # Рог-тор R = r: минимум цены спиральности kappa(A) (2.8.3)
z_R = 1.0     # Рэлеевская длина для эффекта Гуи
num_points = 400
num_frames = 120  # Длина одного полного зацикленного периода

# Траектория замкнута при 4*pi (Спин 1/2)
theta = np.linspace(0, 4 * np.pi, num_points)


def get_wave_mode(theta, phase_shift=0, frame=0, is_positron=False):
    """
    Рассчитывает идеально гладкие 3D координаты самосопряженного рог-тора (Horn Torus).
    Учитывает спиральное кручение, эффект Гуи и инверсию Картанова репера.
    """
    time_phase = (2 * np.pi * frame) / num_frames

    # 1. Эффект Гуи: фазовый сдвиг при прохождении через центральный фокус (z=0).
    #    Сдвиг фазы Gouy = arctan(z / z_R).
    #    ПРИБЛИЖЕНИЕ ПЕРВОГО ПОРЯДКА: z оценивается по theta (z_approx), а не по
    #    итоговому phi, иначе уравнение становится неявным. Расхождение между
    #    z_approx и фактическим z = r*sin(phi) растёт вместе с gouy_phase.
    z_approx = r * np.sin(0.5 * theta)
    gouy_phase = np.arctan(z_approx / z_R)

    # 2. Расчет фазы вращения волнового фронта
    if not is_positron:
        phi = 0.5 * theta + phase_shift - time_phase + gouy_phase
    else:
        # Для позитрона инвертируется пространственная спиральность (киральность)
        phi = -(0.5 * theta + phase_shift) - time_phase + gouy_phase

    # 3. Чистая геометрия Horn Torus (при R = r внутренняя кромка касается оси Z в нуле).
    #    Коэффициент (R + r * cos(phi)) при phi = pi обращается строго в 0
    radial_amplitude = R + r * np.cos(phi)

    x = radial_amplitude * np.cos(theta)
    y = radial_amplitude * np.sin(theta)
    z = r * np.sin(phi)

    # 4. Интенсивность поля (квадрат амплитуды волновой функции), нормирована на [0,1]
    color_intensity = (np.cos(phi) + 1.0) / 2.0

    # 5. Поперечный репер Картанова кручения (норма тождественно равна 1)
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    if is_positron:
        # Зарядовое сопряжение инвертирует ориентацию векторов кручения поля
        tx, ty, tz = -tx, -ty, -tz

    return x, y, z, color_intensity, (tx, ty, tz)


# --- Инициализация двух панелей (Электрон и Позитрон) ---
fig = plt.figure(figsize=(16, 8), facecolor='black')

ax_elec = fig.add_subplot(121, projection='3d')
ax_elec.set_facecolor('black')
ax_elec.set_title("ЭЛЕКТРОН (Левая хиральность, заряд -e)",
                  color='cyan', fontsize=14, pad=20)

ax_posi = fig.add_subplot(122, projection='3d')
ax_posi.set_facecolor('black')
ax_posi.set_title("ПОЗИТРОН (Правая хиральность, заряд +e)",
                  color='magenta', fontsize=14, pad=20)

axes = [ax_elec, ax_posi]

torus = Torus(R=R, r=r)

for ax in axes:
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim(-(R + r), (R + r))
    ax.set_ylim(-(R + r), (R + r))
    ax.set_zlim(-(R + r), (R + r))
    ax.axis('off')
    torus.plot(ax, 32, color='dimgray', alpha=0.75, linewidth=0.8)
    # Ракурс задаётся ОБЕИМ панелям (раньше строка стояла вне цикла
    # и применялась только к последней) и внутри update() не трогается
    ax.view_init(elev=30, azim=0)

cmap = plt.get_cmap('cool')

lines = [None, None]
quivers_data = [None, None]
gyro_data = [None, None]

# --- Слой гирации (2.8.5): точки кернового кольца и их смещения ---
NUM_GYRO = 18
GYRO_AMP = 0.42
theta_gyro = np.linspace(0, 2 * np.pi, NUM_GYRO, endpoint=False)

for p_idx, ax in enumerate(axes):
    ax.text2D(0.02, 0.02,
              "алые: гирация (~40% момента, спин)\n"
              "намотка фазы: ~60%, орбита (2.8.5)",
              transform=ax.transAxes, color='lightgray', fontsize=9)


def gyration(frame, is_positron):
    """u = A [cos(Phi) e_rho + sin(Phi) e_z]; Phi = +-theta - w t (+ pi у позитрона).

    Знаки согласованы с лентой каждой панели: электрон phi = +theta/2 - w t
    => гирация Phi = +theta - w t; позитрон phi = -theta/2 - w t =>
    Phi = -theta - w t. Добавка pi у позитрона — то же C-сопряжение
    (инверсия смещения), что и знак минус у жёлтых стрелок репера.
    Намотка гирации целочисленная (n = 1): векторное поле однозначно,
    полуоборот на обход — свойство репера, а не смещения."""
    time_phase = (2 * np.pi * frame) / num_frames
    sign = -1.0 if is_positron else 1.0
    Phi = sign * theta_gyro - time_phase + (np.pi if is_positron else 0.0)
    ox = R * np.cos(theta_gyro)
    oy = R * np.sin(theta_gyro)
    oz = np.zeros_like(theta_gyro)
    gx = GYRO_AMP * np.cos(Phi) * np.cos(theta_gyro)
    gy = GYRO_AMP * np.cos(Phi) * np.sin(theta_gyro)
    gz = GYRO_AMP * np.sin(Phi)
    return ox, oy, oz, gx, gy, gz

for p_idx, ax in enumerate(axes):
    lc = Line3DCollection(np.zeros((1, 2, 3)), cmap=cmap, lw=2.5, alpha=0.8)
    lc.set_clim(0.0, 1.0)   # интенсивность всегда в [0,1]
    ax.add_collection3d(lc)
    lines[p_idx] = lc


def update(frame):
    return_objects = []

    for p_idx, ax in enumerate(axes):
        if quivers_data[p_idx] is not None:
            quivers_data[p_idx].remove()
            quivers_data[p_idx] = None
        if gyro_data[p_idx] is not None:
            gyro_data[p_idx].remove()
            gyro_data[p_idx] = None

        is_pos = (p_idx == 1)
        x, y, z, colors, (tx, ty, tz) = get_wave_mode(theta, 0, frame,
                                                      is_positron=is_pos)

        # Рендеринг основного гладкого каната
        lines[p_idx].set_segments(to_segments(x, y, z))
        lines[p_idx].set_array(colors[:-1])

        # Переменный шаг: сгущение стрелок к горловине рог-тора.
        # При R = r горловина вырождается в точку (R - r = 0), поэтому
        # core_metric = |r_xy| / 2r и пробегает ровно [0, 1].
        q = adaptive_indices(x, y, R, r, base=6, span=24)

        # Рендеринг поля кручения
        quivers_data[p_idx] = ax.quiver(x[q], y[q], z[q], tx[q], ty[q], tz[q],
                                        length=0.45, color='yellow',
                                        normalize=True)

        # ОТО-прозрачность на основе того же евклидова инварианта
        quivers_data[p_idx].set_alpha(
            alpha_profile(x[q], y[q], R, r, peak=0.9, drop=0.75))

        # Гирация: физическое смещение среды на керновом кольце
        ox, oy, oz, gx, gy, gz = gyration(frame, is_pos)
        gyro_data[p_idx] = ax.quiver(ox, oy, oz, gx, gy, gz,
                                     color='#FF4433', alpha=0.95,
                                     arrow_length_ratio=0.25)

        return_objects.extend([lines[p_idx], quivers_data[p_idx],
                               gyro_data[p_idx]])

    # fig.canvas.draw_idle() здесь НЕ нужен: FuncAnimation сама инициирует
    # перерисовку, а лишний вызов давал ~45x просадку кадра
    return return_objects


# Запуск анимации (blit=False обязателен для интерактивного вращения)
ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              interval=20, blit=False)
plt.show()
