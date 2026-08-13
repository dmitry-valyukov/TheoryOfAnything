import math

import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import adaptive_indices, alpha_profile, to_segments
from lib.torus import Torus

# --- Параметры модели протона ---
r = 1.0                  # Радиус "шланга"
R = math.sqrt(3) * r     # Большой радиус тора
num_points = 400
num_frames = 360         # Цикл идеального зацикливания

# Траектория замкнута при 4*pi (Спин 1/2)
theta = np.linspace(0, 4 * np.pi, num_points)


def get_quark_mode(theta, spatial_shift, frame, quark_type):
    """Рассчитывает 3D координаты, фазу и векторы поляризации для протона (uud)"""
    time_phase = (2 * np.pi * frame) / num_frames

    if quark_type == 'u':
        # u-кварк (заряд +2/3): базовая спиральная намотка
        phi = 0.5 * theta + spatial_shift - time_phase
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
    else:
        # d-кварк (заряд -1/3): инвертированная намотка с удвоенным пространственным периодом
        # Коэффициент 1.0 перед theta и 0.5 перед time_phase геометрически меняют
        # взаимный индекс зацепления Хопфа ровно в 2 раза относительно u-кварка,
        # при этом траектория гарантированно замыкается без разрывов!
        phi = -0.5 * theta + spatial_shift + 0.5 * time_phase

        # Модифицируем геометрию d-моды: добавляем пространственную 3D-осцилляцию (моду Лиссажу)
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        # Дополнительное пространственное кручение, которое меняет топологический индекс зацепления
        z = r * np.sin(phi) * np.cos(0.5 * theta)

    color_phase = (phi % (2 * np.pi)) / (2 * np.pi)

    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    return x, y, z, color_phase, (tx, ty, tz)


# --- Инициализация 3D сцены ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.set_title("Proton: 3 суб-вихря (кварка) и 8 глюонных фазовых переходов",
             color='white', fontsize=14, pad=20)

# Фиксация пропорций 1:1:1
ax.set_box_aspect((1, 1, 1), zoom=1.5)

# Очистка осей
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim(-(R + r), (R + r))
ax.set_ylim(-(R + r), (R + r))
ax.set_zlim(-(R + r), (R + r))
ax.axis('off')

# Начальный ракурс задаётся здесь; внутри update() его больше не трогаем
ax.view_init(elev=30, azim=0)

# Объемная сетка единого "шланга" протона
Torus(R=R, r=r).plot(ax, 32, color='dimgray', alpha=0.75, linewidth=0.8)

# Цветовая карта jet/rainbow идеально подходит для отображения циклического "цвета" кварков
cmap = plt.get_cmap('gist_rainbow')

lines = []
quivers = None

for _ in range(3):
    lc = Line3DCollection(np.zeros((1, 2, 3)), cmap=cmap, lw=3.0, alpha=0.8)
    # Фаза нормирована на [0,1); без явного clim шкала намертво фиксируется
    # по диапазону первого кадра
    lc.set_clim(0.0, 1.0)
    ax.add_collection3d(lc)
    lines.append(lc)


def update(frame):
    global quivers
    if quivers is not None:
        quivers.remove()

    # Время калибровочного цикла
    time_phase = (2 * np.pi * frame) / num_frames

    # Динамическая модуляция углов (ГЛЮОННЫЙ ОБМЕН)
    # d-кварк колеблется как маятник с частотой калибровочного поля
    d_oscillation = 0.5 * np.sin(2 * time_phase)

    # Помещаем d-кварк в базовую позицию 90 градусов (pi/2) — ровно посередине между u-кварками
    shift_d = np.pi / 2 + d_oscillation

    # u-кварки разнесены на 180 градусов (pi), но реагируют на приближение
    # d-кварка, слегка сжимаясь/разжимаясь (эффект отдачи КХД-поля)
    u_recoil = (np.pi / 8) * np.cos(2 * time_phase)
    shift_u1 = u_recoil                 # u1 колеблется около 0
    shift_u2 = np.pi + u_recoil         # u2 колеблется около 180 градусов

    # Массив динамических сдвигов для текущего кадра
    current_shifts = [shift_u1, shift_u2, shift_d]

    all_x, all_y, all_z = [], [], []
    all_tx, all_ty, all_tz = [], [], []
    all_alphas = []  # Топологическая плотность кирального инварианта Черна-Саймонса

    for i in range(3):
        # Первые два (0, 1) — u-кварки, третий (2) — d-кварк
        q_type = 'u' if i < 2 else 'd'
        x, y, z, colors, (tx, ty, tz) = get_quark_mode(
            theta, current_shifts[i], frame, quark_type=q_type)

        # Динамическая разметка нити на 3D сегменты для калибровочного обмена цветом
        lines[i].set_segments(to_segments(x, y, z))
        lines[i].set_array(colors[:-1])

        # Переменный шаг: ОТО-сгущение стрелок к горловине тора
        q = adaptive_indices(x, y, R, r, base=5, span=25)

        all_x.extend(x[q])
        all_y.extend(y[q])
        all_z.extend(z[q])
        all_tx.extend(tx[q])
        all_ty.extend(ty[q])
        all_tz.extend(tz[q])

        # Модуляция яркости: градиент киральной аномалии Черна-Саймонса
        all_alphas.extend(alpha_profile(x[q], y[q], R, r, peak=0.85, drop=0.60))

    # Отрисовка векторного поля связности вакуума с учетом метрического ОТО-сгущения
    quivers = ax.quiver(all_x, all_y, all_z, all_tx, all_ty, all_tz,
                        length=0.3, color='cyan', normalize=True)
    quivers.set_alpha(all_alphas)

    return lines + [quivers]


# Запуск идеально зацикленной анимации протона.
# Период d-кварка вдвое длиннее (0.5 * time_phase), поэтому кадров 2 * num_frames
ani = animation.FuncAnimation(fig, update, frames=num_frames * 2,
                              interval=20, blit=False)
plt.show()
