import math

import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import adaptive_indices, alpha_profile, to_segments
from lib.torus import Torus

# --- Параметры модели DDD-бариона ---
r = 1.0                  # Радиус "шланга"
R = math.sqrt(3) * r     # Большой радиус тора
num_points = 400
num_frames = 360         # Цикл идеального зацикливания

# Траектория замкнута при 4*pi (Спин 1/2)
theta = np.linspace(0, 4 * np.pi, num_points)


def get_quark_mode(theta, spatial_shift=0, frame=0):
    """Рассчитывает 3D координаты, фазу и векторы поляризации для кварка"""
    # Время смены фазы (синхронизация обмена глюонами со скоростью света)
    time_phase = (2 * np.pi * frame) / num_frames

    # Мёбиусов шаг фазы + пространственный разнос кварков внутри сечения шланга
    phi = 0.5 * theta + spatial_shift - time_phase

    # Геометрия круглого тора
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    # Расчет "цвета" (динамической фазы калибровочного глюонного поля),
    # нормированный на [0,1) для циклической карты hsv/rainbow
    color_phase = (phi % (2 * np.pi)) / (2 * np.pi)

    # Поперечный вектор электрического поля кварка, направлен радиально
    # от оси шланга. Норма тождественно равна 1.
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    return x, y, z, color_phase, (tx, ty, tz)


# --- Инициализация 3D сцены ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.set_title("DDD-барион: 3 суб-вихря (кварка) и 8 глюонных фазовых переходов",
             color='white', fontsize=14, pad=20)

# Фиксация пропорций 1:1:1
ax.set_box_aspect((1, 1, 1))

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

# Объемная сетка единого "шланга"
Torus(R=R, r=r).plot(ax, 32, color='dimgray', alpha=0.75, linewidth=0.8)

# Пространственные сдвиги для 3-х кварков (разнос по сечению шланга на 120 градусов)
quark_shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]

cmap = plt.get_cmap('gist_rainbow')

lines = []
quivers = None

for _ in range(3):
    lc = Line3DCollection(np.zeros((1, 2, 3)), cmap=cmap, lw=3.0, alpha=0.8)
    lc.set_clim(0.0, 1.0)   # фаза всегда в [0,1)
    ax.add_collection3d(lc)
    lines.append(lc)


def update(frame):
    global quivers
    if quivers is not None:
        quivers.remove()

    all_x, all_y, all_z = [], [], []
    all_tx, all_ty, all_tz = [], [], []
    all_alphas = []  # Топологическая плотность кирального инварианта Черна-Саймонса

    for i in range(3):
        # Вычисление пространственно-временных координат текущей киральной гармоники
        x, y, z, colors, (tx, ty, tz) = get_quark_mode(theta, quark_shifts[i], frame)

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

    # fig.canvas.draw_idle() здесь НЕ нужен: FuncAnimation сама инициирует
    # перерисовку, а лишний вызов давал ~45x просадку кадра

    return lines + [quivers]


ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              interval=20, blit=False)
plt.show()
