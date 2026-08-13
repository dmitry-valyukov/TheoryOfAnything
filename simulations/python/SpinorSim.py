import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import adaptive_indices, alpha_profile, to_segments
from lib.torus import Torus

# --- Параметры модели самосопряженного тора ---
r = 1.0       # Радиус "шланга"
R = 1 * r     # Большой радиус тора (нулевой просвет => Horn Torus)
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

        return_objects.extend([lines[p_idx], quivers_data[p_idx]])

    # fig.canvas.draw_idle() здесь НЕ нужен: FuncAnimation сама инициирует
    # перерисовку, а лишний вызов давал ~45x просадку кадра
    return return_objects


# Запуск анимации (blit=False обязателен для интерактивного вращения)
ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              interval=20, blit=False)
plt.show()
