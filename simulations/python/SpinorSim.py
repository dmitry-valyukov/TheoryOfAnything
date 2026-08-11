import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Параметры модели самосопряженного тора ---
r = 1.0       # Радиус "шланга"
R = 1 * r     # Большой радиус тора (нулевой просвет)
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

    # 1. Эффект Гуи: фазовый сдвиг при прохождении через центральный фокус (z=0)
    # Сдвиг фазы Gouy = arctan(z / z_R). Аппроксимируем пространственную координату Z через theta
    z_approx = r * np.sin(0.5 * theta)
    gouy_phase = np.arctan(z_approx / z_R)

    # 2. Расчет фазы вращения волнового фронта
    if not is_positron:
        phi = 0.5 * theta + phase_shift - time_phase + gouy_phase
    else:
        # Для позитрона инвертируется пространственная спиральность (киральность)
        phi = -(0.5 * theta + phase_shift) - time_phase + gouy_phase

    # 3. Чистая геометрия Horn Torus (при R = r внутренняя кромка касается оси Z в нуле)
    # Коэффициент (R + r * cos(phi)) при phi=pi обращается строго в 0
    radial_amplitude = R + r * np.cos(phi)

    x = radial_amplitude * np.cos(theta)
    y = radial_amplitude * np.sin(theta)
    z = r * np.sin(phi)

    # 4. Интенсивность поля (квадрат амплитуды волновой функции)
    color_intensity = (np.cos(phi) + 1.0) / 2.0

    # 5. Поперечный репер Картанова кручения
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    if is_positron:
        # Зарядовое сопряжение инвертирует ориентацию векторов кручения поля
        tx, ty, tz = -tx, -ty, -tz

    return x, y, z, color_intensity, (tx, ty, tz)

def get_wave_mode2(theta, phase_shift=0, frame=0, is_positron=False):
    """Рассчитывает идеально гладкие 3D координаты, фазу и интенсивность поля"""
    time_phase = (2 * np.pi * frame) / num_frames

    # Возвращаем линейную пространственную фазу для исключения геометрических изломов
    if not is_positron:
        phi = 0.5 * theta + phase_shift - time_phase
    else:
        phi = -(0.5 * theta + phase_shift) - time_phase

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    # Плавная цветовая интенсивность поля
    color_intensity = (np.cos(phi) + 1.0) / 2.0

    # Расчет поперечного вектора Картанова кручения
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    if is_positron:
        tx, ty, tz = -tx, -ty, -tz

    return x, y, z, color_intensity, (tx, ty, tz)

# --- Инициализация двух окон (Электрон и Позитрон) ---
fig = plt.figure(figsize=(16, 8), facecolor='black')

ax_elec = fig.add_subplot(121, projection='3d')
ax_elec.set_facecolor('black')
ax_elec.set_title("ЭЛЕКТРОН (Левая хиральность, заряд -e)", color='cyan', fontsize=14, pad=20)

ax_posi = fig.add_subplot(122, projection='3d')
ax_posi.set_facecolor('black')
ax_posi.set_title("ПОЗИТРОН (Правая хиральность, заряд +e)", color='magenta', fontsize=14, pad=20)

axes = [ax_elec, ax_posi]

# Генерация полупрозрачной сетки тора для обоих окон
u = np.linspace(0, 2 * np.pi, 32)
v = np.linspace(0, 2 * np.pi, 16)
U, V = np.meshgrid(u, v)
X_torus = (R + r * np.cos(V)) * np.cos(U)
Y_torus = (R + r * np.cos(V)) * np.sin(U)
Z_torus = r * np.sin(V)

for ax in axes:
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim(-(R+r), (R+r))
    ax.set_ylim(-(R+r), (R+r))
    ax.set_zlim(-(R+r), (R+r))
    ax.axis('off')
    ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='dimgray', alpha=0.75, linewidth=0.8, zorder=1)

# ИСПРАВЛЕНИЕ 1: Задаем начальный ракурс здесь. Внутри update() его больше не трогаем!
ax.view_init(elev=30, azim=0)

shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]
cmaps = [plt.cm.cool, plt.cm.plasma, plt.cm.viridis]

lines_data = [[], []]
quivers_data = [None, None]

for p_idx in range(2):
    ax = axes[p_idx]
    for i in range(3):
        dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
        lc = Line3DCollection(dummy, cmap=cmaps[i], lw=2.5, alpha=0.8)
        ax.add_collection3d(lc)
        lines_data[p_idx].append(lc)

def update(frame):
    global quivers_data

    # Очистка стрелок quiver прошлого кадра
    for p_idx in range(2):
        if quivers_data[p_idx] is not None:
            try:
                quivers_data[p_idx].remove()
            except Exception:
                pass
            quivers_data[p_idx] = None

    return_objects = []

    for p_idx in range(2):
        ax = axes[p_idx]
        is_pos = (p_idx == 1)

        all_x, all_y, all_z = [], [], []
        all_tx, all_ty, all_tz = [], [], []
        all_alphas = []  # Массив для нелинейной прозрачности ОТО-ямы

        for i in range(1):
            x, y, z, colors, (tx, ty, tz) = get_wave_mode(theta, shifts[i], frame, is_positron=is_pos)

            # Рендеринг основного гладкого каната
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lines_data[p_idx][i].set_segments(segments)
            lines_data[p_idx][i].set_array(colors[:-1])

            # Алгоритм переменного шага: сгущение на основе евклидова радиуса
            quiver_indices = []
            current_idx = 0

            while current_idx < num_points:
                quiver_indices.append(current_idx)

                # Вычисляем текущее расстояние точки до оси Z в плоскости XY
                r_current = np.sqrt(x[current_idx]**2 + y[current_idx]**2)

                # Минимальный радиус (горловина) равен R - r = 2.0 - 1.0 = 1.0
                # Максимальный радиус (внешний край) равен R + r = 2.0 + 1.0 = 3.0
                # Нормируем близость к ядру: 0.0 - строго в горловине, 1.0 - на внешнем крае
                core_metric = (r_current - (R - r)) / (2.0 * r)
                core_metric = np.clip(core_metric, 0.0, 1.0) # Защита от выхода за границы

                # В самом ядре (core_metric -> 0) шаг равен 6 ячейкам (максимальная плотность)
                # На внешней рыхлой стороне (core_metric -> 1) шаг расширяется до 30 ячеек
                current_step = int(6 + 24 * (core_metric**2))
                current_idx += current_step

            q_idxs = np.array(quiver_indices)

            # Нарезаем массивы векторов поля
            all_x.extend(x[q_idxs])
            all_y.extend(y[q_idxs])
            all_z.extend(z[q_idxs])
            all_tx.extend(tx[q_idxs])
            all_ty.extend(ty[q_idxs])
            all_tz.extend(tz[q_idxs])

            # Вычисляем ОТО-альфу (яркость) на основе того же евклидова инварианта
            for idx in q_idxs:
                r_curr = np.sqrt(x[idx]**2 + y[idx]**2)
                c_met = np.clip((r_curr - (R - r)) / (2.0 * r), 0.0, 1.0)
                all_alphas.append(0.9 - 0.75 * (c_met**2))

        # Рендеринг поля кручения
        quivers_data[p_idx] = ax.quiver(all_x, all_y, all_z, all_tx, all_ty, all_tz,
                                        length=0.45, color='yellow',
                                        normalize=True)

        # Передаем массив ОТО-прозрачности напрямую в коллекцию стрелок
        quivers_data[p_idx].set_alpha(all_alphas)

        return_objects.extend(lines_data[p_idx] + [quivers_data[p_idx]])

    fig.canvas.draw_idle()
    return return_objects

# Запуск анимации (blit=False обязателен для интерактивного вращения)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20, blit=False)
plt.show()
