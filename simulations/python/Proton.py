import math
import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Параметры модели протона ---
r = 1.0       # Радиус "шланга"
R = math.sqrt(3) * r     # Большой радиус тора
z_R = 1.0     # Рэлеевская длина для эффекта Гуи
num_points = 400
num_frames = 360  # Цикл идеального зацикливания

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
ax.set_title("Proton: 3 суб-вихря (кварка) и 8 глюонных фазовых переходов", color='white', fontsize=14, pad=20)

# Фиксация пропорций 1:1:1
ax.set_box_aspect((1, 1, 1), zoom = 1.5)

# Очистка осей
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim(-(R+r), (R+r))
ax.set_ylim(-(R+r), (R+r))
ax.set_zlim(-(R+r), (R+r))
ax.axis('off')

# ИСПРАВЛЕНИЕ 1: Задаем начальный ракурс здесь. Внутри update() его больше не трогаем!
ax.view_init(elev=30, azim=0)

# Генерация объемной сетки единого "шланга" протона
u = np.linspace(0, 2 * np.pi, 32)
v = np.linspace(0, 2 * np.pi, 16)
U, V = np.meshgrid(u, v)
X_torus = (R + r * np.cos(V)) * np.cos(U)
Y_torus = (R + r * np.cos(V)) * np.sin(U)
Z_torus = r * np.sin(V)
ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='dimgray', alpha=0.75, linewidth=0.8, zorder=1)

# Пространственные сдвиги для 3-х кварков (разнос по углам сечения шланга на 120 градусов)
#quark_shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]

# Цветовая карта jet/rainbow идеально подходит для отображения циклического "цвета" кварков
cmap = plt.cm.gist_rainbow

lines = []
quivers = None

for i in range(3):
    # Каноничная инициализация через фиктивный 3D сегмент
    dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
    lc = Line3DCollection(dummy, cmap=cmap, lw=3.0, alpha=0.8)
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

    # 2. u-кварки разнесены на 180 градусов (pi),
    # но реагируют на приближение d-кварка, слегка сжимаясь/разжимаясь (эффект отдачи КХД-поля)
    u_recoil = (np.pi/8) * np.cos(2 * time_phase)
    shift_u1 = u_recoil                 # u1 колеблется около 0
    shift_u2 = np.pi*0.5 + u_recoil     # u2 колеблется около 90 градусов

    # Массив динамических сдвигов для текущего кадра
    current_shifts = [shift_u1, shift_u2, shift_d]

    all_x, all_y, all_z = [], [], []
    all_tx, all_ty, all_tz = [], [], []
    all_alphas = []  # Топологическая плотность кирального инварианта Черна-Саймонса

    fx, fy, fz = [], [], []
    ftx, fty, ftz = [], [], []

    for i in range(3):
        # Вычисление пространственно-временных координат текущей киральной гармоники
        # Первые два (0, 1) — u-кварки, третий (2) — d-кварк
        q_type = 'u' if i < 2 else 'd'
        x, y, z, colors, (tx, ty, tz) = get_quark_mode(theta, current_shifts[i], frame, quark_type=q_type)

        # Динамическая разметка нити на 3D сегменты для калибровочного обмена цветом
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if i < 3: # 0,1 - u-кварки, 2 - d-кварк
            lines[i].set_segments(segments)
            lines[i].set_array(colors[:-1])
        else:
            lines[i].set_segments([])

        # Определение положения бегущего фронта волнового пакета (кваркового керна)
        front_idx = int((frame * (num_points / num_frames)) % num_points)

        fx.append(x[front_idx])
        fy.append(y[front_idx])
        fz.append(z[front_idx])
        ftx.append(tx[front_idx])
        fty.append(ty[front_idx])
        ftz.append(tz[front_idx])

        # Алгоритм переменного шага: ОТО-сгущение на основе евклидова инварианта горловины
        quiver_indices = []
        current_idx = 0

        while current_idx < num_points:
            quiver_indices.append(current_idx)

            # Пространственное расстояние киральной нити до центральной оси сингулярности (Z)
            r_current = np.sqrt(x[current_idx]**2 + y[current_idx]**2)

            # Нормировка метрики ядра (R-r=1.0 -> максимальное сжатие, R+r=3.0 -> внешняя периферия)
            core_metric = np.clip((r_current - (R - r)) / (2.0 * r), 0.0, 1.0)

            # Динамическое прореживание стрелок кручения вакуума в зависимости от градиента плотности
            current_step = int(5 + 25 * (core_metric**2))
            current_idx += current_step
            # Динамическое прореживание стрелок кручения вакуума в зависимости от градиента плотности
            current_step = int(5 + 25 * (core_metric**2))
            current_idx += current_step
            # Динамическое прореживание стрелок кручения вакуума в зависимости от градиента плотности
            current_step = int(5 + 25 * (core_metric**2))
            current_idx += current_step
        q_idxs = np.array(quiver_indices)
        # Формирование локальных векторов кручения физического вакуума
        all_x.extend(x[q_idxs])
        all_y.extend(y[q_idxs])
        all_z.extend(z[q_idxs])
        all_tx.extend(tx[q_idxs])
        all_ty.extend(ty[q_idxs])
        all_tz.extend(tz[q_idxs])

        # Модуляция яркости (альфы): визуализация градиента киральной аномалии Черна-Саймонса
        for idx in q_idxs:
            r_curr = np.sqrt(x[idx]**2 + y[idx]**2)
            c_met = np.clip((r_curr - (R - r)) / (2.0 * r), 0.0, 1.0)
            # Экстремальное сгущение и вспышка поля в ядре (0.85) при затухании на периферии (0.25)
            all_alphas.append(0.85 - 0.60 * (c_met**2))

    # Отрисовка векторного поля связности вакуума с учетом метрического ОТО-сгущения
    quivers = ax.quiver(all_x, all_y, all_z, all_tx, all_ty, all_tz,
                        length=0.3, color='cyan', alpha=0.5, normalize=True)

    #fig.canvas.draw_idle()

    return lines + [quivers]

# Запуск идеально зацикленной анимации протона
ani = animation.FuncAnimation(fig, update, frames=num_frames*2, interval=20, blit=False)
plt.show()
