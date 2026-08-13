import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import to_segments
from lib.torus import Torus

# --- Параметры модели нейтрона ---
r = 1.0           # Базовый радиус "шланга"
R = 2 * r         # Большой радиус тора
num_points = 600  # Плотность точек для плавной кривизны лепестков
num_frames = 120  # Цикл идеального зацикливания

# Коэффициент ОТО-деформации упругости вакуума под действием плотности энергии
k_oto = 0.28

# Шаг прореживания стрелок поля
ARROW_STEP = 24

# Ракурс: строго сверху, чтобы асимметрия лепестков была видна как есть
VIEW_ELEV, VIEW_AZIM = 90, 0

# Траектория замкнута при 4*pi (Спин 1/2)
theta = np.linspace(0, 4 * np.pi, num_points)


def get_deformed_quark_mode(theta, spatial_shift=0, frame=0, is_flipped_quark=False):
    """3D координаты и векторы с учетом ОТО-асимметрии u- и d-лепестков"""
    time_phase = (2 * np.pi * frame) / num_frames

    # Базовый шаг фазы по мёбиусовскому шлангу
    phi = 0.5 * theta + spatial_shift - time_phase

    # Раздельная геометрия для u- и d-кварков.
    # Косинус от theta привязывает деформацию к глобальным осям тора.
    if not is_flipped_quark:
        # u-кварк: упруго расширяется наружу от центральной оси шланга
        r_deformed = r * (1.0 + k_oto * np.cos(theta + spatial_shift))
    else:
        # d-кварки: жестко поджимаются внутрь, к центральному просвету тора
        r_deformed = r * (1.0 - k_oto * np.cos(theta + spatial_shift))

    # Геометрия асимметричного упругого тора
    x = (R + r_deformed * np.cos(phi)) * np.cos(theta)
    y = (R + r_deformed * np.cos(phi)) * np.sin(theta)
    z = r_deformed * np.sin(phi)

    # Расчет "цвета" (фазы)
    color_phase = (phi % (2 * np.pi)) / (2 * np.pi)

    # Векторы нормали. Норма cos^2(phi)*(cos^2+sin^2)(theta) + sin^2(phi)
    # тождественно равна 1, поэтому явная нормировка не нужна
    # (к тому же ниже стоит quiver(normalize=True)).
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)

    # Правило нижнего (d) кварка: инверсия направления векторов поля
    if is_flipped_quark:
        tx, ty, tz = -tx, -ty, -tz
        color_phase = ((phi + np.pi) % (2 * np.pi)) / (2 * np.pi)

    return x, y, z, color_phase, (tx, ty, tz)


# --- Инициализация 3D сцены ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.set_title("НЕЙТРОН (udd) в ОТО: Динамическая асимметрия траекторий и лепестков поля",
             color='white', fontsize=12, pad=20)

ax.set_box_aspect((1, 1, 1))

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim(-(R + r + 0.5), (R + r + 0.5))
ax.set_ylim(-(R + r + 0.5), (R + r + 0.5))
ax.set_zlim(-(R + r + 0.5), (R + r + 0.5))
ax.axis('off')

# Ракурс фиксируется один раз здесь, а не на каждом кадре внутри update()
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

# Базовый каркас недеформированного тора — чтобы деформация нитей была видна
Torus(R=R, r=r).plot(ax, 32, color='dimgray', alpha=0.15, linewidth=0.6)

# Пространственные сдвиги для 3-х суб-вихрей
quark_shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]
cmap = plt.get_cmap('gist_rainbow')

lines = []
quivers_u = None
quivers_d = None

for _ in range(3):
    lc = Line3DCollection(np.zeros((1, 2, 3)), cmap=cmap, lw=3.5, alpha=0.85)
    lc.set_clim(0.0, 1.0)   # фаза всегда в [0,1)
    ax.add_collection3d(lc)
    lines.append(lc)


def update(frame):
    global quivers_u, quivers_d
    if quivers_u is not None:
        quivers_u.remove()
    if quivers_d is not None:
        quivers_d.remove()

    u_x, u_y, u_z, u_tx, u_ty, u_tz = [], [], [], [], [], []
    d_x, d_y, d_z, d_tx, d_ty, d_tz = [], [], [], [], [], []

    for i in range(3):
        is_d_quark = (i in (1, 2))

        x, y, z, colors, (tx, ty, tz) = get_deformed_quark_mode(
            theta, quark_shifts[i], frame, is_flipped_quark=is_d_quark)

        lines[i].set_segments(to_segments(x, y, z))
        lines[i].set_array(colors[:-1])

        s = slice(None, None, ARROW_STEP)
        if not is_d_quark:
            u_x.extend(x[s]); u_y.extend(y[s]); u_z.extend(z[s])
            u_tx.extend(tx[s]); u_ty.extend(ty[s]); u_tz.extend(tz[s])
        else:
            d_x.extend(x[s]); d_y.extend(y[s]); d_z.extend(z[s])
            d_tx.extend(tx[s]); d_ty.extend(ty[s]); d_tz.extend(tz[s])

    # 1. Верхний u-кварк: длинные стрелки на широких внешних лепестках
    quivers_u = ax.quiver(u_x, u_y, u_z, u_tx, u_ty, u_tz,
                          length=0.6, color='white', alpha=0.45, normalize=True)

    # 2. Нижние d-кварки: короткие стрелки уходят строго ВНУТРЬ поджатых треков
    quivers_d = ax.quiver(d_x, d_y, d_z, d_tx, d_ty, d_tz,
                          length=0.3, color='white', alpha=0.55, normalize=True)

    return lines + [quivers_u, quivers_d]


ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              interval=40, blit=False)
plt.show()
