import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- Параметры модели нейтрона ---
r = 1.0       # Базовый радиус "шланга"
R = 2 * r     # Большой радиус тора (самосопряженный, нулевой просвет)
z_R = 1.0     # Рэлеевская длина для эффекта Гуи
num_points = 600  # Увеличиваем плотность точек для плавной кривизны лепестков
num_frames = 120  # Цикл идеального зацикливания

# Коэффициент ОТО-деформации упругости вакуума под действием плотности энергии
k_oto = 0.28

# Траектория замкнута при 4*pi (Спин 1/2)
theta = np.linspace(0, 4 * np.pi, num_points)

def get_deformed_quark_mode(theta, spatial_shift=0, frame=0, is_flipped_quark=False):
    """Рассчитывает 3D координаты и векторы с учетом ИСТИННОЙ ОТО-асимметрии u и d лепестков"""
    time_phase = (2 * np.pi * frame) / num_frames

    # Базовый шаг фазы по мёбиусовскому шлангу
    phi = 0.5 * theta + spatial_shift - time_phase

    # ИСПРАВЛЕННЫЙ ОТО-БЛОК: Раздельная геометрия для u- и d-кварков.
    # Косинус от (theta) гарантирует, что деформация привязана к глобальным осям тора.
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

    # Строгий геометрический расчет векторов нормали
    tx = np.cos(phi) * np.cos(theta)
    ty = np.cos(phi) * np.sin(theta)
    tz = np.sin(phi)
    t_norm = np.sqrt(tx**2 + ty**2 + tz**2)
    tx, ty, tz = tx/t_norm, ty/t_norm, tz/t_norm

    # Правило нижнего (d) кварка: инверсия направления векторов поля
    if is_flipped_quark:
        tx = -tx
        ty = -ty
        tz = -tz
        color_phase = ((phi + np.pi) % (2 * np.pi)) / (2 * np.pi)

    return x, y, z, color_phase, (tx, ty, tz)


# --- Инициализация 3D сцены ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.set_title("НЕЙТРОН (udd) в ОТО: Динамическая асимметрия траекторий и лепестков поля", color='white', fontsize=12, pad=20)

ax.set_box_aspect((1, 1, 1))

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim(-(R+r+0.5), (R+r+0.5))
ax.set_ylim(-(R+r+0.5), (R+r+0.5))
ax.set_zlim(-(R+r+0.5), (R+r+0.5))
ax.axis('off')

# Генерация базового каркаса для объема (оставляем круглым, чтобы была видна деформация нитей)
u = np.linspace(0, 2 * np.pi, 32)
v = np.linspace(0, 2 * np.pi, 16)
U, V = np.meshgrid(u, v)
X_torus = (R + r * np.cos(V)) * np.cos(U)
Y_torus = (R + r * np.cos(V)) * np.sin(U)
Z_torus = r * np.sin(V)
ax.plot_wireframe(X_torus, Y_torus, Z_torus, color='dimgray', alpha=0.15, linewidth=0.6, zorder=1)

# Пространственные сдвиги для 3-х суб-вихрей
quark_shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]
cmap = plt.cm.gist_rainbow

lines = []
quivers_u = None
quivers_d = None

for i in range(3):
    dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
    lc = Line3DCollection(dummy, cmap=cmap, lw=3.5, alpha=0.85)
    ax.add_collection3d(lc)
    lines.append(lc)

def update(frame):
    global quivers_u, quivers_d
    if quivers_u is not None:
        quivers_u.remove()
    if quivers_d is not None:
        quivers_d.remove()

    camera_azim = 0  # Статическая камера для фиксации асимметрии лепестков

    u_x, u_y, u_z, u_tx, u_ty, u_tz = [], [], [], [], [], []
    d_x, d_y, d_z, d_tx, d_ty, d_tz = [], [], [], [], [], []

    for i in range(3):
        is_d_quark = (i == 1 or i == 2)

        # Получаем координаты из новой деформированной ОТО-функции
        x, y, z, colors, (tx, ty, tz) = get_deformed_quark_mode(theta, quark_shifts[i], frame, is_flipped_quark=is_d_quark)

        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lines[i].set_segments(segments)
        lines[i].set_array(colors[:-1])

        if not is_d_quark:
            u_x.extend(x[::24])
            u_y.extend(y[::24])
            u_z.extend(z[::24])
            u_tx.extend(tx[::24])
            u_ty.extend(ty[::24])
            u_tz.extend(tz[::24])
        else:
            d_x.extend(x[::24])
            d_y.extend(y[::24])
            d_z.extend(z[::24])
            d_tx.extend(tx[::24])
            d_ty.extend(ty[::24])
            d_tz.extend(tz[::24])

    # 1. Верхний u-кварк: длинные стрелки (length=0.6) на широких внешних лепестках
    quivers_u = ax.quiver(u_x, u_y, u_z, u_tx, u_ty, u_tz,
                          length=0.6, color='white', alpha=0.45, normalize=True)

    # 2. Нижние d-кварки: короткие стрелки (length=0.3) уходят строго ВНУТРЬ поджатых треков
    quivers_d = ax.quiver(d_x, d_y, d_z, d_tx, d_ty, d_tz,
                          length=0.3, color='white', alpha=0.55, normalize=True)

    ax.view_init(elev=90, azim=camera_azim)
    return lines + [quivers_u, quivers_d]

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=40, blit=False)
plt.show()
