import matplotlib
"""
        - interactive backends:
          GTK3Agg, GTK3Cairo, GTK4Agg, GTK4Cairo, MacOSX, nbAgg, notebook, QtAgg,
          QtCairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo, Qt5Agg, Qt5Cairo

        - non-interactive backends:
          agg, cairo, pdf, pgf, ps, svg, template
"""
#matplotlib.use('QtAgg')
#%matplotlib ipympl

from typing import Optional, Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = None
ani = None

SIZE = 8.0  # базовый размер фигуры

def init(title: str, zoom, elev, azim, size: float = SIZE, theme: str = 'Dark'):
    global fig

    SIZE = size

    if theme == 'Dark' :
        color = 'white'
        background = 'black'
    else :
        color = 'black'
        background = 'white'

    fig = plt.figure(figsize=(SIZE, SIZE), facecolor=background)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d', facecolor=background)
    #ax.set_title(color)

    nsize = -SIZE
    ax.set_xlim(nsize, SIZE)
    ax.set_ylim(nsize, SIZE)
    ax.set_zlim(nsize, SIZE)

    fig.canvas.manager.set_window_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axis('off')

    ax.set_box_aspect([1.0, 1.0, 1.0], zoom = 4)
    ax.view_init(elev = 30, azim = -45)

    return ax

#class Animation()


def animate(func: callable,  # или более точно: Callable[[int], Iterable[Artist] | None]
            frames: Optional[Iterable | int | callable],
            interval ,
            **kwargs):
    global fig, ani
    ani = animation.FuncAnimation(fig, func, frames=frames, interval=interval, blit=False, **kwargs)
    plt.show()


type RgbaColor = tuple[float, float, float, float]

def rgba(color:ColorType, alpha:float=1.0):
    return colors.to_rgba(color, alpha=alpha)

# gist_rainbow, rainbow, hsv (циклическая), twilight, twilight_shifted.
# jet, turbo, nipy_spectral.
class Segments(Line3DCollection):
    def __init__(self, ax: Axes, cmap='hsv', linewidths=3.5, alpha=0.9):
        dummy = np.array([[[0, 0, 0], [0, 0, 0]]])
        cmap = plt.get_cmap(cmap)
        super().__init__(dummy, cmap=cmap, linewidths=linewidths, alpha=alpha)
        ax.add_collection3d(self)

class Arrows(Line3DCollection):
    def __init__(self, ax: Axes, num_arrows, color: RgbaColor, arrow_length_ratio: float = 0.15):
        self.num_arrows = num_arrows
        self.ratio = arrow_length_ratio

        # Генерируем правильную пустую матрицу нужного размера для родительского класса.
        # Каждой стрелке нужно 5 путей: 1 для тела (2 точки), 4 для зонтиков (3 точки).
        # Чтобы не усложнять инициализацию Matplotlib, сделаем ВСЕ сегменты размером в 3 точки.
        # Для тела третья точка просто продублирует вторую — это никак не повлияет на скорость.
        total_paths = num_arrows * 5
        init_segments = np.zeros((total_paths, 3, 3))

        # Инициализируем родительский класс. Передаем структуру сегментов.
        # Движок Matplotlib сам создаст правильный внутренний массив _paths нужной длины.
        super().__init__(init_segments, colors=color)

        # Кэшируем прямую ссылку на внутренние пути
        self.paths = self._paths

        # Предрассчитываем углы для 4 лучей юбки зонтика
        SQRT_2_DIV_2 = 0.5 * np.sqrt(2)  # или 0.7071067811865476
        self._cos_angles = np.array([1.0,  SQRT_2_DIV_2,  0.0, -SQRT_2_DIV_2])
        self._sin_angles = np.array([0.0,  SQRT_2_DIV_2,  1.0,  SQRT_2_DIV_2])

        zeros = np.zeros(num_arrows)
        self.zeros = zeros
        ones = np.ones(num_arrows) * 0.5
        self.recalc(zeros, zeros, zeros, zeros, zeros, ones, 1)

        # Добавляем коллекцию на оси
        ax.add_collection3d(self)

    def recalc(self, ox, oy, oz, vx, vy, vz, step = 1):
        num = self.num_arrows
        ratio = self.ratio

        # Получаем прямую ссылку на весь трехмерный массив геометрии коллекции.
        # Форма массива: (num_arrows * 5, 3, 3) -> [индекс_пути, точка, координата_XYZ]
        segs = self._segments3d

        cos_a = self._cos_angles
        sin_a = self._sin_angles

        for arrow_idx in range(num):
            i = arrow_idx * step

            x1, y1, z1 = ox[i], oy[i], oz[i]
            x2, y2, z2 = x1 + vx[i], y1 + vy[i], z1 + vz[i]

            dx, dy, dz = vx[i], vy[i], vz[i]
            length = np.sqrt(dx**2 + dy**2 + dz**2)

            base_seg_idx = arrow_idx * 5

            if length < 1e-5:
                # Быстрое обнуление всех 5 сегментов (древка и 4 зонтиков)
                segs[base_seg_idx : base_seg_idx + 5, :, 0] = x1
                segs[base_seg_idx : base_seg_idx + 5, :, 1] = y1
                segs[base_seg_idx : base_seg_idx + 5, :, 2] = z1
                continue

            # --- 1. ОБНОВЛЯЕМ ЛИНИЮ-ОСНОВУ (Путь состоит из 3 точек, 3-я дублирует 2-ю) ---
            # Индексы: [номер_пути, номер_точки]
            segs[base_seg_idx, 0, 0] = x1
            segs[base_seg_idx, 0, 1] = y1
            segs[base_seg_idx, 0, 2] = z1  # Хвост
            segs[base_seg_idx, 1, 0] = x2
            segs[base_seg_idx, 1, 1] = y2
            segs[base_seg_idx, 1, 2] = z2  # Остриё
            segs[base_seg_idx, 2, 0] = x2
            segs[base_seg_idx, 2, 1] = y2
            segs[base_seg_idx, 2, 2] = z2  # Дубль

            # --- 2. СТРОИМ БАЗИС ДЛЯ ЗОНТИКА ---
            cx = x2 - ratio * dx
            cy = y2 - ratio * dy
            cz = z2 - ratio * dz

            head_len = ratio * length
            r = head_len * 0.35

            if abs(dx) < abs(dz):
                ux, uy, uz = 1.0, 0.0, 0.0
            else:
                ux, uy, uz = 0.0, 1.0, 0.0

            nx1 = dy * uz - dz * uy
            ny1 = dz * ux - dx * uz
            nz1 = dx * uy - dy * ux
            n1_len = np.sqrt(nx1**2 + ny1**2 + nz1**2)
            nx1 /= n1_len
            ny1 /= n1_len
            nz1 /= n1_len

            nx2 = (dy * nz1 - dz * ny1) / length
            ny2 = (dz * nx1 - dx * nz1) / length
            nz2 = (dx * ny1 - dy * nx1) / length

            # --- 3. ОБНОВЛЯЕМ 4 ПОПАРНЫХ ПУТИ ЗОНТИКА (по 3 точки каждый) ---
            for s in range(4):
                idx = base_seg_idx + 1 + s

                offset_x = r * (cos_a[s] * nx1 + sin_a[s] * nx2)
                offset_y = r * (cos_a[s] * ny1 + sin_a[s] * ny2)
                offset_z = r * (cos_a[s] * nz1 + sin_a[s] * nz2)

                # Записываем in-place 3D геометрию напрямую в массив [Крыло А -> Остриё -> Крыло Б]
                segs[idx, 0, 0] = cx + offset_x
                segs[idx, 0, 1] = cy + offset_y
                segs[idx, 0, 2] = cz + offset_z
                segs[idx, 1, 0] = x2
                segs[idx, 1, 1] = y2
                segs[idx, 1, 2] = z2
                segs[idx, 2, 0] = cx - offset_x
                segs[idx, 2, 1] = cy - offset_y
                segs[idx, 2, 2] = cz - offset_z

        # Уведомляем систему о необходимости перерисовки
        self.stale = True
