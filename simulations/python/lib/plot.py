"""Утилиты 3D-сцены для симуляций: инициализация осей, цветные полилинии и стрелки.

Бэкенд выбирается в lib/backend.py (QtAgg -> TkAgg -> Agg) до импорта pyplot.
"""

from typing import Callable, Iterable, Optional

import numpy as np

from . import backend  # noqa: F401  - должен импортироваться ДО pyplot

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = None
ani = None

SIZE = 8.0  # базовый размер фигуры (обновляется в init())

#: цвета текущей темы; выставляются в init(), используются скриптами
FG_COLOR = 'white'
BG_COLOR = 'black'


def init(title: str,
         zoom: float = 1.0,
         elev: float = 30.0,
         azim: float = -45.0,
         size: float = SIZE,
         theme: str = 'Dark') -> Axes:
    """Создаёт фигуру и 3D-оси во весь холст.

    zoom / elev / azim реально применяются (раньше игнорировались).
    size пишется в модульную SIZE, от которой скрипты строят геометрию.
    """
    global fig, SIZE, FG_COLOR, BG_COLOR

    SIZE = float(size)

    if theme == 'Dark':
        FG_COLOR, BG_COLOR = 'white', 'black'
    else:
        FG_COLOR, BG_COLOR = 'black', 'white'

    fig = plt.figure(figsize=(SIZE, SIZE), facecolor=BG_COLOR)
    ax = fig.add_axes((0, 0, 1, 1), projection='3d', facecolor=BG_COLOR)

    ax.set_xlim(-SIZE, SIZE)
    ax.set_ylim(-SIZE, SIZE)
    ax.set_zlim(-SIZE, SIZE)

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # цвет темы — на случай, если оси включат обратно
    ax.tick_params(colors=FG_COLOR)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(FG_COLOR)
    ax.axis('off')

    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)

    return ax


def animate(func: Callable[[int], object],
            frames: Optional[Iterable | int | Callable],
            interval: float,
            **kwargs):
    """Запускает FuncAnimation на фигуре, созданной init()."""
    global ani
    if fig is None:
        raise RuntimeError("plot.animate(): сначала вызовите plot.init()")
    ani = animation.FuncAnimation(fig, func, frames=frames,
                                  interval=interval, blit=False, **kwargs)
    plt.show()
    return ani


RgbaColor = tuple[float, float, float, float]


def rgba(color: ColorType, alpha: float = 1.0) -> RgbaColor:
    return colors.to_rgba(color, alpha=alpha)


# gist_rainbow, rainbow, hsv (циклическая), twilight, twilight_shifted.
# jet, turbo, nipy_spectral.
class Segments(Line3DCollection):
    """Полилиния, раскрашенная по циклической фазе в [0, 1)."""

    def __init__(self, ax: Axes, cmap='hsv', linewidths=3.5, alpha=0.9,
                 clim=(0.0, 1.0)):
        dummy = np.zeros((1, 2, 3))
        super().__init__(dummy, cmap=plt.get_cmap(cmap),
                         linewidths=linewidths, alpha=alpha)
        # фаза всегда нормирована на [0,1); без явного clim matplotlib
        # намертво фиксирует шкалу по диапазону ПЕРВОГО кадра
        self.set_clim(*clim)
        ax.add_collection3d(self)


class Arrows(Line3DCollection):
    """Пачка 3D-стрелок в одной коллекции: 5 путей на стрелку (древко + 4 луча).

    Геометрия пересчитывается in-place векторно, объект не пересоздаётся.
    """

    def __init__(self, ax: Axes, num_arrows: int, color: RgbaColor,
                 arrow_length_ratio: float = 0.15):
        self.num_arrows = int(num_arrows)
        self.ratio = float(arrow_length_ratio)

        # (num_arrows, 5, 3, 3) -> [стрелка, путь, точка, XYZ].
        # Все пути по 3 точки: у древка третья дублирует вторую.
        flat = np.zeros((self.num_arrows * 5, 3, 3))
        super().__init__(flat, colors=color)

        # Держим ссылку именно на тот буфер, который читает matplotlib,
        # иначе запись in-place не дойдёт до отрисовки.
        stored = getattr(self, '_segments3d', None)
        if stored is None or np.asarray(stored).shape != flat.shape:
            self.set_segments(flat)
            stored = self._segments3d
        self._geom = np.asarray(stored).reshape(self.num_arrows, 5, 3, 3)

        # Углы 4 лучей "зонтика"
        s = np.sqrt(0.5)
        self._cos_angles = np.array([1.0, s, 0.0, -s])
        self._sin_angles = np.array([0.0, s, 1.0, s])

        self._eye = np.eye(3)
        self._idx = None       # кэш индексов выборки
        self._idx_step = None

        ax.add_collection3d(self)

    def _indices(self, step: int, n_available: int) -> np.ndarray:
        if self._idx is None or self._idx_step != step:
            idx = np.arange(self.num_arrows) * step
            if self.num_arrows and idx[-1] >= n_available:
                raise IndexError(
                    f"Arrows: нужно {idx[-1] + 1} точек при step={step}, "
                    f"а передано {n_available}")
            self._idx, self._idx_step = idx, step
        return self._idx

    def recalc(self, ox, oy, oz, vx, vy, vz, step: int = 1):
        ox, oy, oz = np.asarray(ox), np.asarray(oy), np.asarray(oz)
        vx, vy, vz = np.asarray(vx), np.asarray(vy), np.asarray(vz)
        i = self._indices(step, len(ox))

        origin = np.stack((ox[i], oy[i], oz[i]), axis=1)   # (N,3)
        vec = np.stack((vx[i], vy[i], vz[i]), axis=1)      # (N,3)
        tip = origin + vec

        length = np.linalg.norm(vec, axis=1)               # (N,)
        live = length > 1e-5
        safe = np.where(live, length, 1.0)

        # Опорная ось — координата с НАИМЕНЬШИМ модулем: она гарантированно
        # не параллельна vec. Прежний вариант (сравнение |dx| и |dz|) давал
        # нулевое векторное произведение и nan для векторов вдоль оси Y.
        helper = self._eye[np.argmin(np.abs(vec), axis=1)]  # (N,3)

        n1 = np.cross(vec, helper)
        n1_len = np.linalg.norm(n1, axis=1)
        n1 = n1 / np.where(n1_len > 0, n1_len, 1.0)[:, None]
        n2 = np.cross(vec, n1) / safe[:, None]

        centre = tip - self.ratio * vec
        r = (self.ratio * 0.35) * safe

        # (N,4,3): смещения концов лучей зонтика
        offs = r[:, None, None] * (
            self._cos_angles[None, :, None] * n1[:, None, :]
            + self._sin_angles[None, :, None] * n2[:, None, :])

        g = self._geom
        g[:, 0, 0] = origin
        g[:, 0, 1] = tip
        g[:, 0, 2] = tip
        g[:, 1:, 0] = centre[:, None, :] + offs
        g[:, 1:, 1] = tip[:, None, :]
        g[:, 1:, 2] = centre[:, None, :] - offs

        # Вырожденные стрелки схлопываем в точку начала
        if not live.all():
            dead = ~live
            g[dead] = origin[dead][:, None, None, :]

        self.stale = True
