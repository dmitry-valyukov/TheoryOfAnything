import numpy as np
from matplotlib.axes import Axes


class Torus:
    """Тор с большим радиусом R и радиусом трубки r."""

    def __init__(self, R: float, r: float):
        self.R = float(R)
        self.r = float(r)

    def mesh(self, num_uv: int):
        """Сетка (X, Y, Z) поверхности тора: u — угол трубки, v — обход."""
        uv = np.linspace(0, 2 * np.pi, num_uv)
        u, v = np.meshgrid(uv, uv)
        xy = self.R + self.r * np.cos(u)
        return xy * np.cos(v), xy * np.sin(v), self.r * np.sin(u)

    def plot(self, ax: Axes, num_uv: int, color='gray', alpha=0.25,
             linewidth=0.5):
        """Рисует каркас тора и возвращает созданный artist."""
        X, Y, Z = self.mesh(num_uv)
        return ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha,
                                 linewidth=linewidth)
