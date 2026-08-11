import numpy as np
from matplotlib.axes import Axes

class Torus():
    def __init__(self, R: float, r: float):
        self.R = R
        self.r = r

    def plot(self, ax: Axes, num_uv: int, color='gray', alpha=0.25, linewidth=0.5):
        R = self.R
        r= self.r

        uv = np.linspace(0, 2 * np.pi, num_uv)
        u, v = np.meshgrid(uv, uv)
        xy = R + r * np.cos(u)
        X_mesh = xy * np.cos(v)
        Y_mesh = xy * np.sin(v)
        Z_mesh = r * np.sin(u)

        ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color=color, alpha=alpha, linewidth=linewidth)
