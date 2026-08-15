"""render_figures.py — статические иллюстрации к тексту теории.

Рендерит PNG-кадры из тех же параметризаций, что используют интерактивные
симуляции (Electron.py, SpinorSim.py, Proton.py), плюс схему геометрии
адрона к подглаве 5.9. Файлы кладутся в theory/images/.

Зачем отдельный скрипт, а не скриншоты: картинки в документации должны
пересобираться командой, а не восстанавливаться вручную. Формулы здесь
те же, что в симуляциях; расхождение — только в оформлении (статичный
кадр, светлая тема, подписи).

ВАЖНО О СТАТУСЕ КАРТИНОК. Это визуализация ПРИНЯТОГО В МОДЕЛИ АНЗАЦА,
а не решение её уравнений. Кривая рисуется по явной параметризации
(намотка phi = theta/2 и т. д.), а не получается интегрированием
уравнения Ламе — Навье. Иллюстрация показывает, КАК В МОДЕЛИ ПРИНЯТО
представлять солитон, и не является доказательством, что такая
конфигурация устойчива.

Запуск:
    python render_figures.py [--out ../../theory/images]
"""

from __future__ import annotations

# метаданные для панели запуска (SimPanel.py)
PANEL = {"group": 10, "kind": "рендер PNG"}

import argparse
import os

import matplotlib
matplotlib.use("Agg")                      # без экрана

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

DPI = 150
FG = "#111111"
BG = "#ffffff"


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------

def torus_mesh(R: float, r: float, n: int = 32):
    uv = np.linspace(0.0, 2.0 * np.pi, n)
    u, v = np.meshgrid(uv, uv)
    xy = R + r * np.cos(u)
    return xy * np.cos(v), xy * np.sin(v), r * np.sin(u)


def new_3d(title: str, size: float, elev: float = 30.0, azim: float = -60.0):
    fig = plt.figure(figsize=(7.2, 6.4), facecolor=BG)
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.94), projection="3d", facecolor=BG)
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    fig.suptitle(title, color=FG, fontsize=13, y=0.985)
    return fig, ax


def ribbon(ax, x, y, z, values, cmap="hsv", lw=3.0, alpha=0.85):
    pts = np.stack((x, y, z), axis=1).reshape(-1, 1, 3)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=cmap, linewidths=lw, alpha=alpha)
    lc.set_array(values[:-1])
    lc.set_clim(0.0, 1.0)
    ax.add_collection3d(lc)
    return lc


def save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Электрон: самосопряжённый рог-тор, замыкание за 4 pi
# ---------------------------------------------------------------------------

def figure_electron(out_dir: str) -> str:
    R = r = 1.0                       # R = r: рог-тор, горловина вырождена в точку
    n = 900
    theta = np.linspace(0.0, 4.0 * np.pi, n)
    phi = 0.5 * theta                 # мёбиусовский шаг: спин 1/2

    fig, ax = new_3d("Электрон: самосопряжённый рог-тор, замыкание за $4\\pi$",
                     size=2.05, elev=28, azim=-58)
    X, Y, Z = torus_mesh(R, r, 30)
    ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.20, linewidth=0.4)

    # лента из нескольких смещённых по фазе нитей
    for off in np.linspace(-0.10, 0.10, 5):
        p = phi + off
        rad = R + r * np.cos(p)
        ribbon(ax, rad * np.cos(theta), rad * np.sin(theta), r * np.sin(p),
               (theta / (4.0 * np.pi)) % 1.0, lw=2.6, alpha=0.55)

    # репер Картана
    q = np.arange(0, n, 40)
    pq = phi[q]
    rq = R + r * np.cos(pq)
    ax.quiver(rq * np.cos(theta[q]), rq * np.sin(theta[q]), r * np.sin(pq),
              np.cos(pq) * np.cos(theta[q]), np.cos(pq) * np.sin(theta[q]),
              np.sin(pq), length=0.30, color="#d4a017", linewidth=1.0,
              normalize=True)
    return save(fig, out_dir, "electron-horn-torus.png")


# ---------------------------------------------------------------------------
# 2. Электрон и позитрон: инверсия знака намотки
# ---------------------------------------------------------------------------

def figure_pair(out_dir: str) -> str:
    R = r = 1.0
    n = 700
    theta = np.linspace(0.0, 4.0 * np.pi, n)
    sep = 2.6

    fig, ax = new_3d("Электрон и позитрон: один и тот же тор, "
                     "противоположный знак намотки",
                     size=4.0, elev=24, azim=-62)

    for sign, dx, cmap in ((+1.0, -sep, "hsv"), (-1.0, +sep, "hsv")):
        X, Y, Z = torus_mesh(R, r, 26)
        ax.plot_wireframe(X + dx, Y, Z, color="gray", alpha=0.18, linewidth=0.35)
        phi = sign * 0.5 * theta
        rad = R + r * np.cos(phi)
        vals = (theta / (4.0 * np.pi)) % 1.0
        if sign < 0:
            vals = 1.0 - vals                    # обратный обход цвета
        ribbon(ax, rad * np.cos(theta) + dx, rad * np.sin(theta), r * np.sin(phi),
               vals, lw=2.8, alpha=0.85, cmap=cmap)

    ax.text(-sep, 0, 2.0, "$e^-$", color=FG, fontsize=15, ha="center")
    ax.text(+sep, 0, 2.0, "$e^+$", color=FG, fontsize=15, ha="center")
    return save(fig, out_dir, "electron-positron.png")


# ---------------------------------------------------------------------------
# 3. Протон: дикварковый керн uu и d-маятник
# ---------------------------------------------------------------------------

def figure_proton(out_dir: str) -> str:
    """Протон: одна нить, три фокуса, разнесённые по ТОРОИДАЛЬНОМУ углу.

    Две поправки против наивного рисунка (разбор — подглава 5.10):

    1. Постоянный сдвиг ФАЗЫ на pi вырожден: при намотке phi = theta/2 он
       эквивалентен сдвигу theta на 2pi и даёт ТУ ЖЕ замкнутую кривую.
       Поэтому фокусы разносятся по theta (на 2pi/3), а не по phi.

    2. Видимые «две зацепленные намотки» — это одна нить: замыкание
       наступает лишь при 4pi, и кривая выглядит как два вложенных кольца,
       соединённых через горловину. Их зацепление и есть хопфовское
       зацепление одного слоя, несущее топологический заряд.
    """
    r = 1.0
    R = np.sqrt(3.0) * r
    n = 1400
    theta = np.linspace(0.0, 4.0 * np.pi, n)

    fig, ax = new_3d("Протон: одна нить, три фокуса через $2\\pi/3$ по $\\theta$",
                     size=2.9, elev=26, azim=-58)
    X, Y, Z = torus_mesh(R, r, 30)
    ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.16, linewidth=0.4)

    # --- единая нить, окрашенная по параметру: видно, что колец два, кривая одна
    phi = 0.5 * theta
    rad = R + r * np.cos(phi)
    ribbon(ax, rad * np.cos(theta), rad * np.sin(theta), r * np.sin(phi),
           theta / (4.0 * np.pi), cmap="viridis", lw=2.4, alpha=0.9)

    # --- три фокуса, разнесённые по тороидальному углу на 2pi/3
    foci = []
    for t0, lbl, col in ((0.0, "$u_1$", "#c0392b"),
                         (2.0 * np.pi / 3.0, "$u_2$", "#8e44ad"),
                         (4.0 * np.pi / 3.0, "$d$", "#2ca02c")):
        p = 0.5 * t0
        rr = R + r * np.cos(p)
        f = (rr * np.cos(t0), rr * np.sin(t0), r * np.sin(p))
        foci.append(f)
        ax.scatter([f[0]], [f[1]], [f[2]], s=180, color=col, depthshade=False,
                   edgecolors="white", linewidths=1.3, zorder=8)
        ax.text(f[0] * 1.12, f[1] * 1.12, f[2] + 0.5, lbl, color=col,
                fontsize=14, ha="center")

    # --- рёбра треугольника с подписью длин: он НЕ равносторонний
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = foci[i], foci[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#e45756", lw=1.6, ls="--", alpha=0.9)
        d = np.linalg.norm(np.array(a) - np.array(b))
        m = [(a[k] + b[k]) / 2.0 for k in range(3)]
        ax.text(m[0], m[1], m[2] + 0.22, f"{d:.2f}$r$", color="#e45756",
                fontsize=10, ha="center")

    handles = [plt.Line2D([], [], color="#3b528b", lw=3,
                          label="одна нить, замыкание за $4\\pi$"),
               plt.Line2D([], [], color="#e45756", lw=2, ls="--",
                          label="треугольник фокусов (не равносторонний)")]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              labelcolor=FG, fontsize=10)
    return save(fig, out_dir, "proton-modes.png")


# ---------------------------------------------------------------------------
# 4. Геометрия адрона: расстояние между кварками против зарядового радиуса
# ---------------------------------------------------------------------------

def figure_hadron_geometry(out_dir: str) -> str:
    L = 0.8809          # фм, сторона треугольника (5.9)
    Rq = L / np.sqrt(3.0)
    rp = 0.8409         # фм, зарядовый радиус, CODATA

    fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")

    ang = np.array([90.0, 210.0, 330.0]) * np.pi / 180.0
    px, py = Rq * np.cos(ang), Rq * np.sin(ang)

    t = np.linspace(0, 2 * np.pi, 400)
    ax.fill(rp * np.cos(t), rp * np.sin(t), color="#4c78a8", alpha=0.10)
    ax.plot(rp * np.cos(t), rp * np.sin(t), color="#4c78a8", lw=1.6,
            label=f"зарядовый радиус $r_p = {rp}$ фм (опыт)")
    ax.plot(Rq * np.cos(t), Rq * np.sin(t), color="#999999", lw=0.9, ls=":",
            label=f"радиус триады $R_q = L/\\sqrt{{3}} = {Rq:.3f}$ фм")

    for i in range(3):
        j = (i + 1) % 3
        ax.plot([px[i], px[j]], [py[i], py[j]], color="#e45756", lw=2.2,
                zorder=3)
        ax.plot([0, px[i]], [0, py[i]], color="#999999", lw=0.9, ls="--",
                zorder=2)
    ax.scatter(px, py, s=150, color="#e45756", zorder=4)
    ax.scatter([0], [0], s=40, color="#333333", zorder=4)

    mx, my = (px[0] + px[1]) / 2.0, (py[0] + py[1]) / 2.0
    ax.annotate(f"$L = \\frac{{4\\pi}}{{3}}\\bar\\lambda_p = {L:.3f}$ фм",
                xy=(mx, my), xytext=(-1.00, 0.72),
                arrowprops=dict(arrowstyle="-", color="#e45756", lw=0.9),
                color="#e45756", fontsize=12)

    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("фм", color=FG)
    ax.set_ylabel("фм", color=FG)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=FG)
    ax.set_title("Геометрия протона: расстояние между кварками\n"
                 "не совпадает с зарядовым радиусом", color=FG, fontsize=13)
    fig.tight_layout()
    return save(fig, out_dir, "hadron-geometry.png")


# ---------------------------------------------------------------------------
# 5. Фотон: бегущая винтовая фаза в пакете
# ---------------------------------------------------------------------------

def figure_photon(out_dir: str) -> str:
    n = 1400
    x = np.linspace(-6.0, 6.0, n)
    k = 3.2
    env = np.exp(-(x ** 2) / (2.0 * 1.9 ** 2))
    ey = env * np.cos(k * x)
    bz = env * np.sin(k * x)

    fig = plt.figure(figsize=(7.6, 4.6), facecolor=BG)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.9), projection="3d", facecolor=BG)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-72)
    ax.set_box_aspect((3.0, 1.0, 1.0), zoom=1.35)

    ax.plot(x, np.zeros_like(x), np.zeros_like(x), color="#999999", lw=0.8,
            ls="--")
    ax.plot(x, ey, np.zeros_like(x), color="#d62728", lw=1.8, label="$E$")
    ax.plot(x, np.zeros_like(x), bz, color="#1f77b4", lw=1.8, label="$B$")
    ax.plot(x, env, np.zeros_like(x), color="#bbbbbb", lw=0.8)
    ax.plot(x, -env, np.zeros_like(x), color="#bbbbbb", lw=0.8)

    q = np.arange(0, n, 70)
    ax.quiver(x[q], np.zeros_like(x[q]), np.zeros_like(x[q]),
              np.zeros_like(x[q]), ey[q], np.zeros_like(x[q]),
              color="#d62728", linewidth=0.9, arrow_length_ratio=0.12)
    ax.quiver(x[q], np.zeros_like(x[q]), np.zeros_like(x[q]),
              np.zeros_like(x[q]), np.zeros_like(x[q]), bz[q],
              color="#1f77b4", linewidth=0.9, arrow_length_ratio=0.12)

    fig.suptitle("Фотон: поперечный пакет сдвиговой ветви, "
                 "$E \\perp B \\perp k$", color=FG, fontsize=13, y=0.97)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=11)
    return save(fig, out_dir, "photon-packet.png")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Дефицит обхода: заряд как доля пройденного спинорного круга
# ---------------------------------------------------------------------------

def _circuit_panel(ax, title, modes, total_label):
    """Одна панель: полный спинорный обход 4pi как круг, доли мод — дуги."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.75, 1.55)

    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="#cccccc", lw=1.2)
    ax.plot([0, 0], [0.94, 1.06], color="#888888", lw=1.2)
    ax.text(0, 1.16, "$0 = 4\\pi$", color=FG, fontsize=10, ha="center")

    radii = (0.88, 0.66, 0.44)
    # метки разносим по углу, чтобы не наезжали друг на друга
    label_at = (0.75, 0.45, 0.62)
    for (frac, lbl, col), rad, w in zip(modes, radii, label_at):
        a = np.linspace(0.0, 2 * np.pi * frac, 300) + np.pi / 2.0
        ax.plot(rad * np.cos(a), rad * np.sin(a), color=col, lw=5.0,
                solid_capstyle="butt", alpha=0.9)
        mid = np.pi / 2.0 + 2 * np.pi * frac * w
        ax.text(rad * np.cos(mid), rad * np.sin(mid), lbl,
                color="white", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"))

    ax.text(0, -1.42, total_label, color=FG, fontsize=12, ha="center")
    ax.set_title(title, color=FG, fontsize=13, pad=6)


def figure_charge_circuit(out_dir: str) -> str:
    """Заряд как доля полного спинорного обхода 4pi, пройденная модой.

    Одно ребро цветового контура есть 4pi/3, то есть треть полного обхода
    (5.9). Мода в двух плоскостях (u) проходит две трети, мода в одной (d) —
    одну треть, и в обратную сторону. Сумма по адрону обязана быть целой:
    именно это и есть условие замыкания.
    """
    U, D = "#c0392b", "#2ca02c"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    _circuit_panel(
        axes[0], "Протон $uud$",
        [(2 / 3, "$u$  $+2/3$", U), (2 / 3, "$u$  $+2/3$", U),
         (-1 / 3, "$d$  $-1/3$", D)],
        "сумма $= +2/3 + 2/3 - 1/3 = \\mathbf{+1}$  — обход замкнут")

    _circuit_panel(
        axes[1], "Нейтрон $udd$",
        [(2 / 3, "$u$  $+2/3$", U), (-1 / 3, "$d$  $-1/3$", D),
         (-1 / 3, "$d$  $-1/3$", D)],
        "сумма $= +2/3 - 1/3 - 1/3 = \\mathbf{0}$  — обход замкнут")

    fig.suptitle("Дробный заряд как дефицит обхода: доля круга $4\\pi$, "
                 "пройденная модой за цветовой оборот",
                 color=FG, fontsize=13, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, out_dir, "charge-circuit.png")


FIGURES = (figure_electron, figure_pair, figure_proton,
           figure_hadron_geometry, figure_charge_circuit, figure_photon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("..", "..", "theory", "images"))
    args = ap.parse_args()
    print("Рендер иллюстраций к тексту теории")
    for fn in FIGURES:
        print("  ", fn(args.out))
    print("Готово. Статус картинок: визуализация анзаца, а не решение уравнений.")


if __name__ == "__main__":
    main()
