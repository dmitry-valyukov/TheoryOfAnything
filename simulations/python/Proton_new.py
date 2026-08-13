"""Протон в текущей редакции модели: ОДНА намотка, три фокуса, счётчик обхода.

Отличия от `Proton.py`, каждое соответствует конкретному разделу текста:

1. **Одна нить, а не три.** Барион есть триада фокусов интерференции одной
   конструкции (5.6). Три кривых `Proton.py` вырождены: постоянный сдвиг фазы
   на pi переводит мёбиусову намотку саму в себя, `C(th; phi+pi) = C(th+2pi; phi)`
   с точностью 1e-15 (5.10.1).

2. **Фаза и положение разведены.** Нужны ТРИ независимые переменные (5.10.4):
       th        — положение фокуса вдоль нити (тороидальный угол),
       phi = th/2 — намотка (полоидальный угол, мёбиусов шаг),
       psi       — фаза возбуждения.
   Принцип Паули требует `d(psi)_uu = pi` — это фазовый, а НЕ геометрический
   развод. Геометрия разносит фокусы по `d(th) = 2pi/3`.
   В `Proton.py` psi кодировалась как сдвиг phi — отсюда вырождение.

3. **Отдача дифференциальная, а не синфазная** (5.10.3, T-33). d-маятник в
   каждый момент ближе к одному из u, поэтому поправки к двум u имеют
   ПРОТИВОПОЛОЖНЫЕ знаки. Синфазная отдача `Proton.py` не снимает вырождения
   ни при какой амплитуде.

4. **Треугольник фокусов не равносторонний** (5.10.5). При `R = sqrt(3) r`
   стороны 4.393 / 3.619 / 3.041 r, разброс 36.7%; расстояния фокусов от оси
   тора различаются в 2.2 раза. Рисунок это показывает, а не скрывает.

5. **Дробный заряд — счётчик доли обхода, а не третья кривая** (5.11.5).
   За один цветовой оборот u проходит 2/3 полного спинорного обхода 4pi,
   d — 1/3 в обратную сторону. Сумма трёх дуг замыкается ровно в круг: +1.

Запуск:
    python Proton_new.py            # анимация
    python Proton_new.py --check    # только численные проверки, без окна
    python Proton_new.py --save out.png
"""

from __future__ import annotations

import argparse
import itertools
import math
from fractions import Fraction

import lib.backend  # noqa: F401  - выбор бэкенда до импорта pyplot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.modes import to_segments
from lib.torus import Torus

# ---------------------------------------------------------------------------
# Параметры модели
# ---------------------------------------------------------------------------

R_OVER_r = math.sqrt(3.0)     # принято в модели (5.6)
r = 1.0
R = R_OVER_r * r

N_POINTS = 1200
N_FRAMES = 360

#: базовые положения фокусов вдоль нити: равный шаг 2pi/3 (5.10.4)
THETA_0 = {"u1": 0.0, "d": 2.0 * math.pi / 3.0, "u2": 4.0 * math.pi / 3.0}

#: фазы возбуждения: развод Паули между двумя u равен pi (5.10.4)
PSI_0 = {"u1": 0.0, "u2": math.pi, "d": 0.0}

#: omega_d = omega_u / 2 — та же гипотеза, что и «d занимает одну плоскость» (5.11.2)
OMEGA = {"u1": 1.0, "u2": 1.0, "d": 0.5}

#: доля полного спинорного обхода 4pi за один цветовой оборот (5.11.1-5.11.2)
LEG = Fraction(1, 3)
ARC = {"u1": 2 * LEG, "u2": 2 * LEG, "d": -LEG}

#: начальные цветовые фазы: три цвета, разнесённые на 1/3 радужного круга.
#: Одно ребро цветового контура есть d(phi)/4pi = 1/3 полного обхода (5.11.1).
COLOUR_0 = {"u1": Fraction(0), "d": Fraction(1, 3), "u2": Fraction(2, 3)}

#: набег внутренней фазы за один цветовой оборот. Из T_d = T_цв = 4pi/omega
#: (5.9.5) следует omega * T_цв = 4pi — ровно два полных цикла 2pi.
#: Величина выведена, а не подобрана.
PHASE_PER_TURN = 4.0 * math.pi

#: сколько цветовых оборотов проигрывает анимация. Каждый фокус возвращает
#: свой исходный цвет ровно через 3 оборота (проверка в check(), пункт 6).
TURNS = 3

#: ВНИМАНИЕ: обе амплитуды ниже — иллюстративные. Модель фиксирует положение
#: равновесия d(theta) = 2pi/3 (условие замыкания цветового контура, 5.9), но
#: НЕ даёт амплитуды колебаний около него. Значения подобраны для читаемости
#: картинки и физического содержания не несут.
AMP_D = math.pi / 8.0           # амплитуда качания d-маятника по theta
#: амплитуда дифференциальной отдачи двух u; знаки противоположны (5.10.3).
#: При EPS < AMP_D зазоры d-u1 и u2-d ходят в противофазе — это и есть признак
#: того, что отдача дифференциальная, а не синфазная.
EPS_RECOIL = 0.10

#: экспериментальные и модельные величины для перевода в фемтометры (5.9, 5.10.6)
HBAR_C = 197.3269804            # МэВ*фм
M_PROTON = 938.27208816         # МэВ
R_PROTON_EXP = 0.8409           # фм, зарядовый радиус

COLORS = {"u1": "#ff5f4d", "u2": "#4da3ff", "d": "#8bd44a"}
LABELS = {"u1": "u₁", "u2": "u₂", "d": "d"}


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def wound_curve(theta, phase_shift: float = 0.0):
    """Точка (или массив точек) намотанной нити: phi = theta/2 + phase_shift."""
    theta = np.asarray(theta, dtype=float)
    phi = 0.5 * theta + phase_shift
    rad = R + r * np.cos(phi)
    return rad * np.cos(theta), rad * np.sin(theta), r * np.sin(phi)


def focus_thetas(t: float) -> dict:
    """
    Положения трёх фокусов в момент t с ДИФФЕРЕНЦИАЛЬНОЙ отдачей.

    d качается около 2pi/3; знак его смещения определяет, к какому u он ближе,
    поэтому поправки к u входят с противоположными знаками (5.10.3).
    """
    swing = math.sin(2.0 * t)
    return {
        "u1": THETA_0["u1"] - EPS_RECOIL * swing,
        "d": THETA_0["d"] + AMP_D * swing,
        "u2": THETA_0["u2"] + EPS_RECOIL * swing,
    }


#: сдвиг цветовой фазы за один цветовой оборот, общий для всех трёх фокусов.
#: На трёхсайтовом контуре +2 ребра вперёд и 1 ребро назад — ОДНО И ТО ЖЕ
#: смещение: +2/3 = -1/3 (mod 1). Поэтому триада движется жёстко.
COLOUR_STEP = Fraction(2, 3)


def focus_colour_phase(turn: float) -> dict:
    """
    Цветовая фаза каждого фокуса в долях радужного круга.

    Триада **жёсткая**: все три фазы сдвигаются с одной скоростью, взаимные
    зазоры тождественно равны 1/3 в любой момент. Иначе в промежуточные
    моменты два фокуса несли бы одинаковый цвет, и цветовой синглет нарушался
    бы между оборотами, а не только сохранялся на целых.

    Различие u и d — не в скорости, а в НАМОТКЕ: за оборот u проходит
    +2/3 круга вперёд, d — 1/3 назад, и это один и тот же конец пути
    (+2/3 = -1/3 mod 1). Знаковая разность путей и есть заряд (5.11).
    """
    return {k: float(COLOUR_0[k] + COLOUR_STEP * turn) % 1.0 for k in COLOUR_0}


def colour_trail(turn: float, k: str) -> tuple:
    """Знаковый путь набега фазы моды k за текущий оборот: (начало, протяжённость)."""
    done = math.floor(turn + 1e-12)
    frac = turn - done
    start = float(COLOUR_0[k] + COLOUR_STEP * done) % 1.0
    return start, float(ARC[k]) * frac


def colour_permutation(n: int) -> dict:
    """Точные цветовые фазы после n целых цветовых оборотов (дроби, без float)."""
    return {k: (COLOUR_0[k] + COLOUR_STEP * n) % 1 for k in COLOUR_0}


def focus_psis(t: float) -> dict:
    """Фазы возбуждения — независимая переменная, в геометрию НЕ входит."""
    return {k: PSI_0[k] + OMEGA[k] * 2.0 * t for k in PSI_0}


def triangle_metrics(thetas: dict) -> dict:
    """Стороны треугольника фокусов, расстояния от оси тора, разброс."""
    keys = ("u1", "d", "u2")
    pts = {k: np.array(wound_curve(thetas[k])).ravel() for k in keys}
    sides = {}
    for a, b in itertools.combinations(keys, 2):
        sides[f"{a}-{b}"] = float(np.linalg.norm(pts[a] - pts[b]))
    vals = list(sides.values())
    axis = {k: float(math.hypot(pts[k][0], pts[k][1])) for k in keys}
    return {
        "points": pts,
        "sides": sides,
        "mean_edge": sum(vals) / 3.0,
        "spread": (max(vals) - min(vals)) / (sum(vals) / 3.0),
        "ratio": max(vals) / min(vals),
        "axis": axis,
        "axis_ratio": max(axis.values()) / min(axis.values()),
    }


def fm_per_r(thetas: dict) -> float:
    """
    Масштаб: среднее ребро треугольника отождествляется с L = (4pi/3)*lambda_p,
    как в 5.9 с поправкой 5.10.6 («L есть среднее ребро, а не сторона
    равностороннего треугольника»).
    """
    lambda_bar_p = HBAR_C / M_PROTON
    L_fm = 4.0 * math.pi / 3.0 * lambda_bar_p
    return L_fm / triangle_metrics(thetas)["mean_edge"]


# ---------------------------------------------------------------------------
# Численные проверки
# ---------------------------------------------------------------------------


def check() -> None:
    line = "-" * 74
    print(line)
    print("ПРОТОН: ПРОВЕРКИ К ТЕКУЩЕЙ РЕДАКЦИИ МОДЕЛИ")
    print(line)

    print("\n1. Вырождение постоянного сдвига фазы (5.10.1)")
    th = np.linspace(0.0, 4.0 * math.pi, 40001)
    a = np.array(wound_curve(th, math.pi))
    b = np.array(wound_curve(th + 2.0 * math.pi, 0.0))
    worst = float(np.max(np.linalg.norm(a - b, axis=0)))
    print(f"   max |C(th; phi+pi) - C(th+2pi; phi)| = {worst:.3e}")
    print("   => сдвиг фазы на pi даёт ТУ ЖЕ кривую; двух u-нитей не существует")

    print("\n2. Отдача: синфазная против дифференциальной (5.10.3)")
    print(f"   {'eps':>6} {'синфазная':>14} {'дифференциальная':>20}")
    for eps in (0.0, 0.05, 0.10, 0.20, 0.30):
        same = float(np.max(np.linalg.norm(
            np.array(wound_curve(th, math.pi + eps))
            - np.array(wound_curve(th + 2 * math.pi, eps)), axis=0)))
        diff = float(np.max(np.linalg.norm(
            np.array(wound_curve(th, math.pi - eps))
            - np.array(wound_curve(th + 2 * math.pi, eps)), axis=0)))
        print(f"   {eps:>6.2f} {same:>14.3e} {diff:>20.4f}")
    print("   => в этом скрипте отдача дифференциальная (знаки противоположны)")

    print("\n3. Треугольник фокусов в покое (5.10.5)")
    m = triangle_metrics(THETA_0)
    scale = fm_per_r(THETA_0)
    for name, val in sorted(m["sides"].items(), key=lambda kv: -kv[1]):
        print(f"   сторона {name:<7} = {val:6.3f} r = {val * scale:6.4f} фм")
    print(f"   max/min = {m['ratio']:.3f}, разброс {100 * m['spread']:.1f} %")
    print(f"   среднее ребро = {m['mean_edge']:.3f} r "
          f"= {m['mean_edge'] * scale:.4f} фм  (= L из 5.9)")
    print("\n   расстояния фокусов ОТ ОСИ тора:")
    for k, val in sorted(m["axis"].items(), key=lambda kv: -kv[1]):
        print(f"   {LABELS[k]:<3} {val:6.3f} r = {val * scale:6.4f} фм")
    print(f"   отношение max/min = {m['axis_ratio']:.3f}")
    print(f"   все три внутри зарядового облака r_p = {R_PROTON_EXP} фм: "
          f"{max(m['axis'].values()) * scale < R_PROTON_EXP}")

    print("\n4. Фаза возбуждения не влияет на геометрию (5.10.4)")
    p = focus_psis(0.37)
    print(f"   psi(u1) - psi(u2) = {(p['u1'] - p['u2']) % (2 * math.pi):.6f} "
          f"(= pi = {math.pi:.6f}, условие Паули)")
    print("   psi входит только в яркость/размер маркера, в координаты — нет")

    print("\n5. Счётчик доли обхода и заряд (5.11)")
    total = sum(ARC.values())
    for k in ("u1", "u2", "d"):
        print(f"   {LABELS[k]:<3} дуга = {ARC[k]} * 4pi "
              f"= {float(ARC[k]) * 4 * math.pi:+.4f} рад")
    print(f"   сумма дуг = {total} полного обхода  =>  заряд протона = {total}")
    print(f"   замыкание точное: {total == Fraction(1)}")

    print("\n6. Цветовая фаза фокусов (5.11.5)")
    base = colour_permutation(0)
    print("   начальные цвета (в долях радужного круга): "
          + ", ".join(f"{LABELS[k]}={base[k]}" for k in ("u1", "d", "u2")))
    target = set(base.values())
    for n in range(1, 4):
        cur = colour_permutation(n)
        same_set = set(cur.values()) == target
        identity = cur == base
        print(f"   после {n} оборота(ов): "
              + ", ".join(f"{LABELS[k]}={cur[k]}" for k in ("u1", "d", "u2"))
              + f"   набор сохранён: {same_set}   тождество: {identity}")
    print("   +2/3 (u, вперёд) и -1/3 (d, назад) сравнимы по модулю 1: "
          f"{(Fraction(2,3) - (-Fraction(1,3))) % 1 == 0}")
    print("   => один и тот же конец пути, разная намотка; триада жёсткая")
    gaps = set()
    for n in range(0, 121):
        c = focus_colour_phase(n / 40.0)
        v = sorted(c.values())
        gaps |= {round((v[1] - v[0]) % 1.0, 9), round((v[2] - v[1]) % 1.0, 9),
                 round((v[0] - v[2]) % 1.0, 9)}
    print(f"   зазоры между цветами за 3 оборота (121 отсчёт): "
          f"{sorted(gaps)}  => всегда 1/3, совпадений нет")
    print("   => цветовой обмен есть циклическая перестановка триады:")
    print("      набор цветов инвариантен на каждом обороте (синглет),")
    print("      каждый фокус возвращает свой цвет через 3 оборота")

    print("\n7. Что бежит, а что стоит")
    print(f"   T_d = 2pi/omega_d = 4pi/omega,  T_цв = 3*d(phi)/omega = 4pi/omega")
    print(f"   => набег внутренней фазы за цветовой оборот = omega*T_цв = 4pi")
    print(f"      = {PHASE_PER_TURN / (2 * math.pi):.0f} полных цикла 2pi "
          "(в скрипте PHASE_PER_TURN)")
    print("   фаза бежит по нити; амплитуда и фокусы стоят — структура Q-шара")
    print("   f(r) e^{i omega t}: модуль статичен, фаза вращается (2.4.2, 2.5)")
    print("   орбитального вращения триады нет: спин 1/2 уже даёт замыкание 4pi,")
    print("   вращение фокусов добавило бы момент сверх наблюдаемого")

    print("\n" + line)


# ---------------------------------------------------------------------------
# Сцена
# ---------------------------------------------------------------------------


def build_figure():
    fig = plt.figure(figsize=(15.5, 9.5), facecolor="black")
    gs = fig.add_gridspec(3, 2, width_ratios=(1.30, 1.0),
                          height_ratios=(1.0, 1.05, 0.80),
                          left=0.02, right=0.97, top=0.91, bottom=0.06,
                          wspace=0.08, hspace=0.62)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="black")
    ax_arc = fig.add_subplot(gs[0, 1], facecolor="black")
    ax_col = fig.add_subplot(gs[1, 1], facecolor="black")
    ax_rec = fig.add_subplot(gs[2, 1], facecolor="black")

    fig.suptitle("Протон: одна намотка, три фокуса, замыкание обхода",
                 color="white", fontsize=15, y=0.965)

    # --- 3D ---
    ax3d.grid(False)
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.fill = False
    lim = R + r
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.axis("off")
    ax3d.set_box_aspect((1, 1, 1), zoom=1.45)
    ax3d.view_init(elev=42, azim=-52)
    Torus(R=R, r=r).plot(ax3d, 24, color="dimgray", alpha=0.16, linewidth=0.5)

    theta = np.linspace(0.0, 4.0 * math.pi, N_POINTS)
    x, y, z = wound_curve(theta)
    thread = Line3DCollection(to_segments(x, y, z),
                              cmap=plt.get_cmap("hsv"), lw=3.2, alpha=0.95)
    thread.set_clim(0.0, 1.0)
    thread.set_array(((0.5 * theta) % (2 * np.pi))[:-1] / (2 * np.pi))
    ax3d.add_collection3d(thread)
    thread_theta = theta

    # горловина: единственная точка минимального радиуса, theta = 2pi
    gx, gy, gz = wound_curve(2.0 * math.pi)
    ax3d.plot([float(gx)], [float(gy)], [float(gz)], marker="x", ms=7,
              color="#b9b9b4", mew=1.4)
    ax3d.text(float(gx), float(gy), float(gz) - 0.45 * r,
              "горловина, θ = 2π", color="#898781", fontsize=8.5)

    tri = ax3d.plot([], [], [], color="white", lw=1.1, ls="--", alpha=0.55)[0]
    # заливка маркера = цветовая фаза, обод = опознавательный цвет моды,
    # размер и яркость = фаза возбуждения psi. Три атрибута — три переменные.
    dots = {k: ax3d.plot([], [], [], marker="o", ls="none",
                         mec=COLORS[k], mew=2.2)[0]
            for k in COLORS}
    tags = {k: ax3d.text(0, 0, 0, LABELS[k], color=COLORS[k], fontsize=12,
                         weight="bold") for k in COLORS}
    note3d = ax3d.text2D(0.01, 0.02, "", transform=ax3d.transAxes,
                         color="#b9b9b4", fontsize=9, va="bottom")

    # --- счётчик обхода ---
    ax_arc.set_aspect("equal")
    ax_arc.set_xlim(-1.9, 1.9)
    ax_arc.set_ylim(-1.95, 1.15)
    ax_arc.axis("off")
    ax_arc.set_title("Счётчик доли обхода\n"
                     "снаружи — сумма встык · внутри — каждая мода отдельно",
                     color="white", fontsize=11, pad=4)
    ax_arc.add_patch(Circle((0, 0), 1.0, fill=False, ec="#4a4a46", lw=1.2))
    # цель: точка старта, куда обязана вернуться сумма трёх дуг
    ax_arc.plot([0.0], [1.0], marker="|", ms=14, color="white", mew=1.6)
    arc_patches = []
    for k in ("u1", "u2", "d"):
        p = Arc((0, 0), 2.0, 2.0, theta1=0.0, theta2=0.0,
                color=COLORS[k], lw=7.0, alpha=0.9)
        ax_arc.add_patch(p)
        arc_patches.append((k, p))
    # отдельные дуги каждой моды: наглядно, что ни одна не смыкается сама с собой
    solo = []
    for i, k in enumerate(("u1", "u2", "d")):
        rr = 0.30 + 0.13 * i
        ax_arc.add_patch(Circle((0, 0), rr, fill=False, ec="#333331", lw=0.9))
        p = Arc((0, 0), 2 * rr, 2 * rr, theta1=0.0, theta2=0.0,
                color=COLORS[k], lw=3.0, alpha=0.75)
        ax_arc.add_patch(p)
        solo.append((k, p))
    arc_head = ax_arc.plot([], [], marker="o", ls="none", color="white",
                           ms=7, mec="black")[0]
    arc_note = ax_arc.text(0, -1.55, "", color="#b9b9b4", fontsize=9.5,
                           ha="center")

    # --- цветовой круг ---
    ax_col.set_aspect("equal")
    ax_col.set_xlim(-2.15, 2.15)
    ax_col.set_ylim(-3.30, 1.60)
    ax_col.axis("off")
    ax_col.set_title("Цветовая фаза: триада жёсткая, зазоры всегда 1/3\n"
                     "дуга — знаковый путь за оборот: u вперёд 2/3, d назад 1/3",
                     color="white", fontsize=11, pad=4)
    # радужное кольцо
    wheel = np.linspace(0.0, 2.0 * np.pi, 361)
    seg = np.stack([np.stack([np.cos(wheel[:-1]), np.sin(wheel[:-1])], axis=1),
                    np.stack([np.cos(wheel[1:]), np.sin(wheel[1:])], axis=1)],
                   axis=1)
    ring = LineCollection(seg, cmap=plt.get_cmap("hsv"), linewidths=9.0,
                          alpha=0.85)
    ring.set_clim(0.0, 1.0)
    ring.set_array(wheel[:-1] / (2 * np.pi))
    ax_col.add_collection(ring)
    # метки трёх исходных цветов
    for k in ("u1", "d", "u2"):
        a = 2 * math.pi * float(COLOUR_0[k])
        ax_col.plot([1.15 * math.cos(a)], [1.15 * math.sin(a)],
                    marker=".", ms=5, color="#6a6a64")
    col_dots = {k: ax_col.plot([], [], marker="o", ls="none", ms=15,
                               mec=COLORS[k], mew=2.2)[0] for k in COLORS}
    col_tags = {k: ax_col.text(0, 0, LABELS[k], color=COLORS[k], fontsize=10,
                               weight="bold", ha="center", va="center")
                for k in COLORS}
    # след: пройденная за анимацию дуга цветовой фазы, снаружи кольца
    col_trail = {}
    for i, k in enumerate(("u1", "u2", "d")):
        dd = 2.34 + 0.15 * i
        col_trail[k] = ax_col.add_patch(
            Arc((0, 0), dd, dd, theta1=0.0, theta2=0.0,
                color=COLORS[k], lw=3.2, alpha=0.85))
    col_note = ax_col.text(0, -1.72, "", color="#b9b9b4", fontsize=9.5,
                           ha="center", va="top")

    # --- отдача ---
    ax_rec.set_facecolor("black")
    ax_rec.set_title("Дифференциальная отдача: зазоры ходят в противофазе\n"
                     "равновесие Δθ = 2π/3 фиксировано моделью, амплитуды — нет",
                     color="white", fontsize=10.5, pad=4)
    ts = np.linspace(0.0, 2.0 * math.pi, 400)
    base = 2.0 * math.pi / 3.0
    gap1 = np.array([focus_thetas(t)["d"] - focus_thetas(t)["u1"] for t in ts])
    gap2 = np.array([focus_thetas(t)["u2"] - focus_thetas(t)["d"] for t in ts])
    ax_rec.axhline(base, color="#4a4a46", lw=1.0, ls=":")
    ax_rec.plot(ts, gap1, color=COLORS["u1"], lw=2.0, label="d − u₁")
    ax_rec.plot(ts, gap2, color=COLORS["u2"], lw=2.0, label="u₂ − d")
    marker = ax_rec.plot([], [], marker="o", ls="none", color="white", ms=6)[0]
    ax_rec.set_xlim(0, 2 * math.pi)
    ax_rec.set_xlabel("время цикла   (синфазная отдача Proton.py дала бы "
                      "вырождение 3e-15)", color="#898781", fontsize=8.5)
    ax_rec.set_ylabel("Δθ между фокусами", color="#898781", fontsize=9)
    ax_rec.tick_params(colors="#898781", labelsize=8)
    for s in ("top", "right"):
        ax_rec.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax_rec.spines[s].set_color("#383835")
    leg = ax_rec.legend(frameon=False, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color("white")

    return dict(fig=fig, ax3d=ax3d, tri=tri, dots=dots, tags=tags,
                thread=thread, thread_theta=thread_theta,
                note3d=note3d, arc_patches=arc_patches, solo=solo,
                arc_head=arc_head, arc_note=arc_note, col_dots=col_dots,
                col_tags=col_tags, col_trail=col_trail, col_note=col_note,
                marker=marker, ts=ts, gap1=gap1, gap2=gap2)


def make_update(state):
    tri, dots, tags = state["tri"], state["dots"], state["tags"]
    note3d = state["note3d"]
    arc_patches, solo = state["arc_patches"], state["solo"]
    arc_head, arc_note = state["arc_head"], state["arc_note"]
    col_dots, col_tags = state["col_dots"], state["col_tags"]
    col_trail, col_note = state["col_trail"], state["col_note"]
    thread, thread_theta = state["thread"], state["thread_theta"]
    marker, ts = state["marker"], state["ts"]
    gap1, gap2 = state["gap1"], state["gap2"]
    hsv = plt.get_cmap("hsv")

    def update(frame):
        t = 2.0 * math.pi * frame / N_FRAMES
        turn = TURNS * frame / N_FRAMES        # цветовых оборотов пройдено
        cphase = focus_colour_phase(turn)

        # Внутренняя фаза БЕЖИТ по нити. Скорость не свободна: T_d = T_цв
        # (5.9.5), поэтому за один цветовой оборот фаза набирает
        # omega * T_цв = 4pi, то есть ровно ДВА полных цикла 2pi.
        run = PHASE_PER_TURN * turn
        thread.set_array(
            ((0.5 * thread_theta - run) % (2 * np.pi))[:-1] / (2 * np.pi))
        thetas = focus_thetas(t)
        psis = focus_psis(t)
        m = triangle_metrics(thetas)
        pts = m["points"]

        order = ("u1", "d", "u2", "u1")
        tri.set_data([pts[k][0] for k in order], [pts[k][1] for k in order])
        tri.set_3d_properties([pts[k][2] for k in order])

        for k in COLORS:
            p = pts[k]
            dots[k].set_data([p[0]], [p[1]])
            dots[k].set_3d_properties([p[2]])
            # psi входит ТОЛЬКО в яркость и размер, но не в координаты
            pulse = 0.5 * (1.0 + math.cos(psis[k]))
            dots[k].set_markersize(10.0 + 8.0 * pulse)
            dots[k].set_alpha(0.60 + 0.40 * pulse)
            dots[k].set_markerfacecolor(hsv(cphase[k]))
            tags[k].set_position((p[0], p[1]))
            tags[k].set_3d_properties(p[2] + 0.35 * r, zdir=None)

        scale = fm_per_r(thetas)
        rest = triangle_metrics(THETA_0)
        note3d.set_text(
            "одна замкнутая нить, θ: 0 → 4π — два витка через горловину\n"
            "фаза бежит: 2 полных цикла за цветовой оборот (T_d ≡ T_цв, 5.9.5);\n"
            "фокусы стоят — это состояние покоя, а не заморозка\n"
            f"стороны {'/'.join(f'{v:.2f}' for v in sorted(m['sides'].values(), reverse=True))} r,"
            f" разброс {100 * m['spread']:.1f} %"
            f"   (в покое {100 * rest['spread']:.1f} %)\n"
            f"расстояния от оси различаются в {m['axis_ratio']:.2f} раза"
            f"   ({', '.join(f'{v * scale:.2f}' for v in sorted(m['axis'].values()))} фм)\n"
            "Δψ(u₁,u₂) = π — фазовый развод Паули, в координаты не входит")

        # счётчик обхода: дуги встык, сумма замыкается ровно в круг
        turn_frac = turn % 1.0
        cursor = 90.0
        for k, patch in arc_patches:
            span = float(ARC[k]) * 360.0 * turn_frac
            lo, hi = sorted((cursor, cursor + span))
            patch.theta1, patch.theta2 = lo, hi
            cursor += span
        ang = math.radians(cursor)
        arc_head.set_data([math.cos(ang)], [math.sin(ang)])
        # те же дуги по отдельности, каждая от общего старта
        for k, patch in solo:
            span = float(ARC[k]) * 360.0 * turn_frac
            lo, hi = sorted((90.0, 90.0 + span))
            patch.theta1, patch.theta2 = lo, hi
        gap_deg = (cursor - 90.0) - 360.0 * turn_frac
        arc_note.set_text(
            f"оборот {turn:.2f} из {TURNS}:   +2/3 +2/3 −1/3 = +1 круга\n"
            f"невязка замыкания {gap_deg:+.1e}°  —  тождественно нуль")

        # цветовой круг
        for k in COLORS:
            a = 2.0 * math.pi * cphase[k]
            col_dots[k].set_data([math.cos(a)], [math.sin(a)])
            col_dots[k].set_markerfacecolor(hsv(cphase[k]))
            col_tags[k].set_position((1.32 * math.cos(a), 1.32 * math.sin(a)))
            c0, ext = colour_trail(turn, k)
            a0 = 360.0 * c0
            span = 360.0 * ext
            lo, hi = sorted((a0, a0 + span))
            col_trail[k].theta1, col_trail[k].theta2 = lo, hi
        n_done = int(turn + 1e-9)
        exact = colour_permutation(n_done)
        same_set = set(exact.values()) == set(COLOUR_0.values())
        v = sorted(cphase.values())
        min_gap = min((v[1] - v[0]) % 1.0, (v[2] - v[1]) % 1.0, (v[0] - v[2]) % 1.0)
        col_note.set_text(
            f"пройдено {turn:.2f} оборота\n"
            f"минимальный зазор цветов {min_gap:.4f} = 1/3:\n"
            "синглет держится непрерывно, а не только на целых оборотах\n"
            "свой цвет каждый фокус возвращает через 3 оборота")

        i = int(np.searchsorted(ts, t % (2 * math.pi))) % len(ts)
        marker.set_data([ts[i], ts[i]], [gap1[i], gap2[i]])

        return [tri, arc_head, marker] + list(dots.values())

    return update


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="только численные проверки, без окна")
    ap.add_argument("--save", metavar="PNG",
                    help="сохранить один кадр в файл вместо анимации")
    args = ap.parse_args()

    check()
    if args.check:
        return

    state = build_figure()
    update = make_update(state)

    if args.save:
        update(int(0.17 * N_FRAMES))
        state["fig"].savefig(args.save, facecolor="black", dpi=120)
        print(f"кадр сохранён: {args.save}")
        return

    global ani
    ani = animation.FuncAnimation(state["fig"], update, frames=N_FRAMES,
                                  interval=33, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
