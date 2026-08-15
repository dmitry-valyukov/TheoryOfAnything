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

6. **Поляризация показана полем, а не траекторией** (T-37, 2.5.7). Вдоль
   каждой дуги рисуется само поле смещений u поперёк нити: у u-мод вектор
   обходит окружность в сечении (круговая поляризация, две плоскости
   вращения — стрелки образуют винт вдоль дуги, знак винта = знак намотки),
   у d — качается вдоль одного отрезка (линейная поляризация, одна
   плоскость — стрелки лежат в одной плоскости и меняют лишь длину и знак).
   Это свойство поля в окрестности дуги, а не точки, и изображается законно;
   локальные петли у фокуса (точка не имеет обхода) отвергнуты.

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
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import RotationPlanes as RP

from lib.modes import to_segments
from lib.plot import Arrows, rgba
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

#: число ортогональных плоскостей вращения, задействованных модой (5.2):
#: u использует две из трёх, d заперт в одной. |q| = n_planes / D, D = 3.
PLANES = {"u1": 2, "u2": 2, "d": 1}

#: направление топологической намотки. У d знак инвертирован относительно u
#: (5.2) — та же операция, что различает электрон и позитрон.
SENSE = {"u1": +1, "u2": +1, "d": -1}

#: доля полного спинорного обхода 4pi за один цветовой оборот (5.11.1-5.11.2)
LEG = Fraction(1, 3)
#: заряд собирается из двух множителей: сколько плоскостей и в какую сторону.
ARC = {k: Fraction(SENSE[k] * PLANES[k], 3) for k in PLANES}

#: начальные цветовые фазы: три цвета, разнесённые на 1/3 радужного круга.
#: Одно ребро цветового контура есть d(phi)/4pi = 1/3 полного обхода (5.11.1).
COLOUR_0 = {"u1": Fraction(0), "d": Fraction(1, 3), "u2": Fraction(2, 3)}

#: набег внутренней фазы за один цветовой оборот. Из T_d = T_цв = 4pi/omega
#: (5.9.5) следует omega * T_цв = 4pi — ровно два полных цикла 2pi.
#: Величина выведена, а не подобрана.
PHASE_PER_TURN = 4.0 * math.pi

#: модельные входы для счёта плоскостей: ось стоит у d, ходит по дуге у u.
#: Это ПРОВЕРОЧНЫЕ входы, а не решение уравнений поля (T-36).
OMEGA_SAMPLE = {k: (RP.sample_axis_fixed(SENSE[k]) if PLANES[k] == 1
                    else RP.sample_axis_arc(SENSE[k])) for k in PLANES}
PLANE_FRACTIONS = {k: RP.plane_spectrum(OMEGA_SAMPLE[k])["fractions"]
                   for k in PLANES}

#: радиальные сдвиги трёх дуг внутри трубки — чтобы они не сливались в одну
#: линию там, где идут по общему участку нити. Величина чисто изобразительная.
ARC_OFFSET = {"u1": 0.22, "d": 0.0, "u2": -0.22}

#: стрелки поля смещений вдоль дуги (T-37): число на дугу, амплитуда,
#: пространственных периодов поляризационного винта на дугу
N_POL = 14
POL_AMP = 0.42 * r
POL_TURNS = {"u1": 2.0, "u2": 2.0, "d": 1.0}

#: сколько цветовых оборотов проигрывает анимация. Каждый фокус возвращает
#: свой исходный цвет ровно через 3 оборота (проверка в check(), пункт 6).
TURNS = 3

#: Поворот керна за один цветовой оборот. Величина НЕ выведена (см. T-35):
#: сам факт вращения триады модель не фиксирует. Значение 4pi/3 выбрано из
#: соображения соизмеримости — при нём три цветовых оборота дают Theta = 4pi,
#: то есть полный спинорный обход, и дыхание триады возвращается в исходную
#: форму ровно тогда же, когда цвета возвращаются к себе.
KERN_TURN = 4.0 * math.pi / 3.0

#: усиление наведённого отклика d-фокуса на асимметрию расстояний до двух u.
#: Иллюстративное: модель фиксирует равновесие d(theta) = 2pi/3, но не
#: жёсткость отклика.
GAIN_D = 0.45

#: амплитуда дифференциальной отдачи двух u на смещение d; знаки
#: противоположны (5.10.3). Тоже иллюстративная.
EPS_RECOIL = 0.25

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


def local_frame(theta: float):
    """
    Ортонормированный репер в точке нити: (t, n, b).

    t — касательная к нити (направление обхода),
    n — от оси трубки к точке (радиальная в сечении),
    b = t x n.

    Три ортогональные плоскости вращения — это (n,b), (t,n), (t,b).
    Именно они, а не путь нити, несут структуру моды: сколько плоскостей
    задействовано (заряд) и в какую сторону идёт вращение (знак).
    """
    h = 1e-6
    p0 = _point_raw(theta)
    t = _point_raw(theta + h) - _point_raw(theta - h)
    t /= np.linalg.norm(t)
    centre = np.array([R * math.cos(theta), R * math.sin(theta), 0.0])
    n = p0 - centre
    n -= np.dot(n, t) * t
    n /= np.linalg.norm(n)
    b = np.cross(t, n)
    return t, n, b


def wound_curve_offset(theta, off: float = 0.0):
    """
    Та же намотка, но на радиусе трубки r + off.

    Нормаль к нити в сечении трубки есть n = (cos(phi)cos(th), cos(phi)sin(th),
    sin(phi)), поэтому сдвиг вдоль n сводится к замене r -> r + off.
    Нужен, чтобы три дуги, идущие по общей нити, не сливались в одну линию.
    """
    theta = np.asarray(theta, dtype=float)
    phi = 0.5 * theta
    rr = r + off
    rad = R + rr * np.cos(phi)
    return rad * np.cos(theta), rad * np.sin(theta), rr * np.sin(phi)


def mode_arc(theta0: float, frac, off: float = 0.0, num: int = 260):
    """
    Дуга, занимаемая модой: её доля полного обхода.

    Полный спинорный обход есть 4pi по theta. Мода занимает долю frac от него:
    u — плюс 2/3 (8pi/3 вперёд), d — минус 1/3 (4pi/3 назад). Ни одна дуга
    сама на себя не замыкается — это и есть «дефицит обхода» (5.11.1).

    Длина дуги ПОСТОЯННА, а движение передаётся бегущей по ней волной
    яркости: иначе дуга росла бы и обрывалась на каждом обороте, и анимация
    не зацикливалась бы гладко.
    """
    span = float(frac) * 4.0 * math.pi
    th = np.linspace(theta0, theta0 + span, num)
    x, y, z = wound_curve_offset(th, off)
    return th, np.stack([x, y, z], axis=1)


def running_bump(s_, phase: float, width: float = 0.30):
    """Гладкий периодический горб яркости, бегущий по дуге. Период ровно 1."""
    x = (np.asarray(s_, dtype=float) - phase) % 1.0
    d = np.minimum(x, 1.0 - x) / width
    return np.exp(-0.5 * d * d)


def front_ticks(th, off: float = 0.0, half: float = 0.13, step: int = 22):
    """Поперечные штрихи волнового фронта поперёк трубки — как в сцене электрона."""
    sel = th[::step]
    if len(sel) == 0:
        return np.zeros((0, 2, 3))
    a = np.stack(wound_curve_offset(sel, off - half), axis=1)
    b = np.stack(wound_curve_offset(sel, off + half), axis=1)
    return np.stack([a, b], axis=1)


def polarization_field(k: str, th, off: float, run: float, psi: float):
    """
    Поле смещений u поперёк нити вдоль дуги моды k (T-37).

    В каждой точке дуги берётся репер (t, n, b); фаза Phi бежит вдоль дуги
    (POL_TURNS периодов) и во времени (run, psi):
        u-моды: u = A [cos(Phi) n + sense * sin(Phi) b] — вектор обходит
                окружность в сечении: круговая поляризация, две плоскости;
        d:      u = A cos(Phi) n — вектор качается вдоль отрезка:
                линейная поляризация, одна плоскость.
    Возвращает (начала, векторы) формы (N_POL, 3).
    """
    sel = np.linspace(th[0], th[-1], N_POL)
    frac = (sel - th[0]) / max(th[-1] - th[0], 1e-9)
    phi = 2.0 * math.pi * POL_TURNS[k] * frac - OMEGA[k] * run - psi
    origins = np.stack(wound_curve_offset(sel, off), axis=1)
    vecs = np.zeros_like(origins)
    for i, thi in enumerate(sel):
        _, n, b = local_frame(float(thi))
        if PLANES[k] == 1:
            vecs[i] = POL_AMP * math.cos(phi[i]) * n
        else:
            vecs[i] = POL_AMP * (math.cos(phi[i]) * n
                                 + SENSE[k] * math.sin(phi[i]) * b)
    return origins, vecs


def _slots(theta_kern: float) -> dict:
    """Три геометрических гнезда триады при повороте керна на theta_kern."""
    return {
        "u1": theta_kern,
        "d": theta_kern + 2.0 * math.pi / 3.0,
        "u2": theta_kern + 4.0 * math.pi / 3.0,
    }


def _point_raw(theta: float) -> np.ndarray:
    x, y, z = wound_curve(theta)
    return np.array([float(x), float(y), float(z)])


_point = _point_raw


def focus_thetas(turn: float, iters: int = 12) -> dict:
    """
    Положения трёх фокусов при цветовом обороте turn.

    Керн (пара u) вращается по намотке: theta_kern = KERN_TURN * turn. Из-за
    мёбиусовой намотки rad = R + r cos(theta/2) треугольник фокусов при этом
    НЕ жёсткий — расстояние между двумя u дышит с размахом около 45%
    (проверка в check(), пункт 8).

    d-фокус не задан руками, а НАВЕДЁН: он смещается вдоль нити в сторону,
    выравнивающую расстояния до двух u. Это отклик на реакцию пары, а не
    независимое качание. Решается простой итерацией до неподвижной точки;
    смещение d, в свою очередь, даёт двум u дифференциальную отдачу
    противоположных знаков (5.10.3).
    """
    base = _slots(KERN_TURN * turn)
    delta = 0.0
    for _ in range(iters):
        recoil = EPS_RECOIL * delta
        th = {"u1": base["u1"] - recoil,
              "d": base["d"] + delta,
              "u2": base["u2"] + recoil}
        p = {k: _point(v) for k, v in th.items()}
        d1 = float(np.linalg.norm(p["d"] - p["u1"]))
        d2 = float(np.linalg.norm(p["d"] - p["u2"]))
        asym = (d1 - d2) / (0.5 * (d1 + d2))
        delta_new = -GAIN_D * asym         # сдвиг к более далёкому u
        if abs(delta_new - delta) < 1e-12:
            delta = delta_new
            break
        delta = 0.5 * (delta + delta_new)  # демпфирование итерации
    recoil = EPS_RECOIL * delta
    return {"u1": base["u1"] - recoil,
            "d": base["d"] + delta,
            "u2": base["u2"] + recoil}


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


def focus_psis(turn: float) -> dict:
    """Фазы возбуждения — независимая переменная, в геометрию НЕ входит."""
    return {k: PSI_0[k] + OMEGA[k] * PHASE_PER_TURN * turn for k in PSI_0}


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

    print("\n5. Заряд из двух множителей: сколько плоскостей и в какую сторону (5.2)")
    print(f"   {'мода':>5} {'плоскостей':>11} {'намотка':>9} {'|q| = n/3':>10} {'q':>7}")
    for k in ("u1", "u2", "d"):
        print(f"   {LABELS[k]:>5} {PLANES[k]:>11} {SENSE[k]:>+9d} "
              f"{Fraction(PLANES[k], 3)!s:>10} {ARC[k]!s:>7}")
    print("   формула 5.2:  |q| = n_плоскостей / D,  D = 3;  знак — направление")
    print("   намотки (у d инвертировано, как у позитрона относительно электрона)")
    print("   цвет отвечает на вопрос КАКИЕ плоскости, заряд — СКОЛЬКО их")

    print("\n5b. Счётчик доли обхода и заряд (5.11)")
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
    print("   T_d = 2pi/omega_d = 4pi/omega,  T_цв = 3*d(phi)/omega = 4pi/omega")
    print("   => набег внутренней фазы за цветовой оборот = omega*T_цв = 4pi")
    print(f"      = {PHASE_PER_TURN / (2 * math.pi):.0f} полных цикла 2pi "
          "(PHASE_PER_TURN, величина ВЫВЕДЕНА)")
    print("   керн вращается по намотке: Theta = KERN_TURN * turn,")
    print(f"      KERN_TURN = 4pi/3 за оборот => за {TURNS} оборота Theta = 4pi,")
    print("      то есть полный спинорный обход. Сама величина НЕ выведена:")
    print("      замыкание 4pi даёт полуцелость J, а не его величину")
    print("      (Delta-изобара — тот же тор, спин 3/2). См. T-35.")

    print("\n8. Дыхание триады при вращении керна")
    th4 = np.linspace(0.0, 4.0 * math.pi, 4001)
    kern = np.array([float(np.linalg.norm(_point(t) - _point(t + 4 * math.pi / 3)))
                     for t in th4])
    print(f"   расстояние u1-u2 при жёстком повороте: min {kern.min():.4f} r, "
          f"max {kern.max():.4f} r")
    print(f"   размах {100 * (kern.max() - kern.min()) / kern.mean():.1f} % — "
          "керн то сближается, то отдаляется")
    print(f"   период по Theta равен 4pi: |P(0) - P(4pi)| = "
          f"{float(np.linalg.norm(_point(0.0) - _point(4 * math.pi))):.2e}")
    print("   причина — мёбиусова намотка rad = R + r cos(theta/2):")
    print("   треугольник фокусов не жёсткий, при повороте меняет форму")

    print("\n9. Наведённый отклик d-фокуса")
    print(f"   {'оборот':>7} {'delta_d':>9} {'|d-u1|':>9} {'|d-u2|':>9} "
          f"{'асимм.':>9} {'без отклика':>12}")
    for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
        th = focus_thetas(tv)
        p = {k: _point(v) for k, v in th.items()}
        d1 = float(np.linalg.norm(p["d"] - p["u1"]))
        d2 = float(np.linalg.norm(p["d"] - p["u2"]))
        delta = th["d"] - (KERN_TURN * tv + 2 * math.pi / 3)
        b = _slots(KERN_TURN * tv)
        q = {k: _point(v) for k, v in b.items()}
        b1 = float(np.linalg.norm(q["d"] - q["u1"]))
        b2 = float(np.linalg.norm(q["d"] - q["u2"]))
        print(f"   {tv:7.2f} {delta:9.5f} {d1:9.4f} {d2:9.4f} "
              f"{(d1 - d2) / (0.5 * (d1 + d2)):9.5f} "
              f"{(b1 - b2) / (0.5 * (b1 + b2)):12.5f}")
    print("   d не задан руками: он смещается к более далёкому u, выравнивая")
    print("   расстояния; смещение d даёт двум u отдачу противоположных знаков.")
    print("   Усиление GAIN_D и EPS_RECOIL моделью не выводятся.")

    print("\n" + line)


# ---------------------------------------------------------------------------
# Сцена
# ---------------------------------------------------------------------------


def build_figure():
    fig = plt.figure(figsize=(16.0, 11.5), facecolor="black")
    gs = fig.add_gridspec(4, 2, width_ratios=(1.25, 1.0),
                          height_ratios=(0.80, 0.95, 1.00, 0.80),
                          left=0.02, right=0.97, top=0.93, bottom=0.05,
                          wspace=0.06, hspace=0.60)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="black")
    ax_q = fig.add_subplot(gs[0, 1], facecolor="black")
    ax_arc = fig.add_subplot(gs[1, 1], facecolor="black")
    ax_col = fig.add_subplot(gs[2, 1], facecolor="black")
    ax_rec = fig.add_subplot(gs[3, 1], facecolor="black")

    fig.suptitle("Протон: одна намотка, три фокуса, замыкание обхода",
                 color="white", fontsize=15, x=0.28, y=0.975)

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

    # локальные петли вращения: до двух на моду (u — две плоскости, d — одна)
    # дуга, пройденная каждой модой за цветовой оборот: 8pi/3 вперёд у u,
    # 4pi/3 назад у d. Радиальные сдвиги, чтобы три дуги не сливались.
    arcs, arc_ticks, arc_fronts = {}, {}, {}
    for k in COLORS:
        lc = Line3DCollection(np.zeros((1, 2, 3)), linewidths=4.5)
        ax3d.add_collection3d(lc)
        arcs[k] = lc
        tk = Line3DCollection(np.zeros((1, 2, 3)), linewidths=1.4,
                              colors=[(*mcolors.to_rgb(COLORS[k]), 0.55)])
        ax3d.add_collection3d(tk)
        arc_ticks[k] = tk
        arc_fronts[k] = ax3d.plot([], [], [], marker="o", ms=10.0, ls="none",
                                  color=COLORS[k], mec="white", mew=1.0)[0]
    # поле смещений вдоль дуг: винт у u, качание в плоскости у d (T-37)
    pol_arrows = {k: Arrows(ax3d, N_POL, rgba(COLORS[k], 0.95),
                            arrow_length_ratio=0.28) for k in COLORS}
    for k in COLORS:
        pol_arrows[k].set_linewidths(1.3)

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

    # --- откуда берётся заряд ---
    ax_q.set_xlim(-0.1, 10.2)
    ax_q.set_ylim(-1.60, 4.05)
    ax_q.axis("off")
    ax_q.set_title("Откуда берётся ±2/3 и ∓1/3\n"
                   "n_П = rank⟨ωωᵀ⟩ — сколько плоскостей; цвет — какие (5.2)",
                   color="white", fontsize=11, pad=6)
    slot_patches, sense_arcs, q_texts = {}, {}, {}
    for row, k in enumerate(("u1", "d", "u2")):
        y = 2.9 - row
        ax_q.text(0.55, y, LABELS[k], color=COLORS[k], fontsize=12,
                  weight="bold", ha="center", va="center")
        cells = []
        for j in range(3):
            x0 = 1.35 + 1.35 * j
            ax_q.plot([x0, x0 + 1.15], [y - 0.30, y - 0.30],
                      color="#333331", lw=1.0)
            ax_q.plot([x0 + RP.TOL * 1.15] * 2, [y - 0.36, y + 0.36],
                      color="#4a4a46", lw=0.8, ls=":")
            bar = ax_q.barh(y, 0.0, height=0.44, left=x0, color="none",
                            edgecolor="none")[0]
            cells.append((bar, x0))
            if row == 0:
                ax_q.text(x0 + 0.58, 3.62, f"λ{j + 1}", color="#6a6a64",
                          fontsize=8.5, ha="center", va="center")
        slot_patches[k] = cells
        # знак намотки: кривая стрелка, направление читается однозначно
        cx0 = 5.85
        p1, p2 = (cx0 - 0.42, y - 0.10), (cx0 + 0.42, y - 0.10)
        src, dst = (p1, p2) if SENSE[k] > 0 else (p2, p1)
        rad = 0.6 if SENSE[k] > 0 else -0.6
        ax_q.add_patch(FancyArrowPatch(
            src, dst, connectionstyle=f"arc3,rad={rad}", color=COLORS[k],
            lw=2.0, arrowstyle="-|>", mutation_scale=13,
            shrinkA=0, shrinkB=0))
        sense_arcs[k] = None
        ax_q.text(6.60, y, "намотка +" if SENSE[k] > 0 else "намотка −",
                  color="#898781", fontsize=9, ha="left", va="center")
        q_texts[k] = ax_q.text(9.75, y,
                               f"{'+' if ARC[k] > 0 else '−'}"
                               f"{abs(ARC[k])}",
                               color=COLORS[k], fontsize=13, weight="bold",
                               ha="right", va="center")
    ax_q.plot([1.4, 9.9], [0.28, 0.28], color="#4a4a46", lw=0.9)
    ax_q.text(9.75, -0.16, "+2/3 + 2/3 − 1/3 = +1", color="white",
              fontsize=12, weight="bold", ha="right", va="center")
    ax_q.text(0.1, -1.15,
              "полосы — доли собственных значений матрицы ⟨ωωᵀ⟩; за порогом "
              f"(пунктир, {RP.TOL}) — задействованная плоскость,\nцвет полосы "
              "есть цветовой заряд. Ранг целочислен и устойчив к шуму — отсюда "
              "точная квантованность.\nВход ω(t) здесь МОДЕЛЬНЫЙ, а не решение "
              "уравнений поля: см. T-36 и RotationPlanes.py.",
              color="#898781", fontsize=8.0, ha="left", va="center")

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

    # --- дыхание триады ---
    ax_rec.set_facecolor("black")
    ax_rec.set_title("Дыхание триады при вращении керна\n"
                     "Θ = 4π/3 за цветовой оборот; за 3 оборота полный обход 4π",
                     color="white", fontsize=10.5, pad=4)
    ts = np.linspace(0.0, float(TURNS), 400)
    dist = {"u1u2": [], "u1d": [], "du2": []}
    for tv in ts:
        th = focus_thetas(tv)
        p = {k: _point(v) for k, v in th.items()}
        dist["u1u2"].append(float(np.linalg.norm(p["u1"] - p["u2"])))
        dist["u1d"].append(float(np.linalg.norm(p["u1"] - p["d"])))
        dist["du2"].append(float(np.linalg.norm(p["d"] - p["u2"])))
    dist = {k: np.array(v) for k, v in dist.items()}
    ax_rec.plot(ts, dist["u1u2"], color="white", lw=2.4, label="керн u₁–u₂")
    ax_rec.plot(ts, dist["u1d"], color=COLORS["u1"], lw=1.8, label="u₁–d")
    ax_rec.plot(ts, dist["du2"], color=COLORS["u2"], lw=1.8, label="d–u₂")
    marker = ax_rec.plot([], [], marker="o", ls="none", color="white", ms=6)[0]
    ax_rec.set_xlim(0, float(TURNS))
    ax_rec.set_xlabel("цветовых оборотов   (d наведён откликом на пару u)",
                      color="#898781", fontsize=8.5)
    ax_rec.set_ylabel("расстояние, в единицах r", color="#898781", fontsize=9)
    ax_rec.tick_params(colors="#898781", labelsize=8)
    for sp in ("top", "right"):
        ax_rec.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_rec.spines[sp].set_color("#383835")
    leg = ax_rec.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=3)
    for txt in leg.get_texts():
        txt.set_color("white")

    return dict(fig=fig, ax3d=ax3d, tri=tri, dots=dots, tags=tags,
                slot_patches=slot_patches, sense_arcs=sense_arcs,
                arcs=arcs, arc_ticks=arc_ticks, arc_fronts=arc_fronts,
                pol_arrows=pol_arrows, thread=thread, thread_theta=thread_theta,
                note3d=note3d, arc_patches=arc_patches, solo=solo,
                arc_head=arc_head, arc_note=arc_note, col_dots=col_dots,
                col_tags=col_tags, col_trail=col_trail, col_note=col_note,
                marker=marker, ts=ts, dist=dist)


def make_update(state):
    tri, dots, tags = state["tri"], state["dots"], state["tags"]
    note3d = state["note3d"]
    arc_patches, solo = state["arc_patches"], state["solo"]
    arc_head, arc_note = state["arc_head"], state["arc_note"]
    col_dots, col_tags = state["col_dots"], state["col_tags"]
    col_trail, col_note = state["col_trail"], state["col_note"]
    slot_patches, sense_arcs = state["slot_patches"], state["sense_arcs"]
    arcs, arc_ticks = state["arcs"], state["arc_ticks"]
    arc_fronts = state["arc_fronts"]
    pol_arrows = state["pol_arrows"]
    thread, thread_theta = state["thread"], state["thread_theta"]
    marker, ts, dist = state["marker"], state["ts"], state["dist"]
    hsv = plt.get_cmap("hsv")

    def update(frame):
        # цветовых оборотов пройдено; wrap делает анимацию точно зацикленной
        turn = (TURNS * frame / N_FRAMES) % TURNS
        cphase = focus_colour_phase(turn)

        # Внутренняя фаза БЕЖИТ по нити. Скорость не свободна: T_d = T_цв
        # (5.9.5), поэтому за один цветовой оборот фаза набирает
        # omega * T_цв = 4pi, то есть ровно ДВА полных цикла 2pi.
        run = PHASE_PER_TURN * turn
        thread.set_array(
            ((0.5 * thread_theta - run) % (2 * np.pi))[:-1] / (2 * np.pi))
        thetas = focus_thetas(turn)
        psis = focus_psis(turn)
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
            "керн вращается по намотке, треугольник дышит на 45 %;\n"
            "сам факт вращения моделью не выведен (T-35)\n"
            f"стороны {'/'.join(f'{v:.2f}' for v in sorted(m['sides'].values(), reverse=True))} r,"
            f" разброс {100 * m['spread']:.1f} %"
            f"   (в покое {100 * rest['spread']:.1f} %)\n"
            f"расстояния от оси различаются в {m['axis_ratio']:.2f} раза"
            f"   ({', '.join(f'{v * scale:.2f}' for v in sorted(m['axis'].values()))} фм)\n"
            "Δψ(u₁,u₂) = π — фазовый развод Паули, в координаты не входит\n"
            "стрелки — поле смещений u поперёк нити: у u вектор обходит окружность\n"
            "(круговая поляризация, винт вдоль дуги, две плоскости), у d качается\n"
            "вдоль отрезка (линейная, одна плоскость) — 2.5.7, T-37")

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

        # панель заряда: какие плоскости заняты и в какую сторону намотка
        for k in COLORS:
            j = int(round(3.0 * cphase[k])) % 3
            if PLANES[k] == 1:
                active = {j}                      # d занимает одну плоскость
            else:
                active = {0, 1, 2} - {j}          # у u свободна плоскость j

            # доли собственных значений M = <omega omega^T>: большие уходят
            # в задействованные плоскости, малые — в свободные
            big = sorted(PLANE_FRACTIONS[k], reverse=True)
            vals = [0.0] * 3
            for pos, idx in enumerate(sorted(active)):
                vals[idx] = float(big[pos])
            rest = list(big[len(active):])
            for idx in [i for i in range(3) if i not in active]:
                vals[idx] = float(rest.pop(0)) if rest else 0.0
            for idx, (bar, _) in enumerate(slot_patches[k]):
                bar.set_width(1.15 * vals[idx])
                if vals[idx] > RP.TOL:
                    bar.set_facecolor(hsv(cphase[k]))
                    bar.set_alpha(0.92)
                else:
                    bar.set_facecolor("#3a3a37")
                    bar.set_alpha(1.0)

        # дуги мод: длина постоянна (доля обхода), движение — бегущая волна
        for k in COLORS:
            th, pts = mode_arc(thetas[k], ARC[k], ARC_OFFSET[k])
            base_rgb = mcolors.to_rgb(COLORS[k])
            s_ = np.linspace(0.0, 1.0, len(pts))
            bump = running_bump(s_, turn)
            arcs[k].set_segments(to_segments(pts[:, 0], pts[:, 1], pts[:, 2]))
            alpha = 0.22 + 0.78 * bump[:-1]
            arcs[k].set_color([(*base_rgb, float(a)) for a in alpha])
            arcs[k].set_linewidths(1.6 + 4.0 * bump[:-1])
            arc_ticks[k].set_segments(front_ticks(th, ARC_OFFSET[k]))
            i_head = int(np.argmax(bump))
            arc_fronts[k].set_data([pts[i_head, 0]], [pts[i_head, 1]])
            arc_fronts[k].set_3d_properties([pts[i_head, 2]])
            # поле смещений: винт (u, круговая) против качания (d, линейная)
            org, vec = polarization_field(k, th, ARC_OFFSET[k], run, psis[k])
            pol_arrows[k].recalc(org[:, 0], org[:, 1], org[:, 2],
                                 vec[:, 0], vec[:, 1], vec[:, 2])

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

        i = min(int(np.searchsorted(ts, turn % TURNS)), len(ts) - 1)
        marker.set_data([ts[i]] * 3,
                        [dist["u1u2"][i], dist["u1d"][i], dist["du2"][i]])

        artists = [tri, arc_head, marker] + list(dots.values())
        for k in COLORS:
            artists += [arcs[k], arc_ticks[k], arc_fronts[k], pol_arrows[k]]
        return artists

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
