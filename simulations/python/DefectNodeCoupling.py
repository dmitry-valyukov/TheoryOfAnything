"""Зарядовый сектор и уровни: даёт ли дефект (центр дилатации, 3.6) поправку
к E(Q) семейства узлов, способную сдвинуть отношения уровней к лептонным.

Электрон = узел (сектор n = 1, TorusBVP) + точечный дефект u_def = A r^/r²
на оси (3.6.5). Полная плотность |W + u_def|² = rho + 2 Re(W·u_def) + A²/r^4.
Линейный перекрёстный член: (а) градиентная часть — нуль (u_def
соленоидально вне ядра и ортогонально по компонентам поперечной гирации);
(б) нелинейная часть — колеблется как e^{-i mu t} и в среднем по периоду
нуль. Первый неисчезающий вклад — квадратичный по u_def:

    dE_int = int f(rho(r)) |u_def|² dV = 4 pi A² int_{r_c} f(rho) dr / r²,
    f(rho) = -g rho + h rho²  (нелинейный «химпотенциал» среды в узле),

доминируется ядром r -> r_c:  dE_int ~ 4 pi A² f(rho_axis) / r_c.
Собственная энергия дефекта 8 pi mu A² / r_c³ (3.6.4) от Q не зависит.

Что считается:
  1. rho_axis(Q) на оси узла и f(rho_axis) вдоль семейства n = 1;
  2. интеграл dE_int(Q, r_c) по 3D-профилю узла для нескольких r_c —
     закон 1/r_c и коэффициент f(rho_axis) проверяются напрямую;
  3. вывод для уровней: поправка = const_1 (собственная энергия) +
     const_2 · f(rho_axis(Q)); f меняется от -0.5 (Q = 120) до -1.0
     (Q >= 500), монотонно и с насыщением, а по масштабу подавлена
     отношением (r_c/l_узла) ~ 1e-22 (1.5). Ни иерархии 1:207:3477,
     ни вообще заметного сдвига уровней зарядовый сектор дать не может:
     дефект один и тот же у всех лептонов (e одинаков), его связь с
     узлом — контактный член на оси. Побочно: дефект СВЯЗАН с узлом
     (f < 0 в центре: узел размягчает среду, поле дефекта там дешевле),
     и глубже всего — при rho_axis -> g/2h = 1, куда семейство и идёт
     (0.30 -> 0.96): заряд сидит в центре узла не по постулату, а по
     энергии.

Запуск:  python DefectNodeCoupling.py
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G_CUB, H_QUI = 2.0, 1.0


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78)


def f_of(rho):
    return -G_CUB * rho + H_QUI * rho ** 2


def coupling_integral(rho, R, Z, hh, r_c):
    """int f(rho) / r^4 dV по сетке (rho, z) вне сферы r_c; A = 1."""
    r = np.sqrt(R ** 2 + Z ** 2)
    w = 2 * math.pi * R * hh * hh
    mask = r >= r_c
    return float((f_of(rho) * w / r ** 4)[mask].sum())


def main():
    head("ДЕФЕКТ В ПОЛЕ УЗЛА: КОНТАКТНАЯ ПОПРАВКА К E(Q)")
    rec = json.load(open(os.path.join(HERE, "torusbvp_results.json")))
    Qs = sorted({int(round(r["Q"])) for r in rec.values() if r["n"] == 1})
    hh = 0.15
    rcs = (0.6, 0.9, 1.2)
    print("\n  A = 1. dE_int(r_c) = int f(rho)/r^4 dV вне r_c; ожидание ~ 4 pi f(rho_axis)/r_c\n")
    print("   Q    rho_axis  f(rho_axis)   " + "  ".join(f"dE(r_c={rc})" for rc in rcs)
          + "    dE·r_c/(4pi f_axis) при r_c=" + ",".join(str(rc) for rc in rcs))
    rows = []
    for Q in Qs:
        p = os.path.join(HERE, f"torusbvp_results_Q{Q}_n1_tube.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        rho, R, Z = d["rho"], d["R"], d["Z"]
        j = int(np.argmin(np.abs(Z[0])))
        rho_ax = float(rho[0, j])
        fa = f_of(rho_ax)
        vals = [coupling_integral(rho, R, Z, hh, rc) for rc in rcs]
        ratios = [v * rc / (4 * math.pi * fa) for v, rc in zip(vals, rcs)]
        rows.append((Q, rho_ax, fa, vals))
        print(f"  {Q:5d}   {rho_ax:.3f}    {fa:+.3f}      "
              + "  ".join(f"{v:+9.3f}" for v in vals)
              + "        " + ", ".join(f"{x:.2f}" for x in ratios))
    print("""
  Чтение. Интеграл идёт как 1/r_c с коэффициентом ~ 4 pi f(rho_axis)
  (отношение в последнем столбце ~ 1 и растёт к 1 при r_c -> 0: контактный
  член на оси). Знак отрицательный: дефект СВЯЗАН с узлом — центр узла
  размягчает среду (f < 0), и поле дефекта там дешевле, чем в пустоте.
  Глубже всего при rho_axis = g/2h = 1 (минимум f = -g²/4h = -1); семейство
  к этому и идёт: rho_axis 0.30 (Q = 120) -> 0.96 (Q >= 900).

  Для уровней. Полная энергия E(Q) + E_self + 4 pi A² f(rho_axis(Q))/r_c:
  E_self одинакова у всех лептонов (тот же e => тот же дефект), контактный
  член меняется вдоль семейства лишь в 2 раза (f: -0.5 -> -1.0), монотонно
  и с насыщением, а по масштабу подавлен: A ~ d³, r_c ~ d, размер узла
  ~ 1e22 d (1.5) => dE_int/E_узла ~ (d/l)^5 * ... — исчезающе мало.
  Иерархию 1 : 207 : 3477 зарядовый сектор дать не может: ни как поправка
  к семейству, ни сам по себе. Из кандидатов на механизм поколений
  (2.9.5, 2.10) остаётся только внешний масштаб — спектр самой среды
  (калибровочный вход).
""")


if __name__ == "__main__":
    main()
