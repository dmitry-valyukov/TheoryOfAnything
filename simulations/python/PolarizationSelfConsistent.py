"""Самосогласованный отбор поляризации (T-36, третья развязка).

Проблема (T-36, пункт 3). VectorPolarization.py берёт профиль f(r) от СКАЛЯРНОГО
решения (это скрыто круговой случай lam_pol=1) и на нём, при фиксированных моментах
I2=int N^2, I3=int N^3, минимизирует энергию по эллиптичности lam_pol. Профиль по
lam_pol НЕ пересчитывается, намотки нет. Итог: инверсия u/d (круговым выходит d, а не u).
Вопрос: переживёт ли инверсия самосогласованный пересчёт профиля.

Ключевое наблюдение (делает Part A точной, а не приближённой). Усреднение по времени
даёт <|u|^2>=N, <|u|^4>=N^2 P4(lam), <|u|^6>=N^3 P6(lam); градиент/масса/кинетика зависят
только от N. Значит приведённая плотность энергии моды с амплитудой f (N=f^2) и
поляризацией lam есть

    dens = 1/2 f'^2 + 1/2 Lam f^2 - (g/4) P4(lam) f^4 + (h/6) P6(lam) f^6,

то есть В ТОЧНОСТИ кубик-квинтик с ЭФФЕКТИВНЫМИ связями g_eff=g*P4(lam), h_eff=h*P6(lam)
при той же частоте Lam=m^2-omega^2. Поэтому самосогласованный профиль для поляризации
lam получается тем же решателем CubicQuintic с (g_eff, h_eff), а полная энергия —
profile_energy на нём. Никакого нового решателя не нужно — только правильные связи.

Part A: при каждой частоте Lam минимизировать E(lam_pol) по lam_pol, где для каждого
lam_pol профиль пересчитан с (g*P4, h*P6). Сравнить порог с несамосогласованным. Затем
разложить u (частота выше, Lam меньше) и d (omega_d=omega_u/2, Lam больше) и проверить,
кто круговой.

Part B (первый срез, не полная тороидальная геометрия): добавить центробежный член
W/r^2 (W=l(l+1)) — намотку моды — и посмотреть, сдвигает ли она порог/расклад.
Явно оговорено: это сферическая редукция с барьером, а не вихрь в полоидальной плоскости.

Запуск:  python PolarizationSelfConsistent.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

import CubicQuintic as CQ
from VectorPolarization import P4, P6


# ---------------------------------------------------------------------------
# Part A: самосогласованный отбор без намотки — через эффективные связи
# ---------------------------------------------------------------------------

def sc_energy(Lam: float, lam_pol: float, g: float = 1.0, h: float = 1.0):
    """Полная приведённая энергия моды частоты Lam при поляризации lam_pol,
    с САМОСОГЛАСОВАННЫМ профилем (эффективные связи g*P4, h*P6)."""
    ge, he = g * float(P4(lam_pol)), h * float(P6(lam_pol))
    # окно существования эффективной задачи: Lam < 3 ge^2/(16 he)
    if Lam >= 3.0 * ge * ge / (16.0 * he):
        return None
    f0 = CQ.find_ground_state(Lam, ge, he, n_grid=90, n_bisect=40)
    if f0 is None:
        return None
    E, R = CQ.profile_energy(f0, Lam, ge, he)
    # заряд-подобный момент Q ~ int N dV = int f^2 dV (при фикс. omega)
    _, sol = CQ._shoot(f0, Lam, ge, he, fast=False)
    r, f = sol.t, sol.y[0]
    cut = int(np.argmin(np.abs(f)))
    r, f = r[:cut + 1], f[:cut + 1]
    Q = 4 * math.pi * float(np.trapezoid(f ** 2 * r ** 2, r))
    return {"E": E, "R": R, "Q": Q, "f0": f0, "ge": ge, "he": he}


def select_selfconsistent(Lam: float, g: float = 1.0, h: float = 1.0, ngrid: int = 13):
    """Минимум полной энергии по lam_pol при фиксированной частоте Lam."""
    lams = np.linspace(0.0, 1.0, ngrid)
    Es, Qs = [], []
    for lp in lams:
        r = sc_energy(Lam, lp, g, h)
        Es.append(r["E"] if r else np.nan)
        Qs.append(r["Q"] if r else np.nan)
    Es = np.array(Es); Qs = np.array(Qs)
    ok = np.isfinite(Es)
    if ok.sum() < 3:
        return None
    i = int(np.nanargmin(Es))
    # энергия при фиксированном заряде: E/Q как более честная мишень Q-шара
    EQ = Es / Qs
    iq = int(np.nanargmin(EQ))
    return {"lams": lams, "E": Es, "Q": Qs, "lam_star_E": float(lams[i]),
            "lam_star_EbyQ": float(lams[iq]), "E_lin": Es[0], "E_circ": Es[ok][-1]}


# ---------------------------------------------------------------------------
# Part B: намотка как центробежный барьер W/r^2 (первый срез)
# ---------------------------------------------------------------------------

def _rhs_wind(r, y, Lam, ge, he, W):
    f, fp = y
    curv = 0.0 if r < 1e-9 else 2.0 * fp / r
    cent = 0.0 if r < 1e-9 else W * f / r ** 2
    return [fp, -curv + cent + Lam * f - ge * f ** 3 + he * f ** 5]


def _shoot_wind(c: float, Lam, ge, he, W, l, r0=1e-4):
    """Пристрелка с f ~ c r^l у начала (для W=l(l+1)); знак ухода на бесконечности."""
    A = math.sqrt(ge / he)
    L = math.sqrt(he) / ge
    f0 = c * r0 ** l
    fp0 = c * l * r0 ** (l - 1) if l > 0 else 0.0

    def ev_hi(r, y, *a): return y[0] - 3.0 * A
    def ev_lo(r, y, *a): return y[0] + 3.0 * A
    ev_hi.terminal = ev_lo.terminal = True
    sol = solve_ivp(_rhs_wind, [r0, 200.0 * L], [f0, fp0], args=(Lam, ge, he, W),
                    rtol=1e-9, atol=1e-12, events=[ev_hi, ev_lo], max_step=0.05 * L)
    if sol.t_events[0].size: return +1, sol
    if sol.t_events[1].size: return -1, sol
    return (+1 if sol.y[0][-1] > 0 else -1), sol


def ground_state_wind(Lam, ge, he, l, n_grid=200, n_bis=60):
    A = math.sqrt(ge / he)
    grid = np.linspace(0.02 * A, 3.0 * A, n_grid)
    signs = [_shoot_wind(x, Lam, ge, he, l * (l + 1.0), l)[0] for x in grid]
    br = [(grid[i], grid[i + 1]) for i in range(n_grid - 1) if signs[i] != signs[i + 1]]
    if not br:
        return None
    a, b = br[0]; sa = _shoot_wind(a, Lam, ge, he, l * (l + 1.0), l)[0]
    for _ in range(n_bis):
        mid = 0.5 * (a + b)
        if _shoot_wind(mid, Lam, ge, he, l * (l + 1.0), l)[0] == sa:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def sc_energy_wind(Lam, lam_pol, l, g=1.0, h=1.0):
    """Энергия при поляризации lam_pol И намотке l (центробежный барьер)."""
    ge, he = g * float(P4(lam_pol)), h * float(P6(lam_pol))
    if Lam >= 3.0 * ge * ge / (16.0 * he):
        return None
    c = ground_state_wind(Lam, ge, he, l)
    if c is None:
        return None
    W = l * (l + 1.0)
    _, sol = _shoot_wind(c, Lam, ge, he, W, l)
    r, f, fp = sol.t, sol.y[0], sol.y[1]
    cut = int(np.argmin(np.abs(f)))
    r, f, fp = r[:cut + 1], f[:cut + 1], fp[:cut + 1]
    dens = (0.5 * fp ** 2 + 0.5 * W * (f / np.maximum(r, 1e-9)) ** 2
            + 0.5 * Lam * f ** 2 - 0.25 * ge * f ** 4 + he * f ** 6 / 6.0)
    E = 4 * math.pi * float(np.trapezoid(dens * r ** 2, r))
    Q = 4 * math.pi * float(np.trapezoid(f ** 2 * r ** 2, r))
    return {"E": E, "Q": Q}


def select_wind(Lam, l, g=1.0, h=1.0, ngrid=21):
    lams = np.linspace(0.0, 1.0, ngrid)
    Es = []
    for lp in lams:
        r = sc_energy_wind(Lam, lp, l, g, h)
        Es.append(r["E"] if r else np.nan)
    Es = np.array(Es)
    if np.isfinite(Es).sum() < 3:
        return None
    return {"lam_star": float(lams[int(np.nanargmin(Es))]), "E": Es, "lams": lams}


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def report():
    print("=" * 78)
    print("САМОСОГЛАСОВАННЫЙ ОТБОР ПОЛЯРИЗАЦИИ (T-36, третья развязка)")
    print("=" * 78)

    print("\nPart A. Профиль пересчитан по lam_pol (эффективные связи g*P4, h*P6).")
    print("  Сравнение с несамосогласованным (VectorPolarization.py): там линейная")
    print("  при Lam<=0.08, круговая при Lam=0.10.\n")
    print(f"  {'Lam':>6} {'lam* (мин E)':>12} {'lam* (мин E/Q)':>14} {'выбор E':>9} {'E(лин)':>9} {'E(круг)':>9}")
    thr_E = None
    prev = None
    for Lam in (0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12):
        s = select_selfconsistent(Lam)
        if s is None:
            print(f"  {Lam:>6.3f}   ветвь/окно не найдено")
            continue
        ch = "линейная" if s["lam_star_E"] < 0.05 else ("круговая" if s["lam_star_E"] > 0.95 else f"эллипт.{s['lam_star_E']:.2f}")
        Ec = s["E"][np.isfinite(s["E"])][-1]
        print(f"  {Lam:>6.3f} {s['lam_star_E']:>12.3f} {s['lam_star_EbyQ']:>14.3f} {ch:>9} {s['E_lin']:>9.4f} {Ec:>9.4f}")
        if prev is not None and prev < 0.05 <= 999 and s["lam_star_E"] > 0.95 and thr_E is None:
            thr_E = Lam
        prev = s["lam_star_E"]

    print("\nPart A — расклад u/d. omega_d=omega_u/2 => Lam_d>Lam_u (d ближе к тонкой стенке).")
    print("  Берём пару (Lam_u, Lam_d) и смотрим поляризацию каждой при самосогласии:")
    print(f"  {'Lam_u':>7} {'Lam_d':>7} {'pol(u)':>10} {'pol(d)':>10} {'вывод':>28}")
    for Lam_u, Lam_d in ((0.03, 0.06), (0.04, 0.08), (0.05, 0.10), (0.055, 0.11)):
        su = select_selfconsistent(Lam_u); sd = select_selfconsistent(Lam_d)
        if su is None or sd is None:
            print(f"  {Lam_u:>7.3f} {Lam_d:>7.3f}   окно не найдено")
            continue
        pu = "круговая" if su["lam_star_E"] > 0.5 else "линейная"
        pd = "круговая" if sd["lam_star_E"] > 0.5 else "линейная"
        if pu == "линейная" and pd == "круговая":
            verdict = "ИНВЕРСИЯ (d=2/3,u=1/3)"
        elif pu == "круговая" and pd == "линейная":
            verdict = "как в СМ (u=2/3,d=1/3)"
        else:
            verdict = f"обе {pu}"
        print(f"  {Lam_u:>7.3f} {Lam_d:>7.3f} {pu:>10} {pd:>10} {verdict:>28}")

    import sys; print("\nPart B. Намотка как центробежный барьер W=l(l+1) (первый срез, сфера).")
    print("  Порог перехода линейная->круговая при разных l:")
    print(f"  {'l':>5} {'W':>6} {'Lam порога (круговая при Lam>порога)':>40}")
    for l in (0.0, 0.5, 1.0):
        thr = None
        for Lam in np.linspace(0.02, 0.16, 15):
            s = select_wind(float(Lam), l)
            if s is None:
                continue
            if s["lam_star"] > 0.5:
                thr = float(Lam); break
        print(f"  {l:>5.1f} {l*(l+1):>6.2f} {('~%.3f'%thr) if thr else 'не достигнут в окне':>40}")
    print("\n  Оговорка: центробежный барьер в сферической редукции — не вихрь в")
    print("  полоидальной плоскости тора; при l>0 f(0)=0 (кольцо), моменты меняются.")
    print("  Это срез чувствительности, а не тороидальный расчёт.")


if __name__ == "__main__":
    report()
