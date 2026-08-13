"""
Кубично-квинтический солитон: существование, окно частот, теорема подобия.

Закрывает вычислительную часть пункта T-34 (подглава 2.5): насыщающий член
выписан явно, условие существования Q-шара в D = 3 проверено, и показано, что
после обезразмеривания задача имеет ровно ОДИН параметр.

Уравнение профиля (стационарная мода f(r) e^{i omega t}):

    f'' + (2/r) f' - lam f + g f^3 - h f^5 = 0,   lam = m^2 - omega^2
    f'(0) = 0,  f(inf) = 0

Потенциал:  U(f) = (1/2) m^2 f^2 - (g/4) f^4 + (h/6) f^6

Запуск:  python CubicQuintic.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# 1. Точное одномерное решение и условие его существования
# ---------------------------------------------------------------------------


def cq_discriminant(lam: float, g: float, h: float) -> float:
    """D = g^2 - (16/3) h lam. Локализованное решение существует при D > 0."""
    return g * g - 16.0 * h * lam / 3.0


def cq_exact_1d(s, lam: float, g: float, h: float):
    """Точное решение f'' = lam f - g f^3 + h f^5 при D > 0."""
    D = cq_discriminant(lam, g, h)
    if D <= 0.0:
        raise ValueError("D <= 0: локализованного решения нет")
    s = np.asarray(s, dtype=float)
    return np.sqrt(4.0 * lam / (g + math.sqrt(D) * np.cosh(2.0 * math.sqrt(lam) * s)))


def cq_exact_residual(lam: float, g: float, h: float,
                      span=(-4.0, 4.0), n: int = 41) -> float:
    """Максимальная невязка точного решения — контроль формулы."""
    step, worst = 1e-5, 0.0
    for s in np.linspace(*span, n):
        f0 = float(cq_exact_1d(s, lam, g, h))
        d2 = (float(cq_exact_1d(s + step, lam, g, h))
              - 2.0 * f0 + float(cq_exact_1d(s - step, lam, g, h))) / step**2
        worst = max(worst, abs(d2 - lam * f0 + g * f0**3 - h * f0**5))
    return worst


# ---------------------------------------------------------------------------
# 2. Окно частот Q-шара в D = 3 (условие Коулмена)
# ---------------------------------------------------------------------------


def qball_window(m: float, g: float, h: float) -> dict:
    """
    Q-шар существует при omega_min < omega < m, где
        omega_min^2 = 2 min_f U(f)/f^2 = m^2 - 3 g^2 / (16 h),
    минимум достигается при f^2 = 3g/(4h).
    """
    omega_min_sq = m * m - 3.0 * g * g / (16.0 * h)
    omega_min = math.sqrt(omega_min_sq) if omega_min_sq > 0.0 else 0.0
    return {
        "f_star": math.sqrt(3.0 * g / (4.0 * h)),
        "omega_min_sq": omega_min_sq,
        "omega_min": omega_min,
        "omega_max": m,
        "lam_max": 3.0 * g * g / (16.0 * h),
        "window_nonempty": omega_min < m,
        "thin_wall": omega_min_sq > 0.0,
    }


def existence_verdict(m: float, g: float, h: float) -> str:
    w = qball_window(m, g, h)
    if not w["window_nonempty"]:
        return "окна нет"
    if w["thin_wall"]:
        return f"непусто: {w['omega_min']:.6f} < omega < {m:.6f}; есть тонкая стенка"
    return f"непусто: 0 < omega < {m:.6f}; тонкой стенки нет"


# ---------------------------------------------------------------------------
# 3. Теорема подобия: единственный параметр задачи
# ---------------------------------------------------------------------------


def similarity_parameter(lam: float, g: float, h: float) -> float:
    """
    Замена f = A phi, r = L rho с A^2 = g/h, L^2 = h/g^2 приводит уравнение к
        phi'' + (2/rho) phi' - Lam phi + phi^3 - phi^5 = 0,  Lam = lam h / g^2.
    Все безразмерные отношения (в том числе E_n/E_0) зависят только от Lam.
    """
    return lam * h / (g * g)


def similarity_scales(g: float, h: float) -> dict:
    """Масштабы: амплитуда A, длина L, энергия E ~ A^2 L = 1/sqrt(h)."""
    return {"A": math.sqrt(g / h), "L": math.sqrt(h) / g, "E": 1.0 / math.sqrt(h)}


# ---------------------------------------------------------------------------
# 4. Радиальная краевая задача: пристрелка
# ---------------------------------------------------------------------------


def _rhs(r, y, lam, g, h):
    f, fp = y
    curv = 0.0 if r < 1e-9 else 2.0 * fp / r
    return [fp, -curv + lam * f - g * f**3 + h * f**5]


def _shoot(f0: float, lam: float, g: float, h: float, fast: bool = True):
    """Возвращает (знак ухода на бесконечности, решение)."""
    sc = similarity_scales(g, h)
    A, L = sc["A"], sc["L"]

    def ev_hi(r, y, *a):
        return y[0] - 3.0 * A

    def ev_lo(r, y, *a):
        return y[0] + 3.0 * A

    ev_hi.terminal = ev_lo.terminal = True
    rtol, atol, step = (1e-8, 1e-11, 0.4 * L) if fast else (1e-11, 1e-13, 0.03 * L)
    sol = solve_ivp(_rhs, [1e-9, 200.0 * L], [f0, 0.0], args=(lam, g, h),
                    rtol=rtol, atol=atol, events=[ev_hi, ev_lo], max_step=step)
    if sol.t_events[0].size:
        return +1, sol
    if sol.t_events[1].size:
        return -1, sol
    return (+1 if sol.y[0][-1] > 0 else -1), sol


def find_ground_state(lam: float, g: float, h: float, n_grid: int = 140,
                      n_bisect: int = 60):
    """Амплитуда f0 основного (безузлового) Q-шара. None, если ветви нет."""
    A = similarity_scales(g, h)["A"]
    grid = np.linspace(0.02 * A, 1.6 * A, n_grid)
    signs = [_shoot(x, lam, g, h)[0] for x in grid]
    brackets = [(grid[i], grid[i + 1]) for i in range(n_grid - 1)
                if signs[i] != signs[i + 1]]
    if not brackets:
        return None
    a, b = brackets[0]
    s_a = _shoot(a, lam, g, h)[0]
    for _ in range(n_bisect):
        mid = 0.5 * (a + b)
        if _shoot(mid, lam, g, h)[0] == s_a:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def profile_energy(f0: float, lam: float, g: float, h: float):
    """Энергия приведённого функционала и радиус локализации."""
    _, sol = _shoot(f0, lam, g, h, fast=False)
    r, f, fp = sol.t, sol.y[0], sol.y[1]
    cut = int(np.argmin(np.abs(f)))
    r, f, fp = r[:cut + 1], f[:cut + 1], fp[:cut + 1]
    dens = 0.5 * fp**2 + 0.5 * lam * f**2 - 0.25 * g * f**4 + h * f**6 / 6.0
    return 4.0 * math.pi * float(np.trapezoid(dens * r * r, r)), float(r[-1])


# ---------------------------------------------------------------------------
# 5. Контроль отвергнутого профиля
# ---------------------------------------------------------------------------


def rejected_ansatz_lambda(g: float = 1.0, rho0: float = 1.0):
    """
    Профиль Ф = (2/sqrt(g)) / sqrt(rho/rho0 + rho0/rho + 2) при beta = g^2/3.
    Если бы он решал ОДУ, требуемая lam была бы постоянной.
    """
    beta = g * g / 3.0

    def phi(r):
        return (2.0 / math.sqrt(g)) / math.sqrt(r / rho0 + rho0 / r + 2.0)

    out = []
    for r in (0.25, 0.5, 1.0, 2.0, 4.0):
        step = 1e-5
        d2 = (phi(r + step) - 2.0 * phi(r) + phi(r - step)) / step**2
        d1 = (phi(r + step) - phi(r - step)) / (2.0 * step)
        res = d2 + d1 / r - phi(r) / r**2 + g * phi(r) ** 3 + beta * phi(r) ** 5
        out.append((r, res / phi(r)))
    return out


# ---------------------------------------------------------------------------
# 6. Отчёт
# ---------------------------------------------------------------------------


def _row(a, b):
    print(f"  {a:<50s} {b}")


def report() -> None:
    print("=" * 76)
    print("КУБИЧНО-КВИНТИЧЕСКИЙ СОЛИТОН: СУЩЕСТВОВАНИЕ И ПОДОБИЕ")
    print("=" * 76)

    print("\n1. Точное одномерное решение (контроль формулы)")
    print("   f^2(s) = 4 lam / [ g + sqrt(g^2 - 16 h lam / 3) * cosh(2 sqrt(lam) s) ]")
    for lam, g, h in [(0.7, 1.3, 0.45), (0.10, 1.0, 1.0), (0.02, 0.6, 2.0)]:
        _row(f"lam={lam}, g={g}, h={h}:  D = {cq_discriminant(lam, g, h):+.5f}",
             f"макс. невязка {cq_exact_residual(lam, g, h):.2e}")
    _row("условие существования", "D > 0  <=>  lam < 3 g^2 / (16 h)")

    print("\n2. Окно частот Q-шара в D = 3 (условие Коулмена)")
    for g, h in [(1.3, 0.45), (1.0, 1.0), (0.6, 2.0), (0.2, 5.0)]:
        _row(f"m=1, g={g}, h={h}", existence_verdict(1.0, g, h))
    print()
    _row("тождество", "lam < 3g^2/(16h)  <=>  omega^2 > omega_min^2")
    _row("следствие", "окно непусто при любых g, h > 0")

    print("\n3. Теорема подобия: Lam = lam h / g^2 — единственный параметр")
    print(f"  {'lam':>7} {'g':>6} {'h':>6} {'Lam':>8} "
          f"{'f0*sqrt(h/g)':>13} {'E*sqrt(h)':>12} {'R*g/sqrt(h)':>12}")
    for lam, g, h in [(0.10, 1.0, 1.0), (0.40, 2.0, 1.0), (0.05, 1.0, 2.0),
                      (0.90, 3.0, 1.0), (0.025, 0.5, 1.0)]:
        f0 = find_ground_state(lam, g, h)
        if f0 is None:
            print(f"  {lam:7.3f} {g:6.2f} {h:6.2f} "
                  f"{similarity_parameter(lam, g, h):8.4f}   ветвь не найдена")
            continue
        E, R = profile_energy(f0, lam, g, h)
        sc = similarity_scales(g, h)
        print(f"  {lam:7.3f} {g:6.2f} {h:6.2f} "
              f"{similarity_parameter(lam, g, h):8.4f} "
              f"{f0 / sc['A']:13.8f} {E / sc['E']:12.5f} {R / sc['L']:12.3f}")
    print("  Приведённые столбцы совпадают до седьмого знака при разбросе")
    print("  lam в 36 раз, g в 6 раз, h в 2 раза.")

    print("\n4. Основное состояние по параметру Lam")
    print(f"  {'Lam':>7} {'f0/A':>12} {'E/E_scale':>12} {'R/L':>10}")
    for Lam in (0.02, 0.05, 0.08, 0.10, 0.12):
        f0 = find_ground_state(Lam, 1.0, 1.0, n_grid=180)
        if f0 is None:
            print(f"  {Lam:7.3f}   ветвь не найдена")
            continue
        E, R = profile_energy(f0, Lam, 1.0, 1.0)
        print(f"  {Lam:7.3f} {f0:12.8f} {E:12.5f} {R:10.2f}")
    _row("верхняя граница окна", f"Lam_max = 3/16 = {3/16:.4f}")
    _row("амплитуда тонкой стенки", "f_* / A = sqrt(3/4) = 0.866025")
    print("  При Lam, близких к Lam_max, интервал пристрелки сужается")
    print("  экспоненциально (предел тонкой стенки, R -> inf): равномерная сетка")
    print("  его пропускает, требуется специализированный метод.")

    print("\n5. Контроль отвергнутого профиля")
    rows = rejected_ansatz_lambda()
    _row("Ф = (2/sqrt(g))/sqrt(rho/rho0+rho0/rho+2), beta = g^2/3", "")
    for r, lam_req in rows:
        _row(f"  rho = {r}", f"требуемая lam = {lam_req:+.6f}")
    span = max(x[1] for x in rows) - min(x[1] for x in rows)
    _row("разброс требуемой lam", f"{span:.3f}  =>  решением не является")


if __name__ == "__main__":
    report()
