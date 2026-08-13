"""Расплывание фотонного пакета на космологических расстояниях.

ЧТО ИСПРАВЛЕНО ПО СРАВНЕНИЮ СО СТАРОЙ ВЕРСИЕЙ (CODE_REVIEW.md §2.2)
-------------------------------------------------------------------
1. В формуле дисперсионного расплывания был потерян КВАДРАТ:

       было:  tau_0*(1+z) * sqrt(1 +  b2*L/(tau_0*(1+z))**2 )
       надо:  tau_0*(1+z) * sqrt(1 + (b2*L/(tau_0*(1+z))**2)**2)

   Без квадрата подкоренное выражение при b2 < 0 могло уйти в минус (nan),
   а зависимость была линейной вместо квадратичной.

2. omega_p = 1e-13 рад/с было подписано как "реалистичное значение".
   Реальная плазменная частота диффузной межгалактической среды при
   n_e ~ 2e-7 см^-3 равна ~24 рад/с, то есть на 15 ПОРЯДКОВ больше.
   С прежним числом b2*L/tau^2 ~ 1e-25, и эффект был не виден вообще:
   третий столбик диаграммы численно дублировал второй.

3. L = 9.46e24 м (1 млрд св. лет) и z = 1 противоречили друг другу.
   В плоской LCDM 1 млрд св. лет — это z = 0.070, а z = 1 — это 11.1 млрд
   св. лет. Теперь задаётся только z, а расстояние из него интегрируется.

4. n_e не постоянна: она растёт как (1+z)^3, а локальная частота фотона —
   как (1+z). Поэтому вместо b2*L считается честный интеграл по лучу
   (см. gdd() ниже).

5. Три столбика в линейном масштабе при разбросе в 4 порядка заменены на
   график tau(z) в логарифмическом масштабе.

ГРАНИЦЫ ПРИМЕНИМОСТИ
--------------------
* Учтена ТОЛЬКО дисперсия на средней плотности IGM. Рассеяние на
  турбулентных неоднородностях (которое на практике доминирует в уширении
  быстрых радиовсплесков) здесь не рассматривается.
* Разложение до beta_2 предполагает малую относительную ширину спектра.
  Для tau_0 = 1 фс на 1 мкм она порядка единицы — скрипт это проверяет
  и печатает предупреждение.

Запуск:
    python photon_broadening.py
    python photon_broadening.py --tau0 1e-12 --lambda-um 0.5 --no-plot
"""

import argparse
import math

import numpy as np

# --- Физические константы (CODATA 2018) ---
C_LIGHT = 2.99792458e8        # м/с
E_CHARGE = 1.602176634e-19    # Кл
EPS0 = 8.8541878128e-12       # Ф/м
M_E = 9.1093837015e-31        # кг
M_P = 1.67262192369e-27       # кг
G_NEWTON = 6.67430e-11        # м^3/(кг с^2)

# --- Космология (Planck 2018, плоская LCDM) ---
H0_KM_S_MPC = 67.4
MPC = 3.0856775814913673e22   # м
H0 = H0_KM_S_MPC * 1e3 / MPC  # 1/с
OMEGA_M = 0.315
OMEGA_L = 0.685
OMEGA_B = 0.0493
Y_HE = 0.2453                 # массовая доля гелия
F_IGM = 0.84                  # доля барионов в диффузной IGM

LIGHT_YEAR = 9.4607304725808e15   # м


def electron_density_today() -> float:
    """Средняя концентрация свободных электронов IGM сегодня, 1/м^3.

    Полностью ионизованные H и He дают (1 - Y/2) электронов на массу m_p.
    """
    rho_crit = 3.0 * H0 ** 2 / (8.0 * math.pi * G_NEWTON)
    rho_b = OMEGA_B * rho_crit
    return F_IGM * rho_b * (1.0 - Y_HE / 2.0) / M_P


N_E0 = electron_density_today()

#: Плазменная частота IGM сегодня, рад/с. Ср. со старым значением 1e-13.
OMEGA_P0 = math.sqrt(N_E0 * E_CHARGE ** 2 / (EPS0 * M_E))


def hubble_e(z):
    """E(z) = H(z)/H0 для плоской LCDM."""
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def _integrate(f, z_max: float, n: int = 4000) -> float:
    """Интеграл по z методом Симпсона (n чётное)."""
    n = n + (n % 2)
    z = np.linspace(0.0, z_max, n + 1)
    y = f(z)
    h = z[1] - z[0]
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def comoving_distance(z: float) -> float:
    """Сопутствующее расстояние, м."""
    if z <= 0:
        return 0.0
    return C_LIGHT / H0 * _integrate(lambda zz: 1.0 / hubble_e(zz), z)


def gdd(z: float, omega_obs: float) -> float:
    """Накопленная дисперсия групповой задержки (GDD), с^2.

    Локально beta_2 = -omega_p^2 / (c * omega^3). Вдоль луча
    omega(z) = omega_obs*(1+z), n_e(z) = n_e0*(1+z)^3, а собственная длина
    элемента пути dl = c*dz / ((1+z)*H(z)). Фаза — инвариант вдоль луча,
    поэтому дифференцируем накопленную фазу по НАБЛЮДАЕМОЙ частоте:

        d2Phi/domega_obs^2 = -(omega_p0^2 / omega_obs^3) * int_0^z (1+z')/H(z') dz'

    При z -> 0 это переходит в обычное -omega_p^2 * L / (c * omega^3).
    """
    if z <= 0:
        return 0.0
    integral = _integrate(lambda zz: (1.0 + zz) / hubble_e(zz), z) / H0
    return -(OMEGA_P0 ** 2 / omega_obs ** 3) * integral


def durations(z: float, tau0: float, omega_obs: float) -> tuple:
    """Возвращает (tau_cosmo, tau_total) в секундах.

    tau_cosmo — только космологическое растяжение,
    tau_total — растяжение плюс дисперсионное расплывание (формула с квадратом).
    """
    tau_cosmo = tau0 * (1.0 + z)
    d = gdd(z, omega_obs)
    tau_total = tau_cosmo * math.sqrt(1.0 + (d / tau_cosmo ** 2) ** 2)
    return tau_cosmo, tau_total


def bandwidth_check(tau0: float, omega_obs: float) -> tuple:
    """Относительная ширина спектра преобразованно-ограниченного гауссова импульса.

    Для гауссова импульса произведение ширин по половине высоты
    dnu * dtau = 0.4413. Разложение до beta_2 корректно лишь при dnu/nu << 1.
    """
    nu0 = omega_obs / (2.0 * math.pi)
    d_nu = 0.4413 / tau0
    return d_nu / nu0, nu0


# --------------------------------------------------------------------------

def report(z_list, tau0: float, lambda_m: float):
    omega_obs = 2.0 * math.pi * C_LIGHT / lambda_m

    print("=" * 76)
    print("РАСПЛЫВАНИЕ ФОТОННОГО ПАКЕТА В МЕЖГАЛАКТИЧЕСКОЙ ПЛАЗМЕ")
    print("=" * 76)
    print(f"Длина волны (набл.)     : {lambda_m * 1e6:.4g} мкм")
    print(f"omega_obs               : {omega_obs:.4e} рад/с")
    print(f"Исходная длительность   : {tau0:.3e} с")
    print(f"n_e(0) в IGM            : {N_E0:.3e} 1/м^3 = {N_E0 * 1e-6:.3e} см^-3")
    print(f"omega_p(0)              : {OMEGA_P0:.3f} рад/с "
          f"(в старой версии стояло 1.0e-13)")

    frac_bw, nu0 = bandwidth_check(tau0, omega_obs)
    print(f"Относит. ширина спектра : {frac_bw:.3g}")
    if frac_bw > 0.1:
        print()
        print("  ВНИМАНИЕ: разложение дисперсии до beta_2 предполагает узкий спектр,")
        print(f"  а здесь d(nu)/nu = {frac_bw:.2f}. Числа ниже — оценка по порядку")
        print("  величины, а не точный расчёт. Для корректного результата возьмите")
        print(f"  tau_0 >> {0.4413 / nu0 / 0.1:.1e} с (ключ --tau0).")
    print()

    print(f"{'z':>8}{'D_C, Гсв.лет':>15}{'GDD, с^2':>14}"
          f"{'только z':>13}{'z + плазма':>13}{'во сколько':>12}")
    print("-" * 76)
    for z in z_list:
        tc, tt = durations(z, tau0, omega_obs)
        d_c = comoving_distance(z) / LIGHT_YEAR / 1e9
        print(f"{z:>8.3g}{d_c:>15.3f}{gdd(z, omega_obs):>14.3e}"
              f"{tc:>13.3e}{tt:>13.3e}{tt / tc:>12.3g}")
    print()

    z_1gly = 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):                      # деление пополам
        mid = 0.5 * (lo + hi)
        if comoving_distance(mid) < 1e9 * LIGHT_YEAR:
            lo = mid
        else:
            hi = mid
    z_1gly = 0.5 * (lo + hi)
    print(f"Справка: 1 млрд св. лет соответствует z = {z_1gly:.4f}, "
          f"а z = 1 — это {comoving_distance(1.0) / LIGHT_YEAR / 1e9:.2f} млрд св. лет.")
    print("В старой версии L и z были заданы независимо и противоречили друг другу.")


def make_plot(tau0: float, lambda_m: float, theme: str = "light", out_path=None):
    import matplotlib.pyplot as plt
    from lib.theme import THEMES, style_axes, legend

    t = THEMES[theme]
    omega_obs = 2.0 * math.pi * C_LIGHT / lambda_m

    z = np.logspace(-2, 1, 200)
    tau_cosmo = np.array([durations(zz, tau0, omega_obs)[0] for zz in z])
    tau_total = np.array([durations(zz, tau0, omega_obs)[1] for zz in z])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=t["surface"])

    ax.axhline(tau0, color=t["axis"], linestyle=(0, (4, 4)), linewidth=1.2, zorder=1)
    ax.annotate(f"исходная длительность {tau0:.0e} с", xy=(1.05e-2, tau0),
                xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=t["muted"], va="bottom")

    ax.plot(z, tau_cosmo, color=t["s1"], lw=2.0, zorder=3,
            label="только космологическое растяжение")
    ax.plot(z, tau_total, color=t["s2"], lw=2.0, zorder=4,
            label="растяжение + дисперсия в плазме IGM")

    # прямая подпись на конце каждой кривой — идентичность не только цветом
    ax.annotate("+ плазма", xy=(z[-1], tau_total[-1]), xytext=(6, 0),
                textcoords="offset points", color=t["s2"], fontsize=9.5, va="center")
    ax.annotate("только z", xy=(z[-1], tau_cosmo[-1]), xytext=(6, 0),
                textcoords="offset points", color=t["s1"], fontsize=9.5, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 1.4e1)
    style_axes(ax, t,
               xlabel="Красное смещение z",
               ylabel="Длительность импульса, с",
               title=f"Расплывание импульса {tau0:.0e} с на "
                     f"{lambda_m * 1e6:.3g} мкм в межгалактической плазме")
    legend(ax, t, loc="upper left")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=t["surface"])
        print(f"\nГрафик сохранён: {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Расплывание фотонного пакета в межгалактической плазме")
    p.add_argument("--tau0", type=float, default=1e-15,
                   help="исходная длительность, с (по умолчанию 1e-15)")
    p.add_argument("--lambda-um", type=float, default=1.0,
                   help="наблюдаемая длина волны, мкм (по умолчанию 1.0)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--theme", choices=("light", "dark"), default="light")
    p.add_argument("--save", metavar="PATH")
    args = p.parse_args()

    lambda_m = args.lambda_um * 1e-6
    report([0.07, 0.5, 1.0, 2.0, 5.0], args.tau0, lambda_m)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        make_plot(args.tau0, lambda_m, theme=args.theme, out_path=args.save)
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
