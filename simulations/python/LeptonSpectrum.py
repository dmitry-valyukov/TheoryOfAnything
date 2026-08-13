"""
LeptonSpectrum.py — спектр масс на полной сетке частиц поляризации вакуума.

Модель (theory/04-3, theory/04-5): стационарные моды тороидального вакуумного
солитона отвечают корням спектрального уравнения для конической функции Лежандра

    P_{-1/2 + i*tau_n}(cosh eta_eff) = 0

Масса моды выражается через частоту резонанса и интеграл перекрытия фаз:

    m_n  ~  tau_n^3 * Gamma_n,     Gamma_n = int_0^eta_eff |P|^2 sinh(eta) d(eta)

Настоящий модуль:
  1. вычисляет P_{-1/2+i*tau}(cosh eta) двумя независимыми способами
     (интеграл Мелера — Дирихле и прямое интегрирование ОДУ) и сверяет их;
  2. находит корни tau_n;
  3. считает Gamma_n;
  4. строит лестницу масс, нормируя её на САМУЮ ТЯЖЁЛУЮ частицу сетки
     (t-кварк), а не на электрон;
  5. сопоставляет с полной сеткой частиц, дающих вклад в поляризацию вакуума;
  6. решает обратную задачу: какое eta_eff потребовалось бы, чтобы лестница села
     на наблюдаемые массы.

Запуск:  python LeptonSpectrum.py
"""

import warnings

import numpy as np
from scipy.integrate import quad, solve_ivp, IntegrationWarning
from scipy.optimize import brentq
from scipy.special import j0

# ----------------------------------------------------------------------------
# 1. Геометрия каустики (theory/04-2)
# ----------------------------------------------------------------------------

ALPHA_0 = 1.0 / (48.0 * np.pi)                 # топологический голый заряд
COSH_ETA_0 = (96.0 * np.pi ** 2) ** 0.25       # конформная фиксация тонкости тора
ETA_0 = np.arccosh(COSH_ETA_0)
SCHWINGER = 1.0 - ALPHA_0 / (2.0 * np.pi)      # швингеровская деформация
COSH_ETA_EFF = COSH_ETA_0 * SCHWINGER
ETA_EFF = np.arccosh(COSH_ETA_EFF)


# ----------------------------------------------------------------------------
# 2. Коническая функция Лежандра P_{-1/2 + i*tau}(cosh eta)
# ----------------------------------------------------------------------------

def P_mehler(tau, eta):
    """Интеграл Мелера — Дирихле.

        P_{-1/2+i*tau}(cosh eta) = (2/pi) * int_0^eta cos(tau*t) dt
                                              / sqrt(2*(cosh eta - cosh t))

    Корневая особенность на верхнем пределе снимается заменой t = eta - u^2.
    """
    if eta <= 0.0:
        return 1.0
    ch = np.cosh(eta)

    def integrand(u):
        t = eta - u * u
        d = 2.0 * (ch - np.cosh(t))
        if d <= 0.0:
            # предел u -> 0: cosh eta - cosh(eta-u^2) ~ sinh(eta)*u^2
            return 2.0 / np.sqrt(2.0 * np.sinh(eta)) * np.cos(tau * eta)
        return 2.0 * u * np.cos(tau * t) / np.sqrt(d)

    with warnings.catch_warnings():
        # Подынтегральная функция быстро осциллирует при больших tau;
        # предупреждения о числе разбиений здесь неинформативны — точность
        # контролируется сверкой с независимым методом (cross_check).
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(integrand, 0.0, np.sqrt(eta), limit=800)
    return (2.0 / np.pi) * val


def P_ode(tau, eta_max, n_out=400):
    """Прямое интегрирование уравнения Лежандра в переменной eta:

        y'' + coth(eta) * y' + (1/4 + tau^2) * y = 0,   y(0)=1, y'(0)=0

    Вблизи нуля coth(eta) ~ 1/eta, и решение выходит на J_0(sqrt(1/4+tau^2)*eta).
    Возвращает (eta_grid, y_grid).
    """
    lam = 0.25 + tau * tau
    k = np.sqrt(lam)
    eps = 1e-6
    y0 = [j0(k * eps), -k * 0.5 * k * eps]  # J_0'(x) = -J_1(x) ~ -x/2

    def rhs(e, y):
        return [y[1], -(1.0 / np.tanh(e)) * y[1] - lam * y[0]]

    grid = np.linspace(eps, eta_max, n_out)
    sol = solve_ivp(rhs, (eps, eta_max), y0, t_eval=grid,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    return sol.t, sol.y[0]


def P_at(tau, eta):
    """P в одной точке. Используется интеграл Мелера — Дирихле: он на два
    порядка быстрее интегрирования ОДУ и не теряет точности при больших tau,
    где решение сильно осциллирует."""
    return P_mehler(tau, eta)


def cross_check(eta):
    """Сверка двух независимых методов."""
    print(f"\nСверка методов при eta = {eta:.6f}")
    print(f"{'tau':>8} {'Мелер-Дирихле':>16} {'ОДУ':>16} {'|разность|':>12}")
    for tau in (0.5, 1.0, np.sqrt(2.0), 3.0, 8.31211, 22.14618):
        a = P_mehler(tau, eta)
        _, yy = P_ode(tau, eta, n_out=2)
        b = float(yy[-1])
        print(f"{tau:8.5f} {a:16.10f} {b:16.10f} {abs(a - b):12.2e}")


# ----------------------------------------------------------------------------
# 3. Корни спектрального уравнения и интегралы перекрытия
# ----------------------------------------------------------------------------

def find_roots(eta, n_roots=80, step=None):
    """Корни P_{-1/2+i*tau}(cosh eta) = 0 по tau, снизу вверх.

    Корни асимптотически эквидистантны с шагом pi/eta, поэтому сначала идёт
    грубое сканирование с шагом заведомо меньше половины периода, затем
    уточнение методом Брента на каждом интервале смены знака.
    """
    period = np.pi / eta
    if step is None:
        step = period / 6.0
    tau_max = (n_roots + 2) * period
    taus = np.arange(1e-3, tau_max, step)
    vals = np.array([P_at(t, eta) for t in taus])
    roots = []
    sign_change = np.where(vals[:-1] * vals[1:] < 0.0)[0]
    for i in sign_change:
        roots.append(brentq(lambda t: P_at(t, eta), taus[i], taus[i + 1],
                            xtol=1e-12, rtol=1e-14))
        if len(roots) >= n_roots:
            break
    return np.array(roots)


def gamma_int(tau, eta, n=None):
    """Интеграл перекрытия фаз Gamma = int_0^eta |P|^2 sinh(eta') d(eta').

    Число узлов подбирается так, чтобы на период осцилляции решения
    (примерно pi/tau по eta) приходилось не менее 20 точек.
    """
    if n is None:
        n = max(400, int(40 * tau * eta / np.pi) + 400)
    grid = np.linspace(0.0, eta, n)
    y = np.array([P_mehler(tau, e) for e in grid])
    return float(np.trapezoid(y * y * np.sinh(grid), grid))


# ----------------------------------------------------------------------------
# 4. Сетка частиц поляризации вакуума
# ----------------------------------------------------------------------------
# Все частицы, дающие вклад в поляризацию вакуума (бегущую константу связи).
# Массы: PDG, МэВ. Для кварков — токовые массы в схеме MS-bar.

VACUUM_POLARIZATION_GRID = [
    # (символ, название, масса МэВ, тип, заряд)
    ("e",  "электрон",   0.51099895, "лептон", -1.0),
    ("u",  "u-кварк",    2.16,       "кварк",  +2 / 3),
    ("d",  "d-кварк",    4.67,       "кварк",  -1 / 3),
    ("s",  "s-кварк",    93.4,       "кварк",  -1 / 3),
    ("mu", "мюон",       105.6583755, "лептон", -1.0),
    ("c",  "c-кварк",    1270.0,     "кварк",  +2 / 3),
    ("tau", "тау-лептон", 1776.86,   "лептон", -1.0),
    ("b",  "b-кварк",    4180.0,     "кварк",  -1 / 3),
    ("W",  "W-бозон",    80377.0,    "бозон",  -1.0),
    ("Z",  "Z-бозон",    91187.6,    "бозон",   0.0),
    ("H",  "бозон Хиггса", 125250.0, "бозон",   0.0),
    ("t",  "t-кварк",    172690.0,   "кварк",  +2 / 3),
]

CHARGED_LEPTONS = ["e", "mu", "tau"]


def grid_sorted_desc():
    return sorted(VACUUM_POLARIZATION_GRID, key=lambda r: -r[2])


# ----------------------------------------------------------------------------
# 5. Лестница масс, нормированная сверху
# ----------------------------------------------------------------------------

def mass_ladder(taus, gammas, anchor_index, anchor_mass):
    """m_n = anchor_mass * (tau_n/tau_a)^3 * (Gamma_n/Gamma_a)."""
    ta, ga = taus[anchor_index], gammas[anchor_index]
    return anchor_mass * (taus / ta) ** 3 * (gammas / ga)


def assign_from_top(taus, gammas, grid_desc):
    """Жадная привязка: самая тяжёлая частица — к самому высокому
    использованному корню, дальше вниз по лестнице ищем ближайший корень."""
    heaviest = grid_desc[0]
    n_top = len(taus) - 1
    ladder = mass_ladder(taus, gammas, n_top, heaviest[2])
    out = []
    for sym, name, m_obs, kind, q in grid_desc:
        idx = int(np.argmin(np.abs(np.log(ladder) - np.log(m_obs))))
        m_pred = ladder[idx]
        out.append((sym, name, kind, m_obs, idx + 1, taus[idx], m_pred,
                    100.0 * (m_pred - m_obs) / m_obs))
    return ladder, out


# ----------------------------------------------------------------------------
# 6. Обратная задача: какое eta даёт наблюдаемое отношение масс
# ----------------------------------------------------------------------------

def required_ratio_top_to_electron():
    return VACUUM_POLARIZATION_GRID[-1][2] / VACUUM_POLARIZATION_GRID[0][2]


def asymptotic_level_count(eta):
    """Асимптотически tau_n ~ n*pi/eta, и при m ~ tau^3 (Gamma ~ const)
    отношение масс m_N/m_1 = N^3. Сколько уровней нужно на сетку?"""
    ratio = required_ratio_top_to_electron()
    return ratio ** (1.0 / 3.0)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def effective_exponent(taus, gammas):
    """Численно установить, как ведёт себя Gamma_n, и какой показатель p
    получается для эффективного закона m ~ tau^p.

    Полный закон модели: m ~ tau^3 * Gamma. Если Gamma ~ tau^(-q),
    то эффективно m ~ tau^(3-q).
    """
    lt, lg = np.log(taus[2:]), np.log(gammas[2:])
    q = -np.polyfit(lt, lg, 1)[0]
    return q, 3.0 - q


def null_test(taus, gammas, grid_desc, n_trials=2000, seed=20260812):
    """Контрольный тест: та же жадная привязка, но к СЛУЧАЙНОЙ лестнице
    с тем же числом уровней и тем же полным диапазоном.

    Если совпадения на реальной лестнице не лучше, чем на случайной, значит
    они порождены густотой уровней, а не физикой модели.
    """
    rng = np.random.default_rng(seed)
    real_ladder, real_assign = assign_from_top(taus, gammas, grid_desc)
    real_err = np.array([abs(a[7]) for a in real_assign])

    lo, hi = np.log(real_ladder[0]), np.log(real_ladder[-1])
    n_lv = len(real_ladder)
    obs = np.array([r[2] for r in grid_desc])

    worse = 0
    rnd_med = []
    for _ in range(n_trials):
        lv = np.sort(np.exp(rng.uniform(lo, hi, n_lv)))
        lv = lv * (real_ladder[-1] / lv[-1])          # та же верхняя привязка
        err = []
        for m in obs:
            j = int(np.argmin(np.abs(np.log(lv) - np.log(m))))
            err.append(abs(100.0 * (lv[j] - m) / m))
        err = np.array(err)
        rnd_med.append(np.median(err))
        if np.median(err) > np.median(real_err):
            worse += 1
    return real_err, np.array(rnd_med), worse / n_trials


def ladder_floor(taus, gammas, anchor_mass):
    """Наименьшая масса, достижимая на лестнице (уровень n = 1)."""
    ladder = mass_ladder(taus, gammas, len(taus) - 1, anchor_mass)
    return ladder[0]


def main():
    print("=" * 78)
    print("СПЕКТР МАСС НА СЕТКЕ ЧАСТИЦ ПОЛЯРИЗАЦИИ ВАКУУМА")
    print("=" * 78)
    print(f"alpha_0        = 1/(48*pi) = {ALPHA_0:.10f}")
    print(f"cosh eta_0     = (96*pi^2)^(1/4) = {COSH_ETA_0:.7f}")
    print(f"eta_0          = {ETA_0:.7f}")
    print(f"1 - alpha_0/2pi= {SCHWINGER:.9f}")
    print(f"cosh eta_eff   = {COSH_ETA_EFF:.7f}")
    print(f"eta_eff        = {ETA_EFF:.7f}")

    cross_check(ETA_EFF)

    print("\nПоиск корней спектрального уравнения ...")
    taus = find_roots(ETA_EFF, n_roots=80)
    print(f"найдено корней: {len(taus)}")
    print("первые 12:", np.array2string(taus[:12], precision=5))
    d = np.diff(taus)
    print(f"шаг между корнями: среднее {d.mean():.6f}, "
          f"теоретический pi/eta_eff = {np.pi / ETA_EFF:.6f}")

    print("\nПроверка значений, заявленных в исходных материалах:")
    for name, t in [("tau_1 = sqrt(2)", np.sqrt(2.0)),
                    ("tau_2 = 8.31211", 8.31211),
                    ("tau_3 = 22.14618", 22.14618)]:
        print(f"  P(tau) при {name:18s} = {P_at(t, ETA_EFF):+.6f}  "
              f"(корень => 0)")

    print("\nИнтегралы перекрытия для первых корней:")
    gammas = np.array([gamma_int(t, ETA_EFF) for t in taus])
    for i in range(min(8, len(taus))):
        print(f"  n={i+1:2d}  tau={taus[i]:9.5f}  Gamma={gammas[i]:.6f}")

    grid = grid_sorted_desc()
    ladder, assign = assign_from_top(taus, gammas, grid)

    print("\n" + "=" * 78)
    print("ПРИВЯЗКА СВЕРХУ: нормировка на t-кварк (самая тяжёлая частица сетки)")
    print("=" * 78)
    print(f"{'част.':>6} {'тип':>7} {'m_набл, МэВ':>13} {'n':>4} "
          f"{'tau_n':>9} {'m_мод, МэВ':>13} {'откл., %':>10}")
    for sym, name, kind, m_obs, n, tau, m_pred, err in assign:
        print(f"{sym:>6} {kind:>7} {m_obs:13.4f} {n:4d} {tau:9.4f} "
              f"{m_pred:13.4f} {err:+10.2f}")

    print("\n" + "=" * 78)
    print("ДИАГНОСТИКА")
    print("=" * 78)
    ratio = required_ratio_top_to_electron()
    print(f"Наблюдаемый диапазон m_t/m_e = {ratio:.4e}")
    print(f"При m ~ tau^3 и tau_n ~ n требуется N = {asymptotic_level_count(ETA_EFF):.1f}"
          f" уровней между электроном и t-кварком.")
    print(f"Плотность уровней при eta_eff = {ETA_EFF:.4f}: "
          f"шаг d(tau) = pi/eta = {np.pi/ETA_EFF:.4f}")
    print("Заполненными оказываются лишь единицы уровней из десятков — "
          "модель должна объяснить, почему остальные пусты.")

    q, p_eff = effective_exponent(taus, gammas)
    print(f"\nПоведение интеграла перекрытия: Gamma_n ~ tau^(-{q:.3f})")
    print(f"Следовательно, полный закон m ~ tau^3 * Gamma сводится к "
          f"m ~ tau^{p_eff:.3f}")
    print(f"а при tau_n ~ n*pi/eta — к m_n ~ n^{p_eff:.3f}")
    print("Это существенно: заявленное в исходниках масштабирование как куб "
          "частоты\nв действительности гасится интегралом перекрытия почти "
          "до квадрата.")

    floor = ladder_floor(taus, gammas, grid[0][2])
    print(f"\nНИЖНЯЯ ГРАНИЦА ЛЕСТНИЦЫ (уровень n=1): {floor:.3f} МэВ")
    print(f"Масса электрона: {VACUUM_POLARIZATION_GRID[0][2]:.5f} МэВ "
          f"— в {floor/VACUUM_POLARIZATION_GRID[0][2]:.0f} раз ниже границы.")
    print("Электрон, u- и d-кварки на эту лестницу не попадают ни при каком "
          "выборе уровня.")

    print("\n" + "=" * 78)
    print("КОНТРОЛЬНЫЙ ТЕСТ: сравнение со случайной лестницей")
    print("=" * 78)
    real_err, rnd_med, frac = null_test(taus, gammas, grid)
    print(f"Медианное отклонение на лестнице модели:   "
          f"{np.median(real_err):6.2f} %")
    print(f"Медианное отклонение на случайной лестнице: "
          f"{np.median(rnd_med):6.2f} %  (медиана по 2000 розыгрышам)")
    print(f"Доля случайных лестниц, которые ХУЖЕ модельной: {frac*100:.1f} %")
    if frac < 0.95:
        print("ВЫВОД: совпадения не значимы — случайная лестница с тем же "
              "числом уровней\n"
              "справляется не хуже. Согласие на тяжёлом конце порождено "
              "густотой уровней:\n"
              "при m ~ n^2 относительный зазор между соседними уровнями равен "
              "2/n,\n"
              "то есть у n=80 он около 2.5 %, и туда попадает что угодно.")
    else:
        print("ВЫВОД: лестница модели значимо лучше случайной.")

    print("\n" + "=" * 78)
    print("ОБРАТНАЯ ЗАДАЧА")
    print("=" * 78)
    eta_independence_note()
    inverse_search()

    only_leptons = [a for a in assign if a[0] in CHARGED_LEPTONS]
    print("\nТолько заряженные лептоны:")
    for sym, name, kind, m_obs, n, tau, m_pred, err in only_leptons:
        print(f"  {sym:4s} n={n:3d}  m_набл={m_obs:11.5f}  "
              f"m_мод={m_pred:11.5f}  откл. {err:+.2f} %")

    return taus, gammas, ladder, assign




# ----------------------------------------------------------------------------
# 7. Обратная задача: можно ли спасти лестницу подбором eta и уровней
# ----------------------------------------------------------------------------

LEPTON_RATIOS = (
    105.6583755 / 0.51099895,   # m_mu / m_e = 206.77
    1776.86 / 0.51099895,       # m_tau / m_e = 3477.5
)


def inverse_search(eta=None, n_levels=200, powers=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
                   verbose=True):
    """Обратная задача: существует ли тройка уровней (i<j<k), воспроизводящая
    отношения масс заряженных лептонов при законе m ~ tau^p?

    Ответ (см. вывод): существует, и притом для ЛЮБОГО показателя p, с точностью
    лучше процента. Это и есть диагноз: при свободном выборе трёх целых номеров
    схема не обладает предсказательной силой — она подгоняется подо что угодно.
    """
    if eta is None:
        eta = ETA_EFF
    taus = find_roots(eta, n_roots=n_levels)
    r_mu, r_tau = LEPTON_RATIOS
    results = []
    for p in powers:
        m = taus ** p
        best = None
        for i in range(len(taus) - 2):
            rel = m / m[i]
            j = int(np.argmin(np.abs(np.log(rel) - np.log(r_mu))))
            k = int(np.argmin(np.abs(np.log(rel) - np.log(r_tau))))
            if j <= i or k <= j:
                continue
            e1 = abs(rel[j] / r_mu - 1.0)
            e2 = abs(rel[k] / r_tau - 1.0)
            sc = max(e1, e2)
            if best is None or sc < best[0]:
                best = (sc, i + 1, j + 1, k + 1, e1, e2)
        results.append((p,) + best)
    if verbose:
        print(f"\nПодбор тройки уровней при m ~ tau^p "
              f"({n_levels} уровней, eta = {eta:.4f}):")
        print(f"{'p':>6} {'n_e':>5} {'n_mu':>6} {'n_tau':>6} "
              f"{'ошибка m_mu':>13} {'ошибка m_tau':>13}")
        for p, sc, i, j, k, e1, e2 in results:
            print(f"{p:6.1f} {i:5d} {j:6d} {k:6d} "
                  f"{e1*100:12.3f}% {e2*100:12.3f}%")
        print("Подгонка удаётся при ЛЮБОМ показателе — значит, схема со "
              "свободным\nвыбором номеров уровней ничего не предсказывает. "
              "Модель обязана\nвыводить номера, а не подбирать их.")
    return results


def eta_independence_note(eta_a=1.5, eta_b=4.0, n=40):
    """Показать, что при больших n отношения корней от eta не зависят:
    tau_n ~ (n + delta) * pi / eta, и eta сокращается в любом отношении.
    Следовательно, eta задаёт только общий масштаб, но не отношения масс.
    """
    ra = find_roots(eta_a, n_roots=n)
    rb = find_roots(eta_b, n_roots=n)
    print(f"\nОтношения корней tau_n/tau_5 при двух разных eta:")
    print(f"{'n':>4} {'eta='+format(eta_a,'.1f'):>12} {'eta='+format(eta_b,'.1f'):>12}"
          f" {'расхождение':>13}")
    for n_ in (10, 20, 30, 39):
        a, b = ra[n_] / ra[4], rb[n_] / rb[4]
        print(f"{n_+1:4d} {a:12.6f} {b:12.6f} {abs(a-b)/a*100:12.3f} %")
    print("Отношения практически совпадают: eta задаёт только общий масштаб\n"
          "спектра, но НЕ отношения масс. Подбором eta спектр не спасти.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# 8. Соотношение Коиде как кандидат в правила отбора уровней (T-15)
# ----------------------------------------------------------------------------

KOIDE_TARGET = 2.0 / 3.0


def koide(masses):
    """R = sum(m) / (sum(sqrt(m)))^2. Для трёх заряженных лептонов R ~ 2/3."""
    m = np.asarray(masses, dtype=float)
    return m.sum() / (np.sqrt(m).sum() ** 2)


def koide_experimental():
    m = [VACUUM_POLARIZATION_GRID[0][2],    # e
         105.6583755,                        # mu
         1776.86]                            # tau
    return koide(m)


def koide_diophantine_search(p=2, i_max=60, j_max=900, k_max=3600):
    """Точные целочисленные решения условия Коиде на степенной лестнице m ~ n^p.

    При чётном p подстановка x = n^(p/2) целочисленна, и условие R = 2/3
    превращается в однородное диофантово уравнение

        3(x^2 + y^2 + z^2) = 2(x + y + z)^2   <=>   x^2+y^2+z^2 = 4(xy+yz+zx)

    Возвращает список найденных решений (для p = 2, 4, 6 он пуст — см. теорему
    в koide_no_solution_proof()).
    """
    h = p // 2
    sols = []
    for i in range(1, i_max):
        a = i ** h
        for j in range(i + 1, j_max):
            b = j ** h
            for k in range(j + 1, k_max):
                c = k ** h
                if a * a + b * b + c * c == 4 * (a * b + b * c + c * a):
                    sols.append((i, j, k))
    return sols


def koide_no_solution_proof(verbose=True):
    """Доказательство отсутствия нетривиальных решений (спуск Ферма).

    Уравнение x^2 + y^2 + z^2 = 4(xy + yz + zx) имеет правую часть, кратную 4.
    Квадрат целого по модулю 4 равен 0 или 1, поэтому сумма трёх квадратов
    сравнима с нулём по модулю 4 только если все три числа чётны.
    Деление на 2 даёт то же уравнение => бесконечный спуск => решений нет.
    """
    residues = sorted({(r * r) % 4 for r in range(4)})
    combos = [(a, b, c) for a in residues for b in residues for c in residues
              if (a + b + c) % 4 == 0]
    if verbose:
        print("\nТеорема (спуск Ферма).")
        print("  Уравнение:  x^2 + y^2 + z^2 = 4(xy + yz + zx)")
        print(f"  Квадраты по модулю 4 принимают значения {residues}.")
        print(f"  Сумма трёх квадратов кратна 4 только в комбинации {combos},")
        print("  то есть только когда x, y, z все чётные.")
        print("  Деление на 2 воспроизводит то же уравнение => бесконечный спуск.")
        print("  Следовательно, нетривиальных целочисленных решений НЕТ.")
    return combos


def koide_near_solutions(tol=3e-5, i_max=60, j_max=900, k_max=3600, top=8):
    """Приближённые решения: тройки уровней, дающие R близко к 2/3 при m ~ n^2."""
    out = []
    for i in range(1, i_max):
        for j in range(i + 1, j_max):
            for k in range(j + 1, k_max):
                r = (i * i + j * j + k * k) / ((i + j + k) ** 2)
                if abs(r - KOIDE_TARGET) < tol:
                    out.append((abs(r - KOIDE_TARGET), i, j, k, r))
    out.sort()
    return len(out), out[:top]


def koide_report():
    """Полный отчёт по пункту T-15: годится ли Коиде как правило отбора."""
    print("=" * 78)
    print("СООТНОШЕНИЕ КОИДЕ КАК КАНДИДАТ В ПРАВИЛА ОТБОРА УРОВНЕЙ")
    print("=" * 78)
    r_exp = koide_experimental()
    print(f"На экспериментальных массах:  R = {r_exp:.9f}")
    print(f"Целевое значение 2/3:         R = {KOIDE_TARGET:.9f}")
    print(f"Отклонение: {(r_exp / KOIDE_TARGET - 1) * 100:+.5f} %  "
          f"— соотношение выполняется с точностью 1e-5")

    koide_no_solution_proof()

    print("\nЧисленная проверка отсутствия решений:")
    for p in (2, 4, 6):
        s = koide_diophantine_search(p=p, i_max=40, j_max=200, k_max=600)
        print(f"  m ~ n^{p}: найдено точных решений — {len(s)}")

    n_near, top = koide_near_solutions()
    print(f"\nПриближённые решения (|R - 2/3| < 3e-5): {n_near} троек")
    print("Примеры:")
    for d, i, j, k, r in top[:5]:
        print(f"  ({i:3d},{j:4d},{k:5d})  R = {r:.9f}   "
              f"m_j/m_i = {j*j/(i*i):10.2f}  m_k/m_i = {k*k/(i*i):11.2f}")
    print("Наблюдаемые отношения: m_mu/m_e = 206.77, m_tau/m_e = 3477.23")

    print("\nВЫВОД:")
    print("  1. Точно Коиде на квадратичной лестнице НЕВОЗМОЖНО (теорема выше).")
    print("     Это прямое противоречие между подглавами 4.4 и 4.5.")
    print("  2. Приближённо решений слишком много (десятки тысяч), поэтому")
    print("     правилом отбора уровней соотношение Коиде служить не может.")


# ----------------------------------------------------------------------------
# 9. Четвёртое поколение: предсказания модели против экспериментальных границ
# ----------------------------------------------------------------------------

# Экспериментальные ограничения (LEP / PDG), ГэВ
LIMIT_L4_CHARGED = 100.8   # прямой поиск последовательного заряженного лептона
LIMIT_NU4_DIRAC = 45.6     # из N_nu = 2.984 +- 0.008 (m_Z/2)
N_NU_LEP = (2.984, 0.008)


def koide_extend(masses, target, pick="all"):
    """Решить уравнение Коиде относительно новой массы.

        (S + x^2) / (T + x)^2 = target,   S = sum(m), T = sum(sqrt(m)), x = sqrt(m_new)

    приводится к  (1 - target) x^2 - 2*target*T*x + S - target*T^2 = 0.
    Возвращает список положительных решений m_new = x^2.
    """
    m = np.asarray(masses, dtype=float)
    S, T = m.sum(), np.sqrt(m).sum()
    a = 1.0 - target
    b = -2.0 * target * T
    c = S - target * T * T
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return []
        return [(-c / b) ** 2]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    out = sorted(r * r for r in roots if r > 0)
    if pick == "heavy":
        return out[-1:]
    if pick == "light":
        return out[:1]
    return out


def fourth_generation_report():
    """Сопоставление всех способов продолжить лестницу с границами LEP."""
    me = VACUUM_POLARIZATION_GRID[0][2]
    mmu, mtau = 105.6583755, 1776.86
    three = [me, mmu, mtau]

    print("=" * 78)
    print("ЧЕТВЁРТОЕ ПОКОЛЕНИЕ: ПРЕДСКАЗАНИЯ МОДЕЛИ ПРОТИВ ГРАНИЦ LEP")
    print("=" * 78)
    print(f"Коиде на (e, mu, tau):  R = {koide(three):.9f}   (2/3 = {2/3:.9f})")
    print()
    print("Экспериментальные ограничения:")
    print(f"  число лёгких нейтрино N = {N_NU_LEP[0]} +- {N_NU_LEP[1]}"
          f"  =>  m(nu_4) > {LIMIT_NU4_DIRAC} ГэВ")
    print(f"  прямой поиск на LEP           =>  m(L_4) > {LIMIT_L4_CHARGED} ГэВ")
    print()

    rows = [("A  исходный расчёт 4.4 (tau_4 ~ 44.2)", 17.5),]
    for m4 in koide_extend(three, 0.5):
        rows.append((f"B  продолжение R_4 = 1/2 (2/D при D=4)", m4 / 1000.0))
    for m4 in koide_extend(three, 2.0 / 3.0, pick="heavy"):
        rows.append(("B' сохранение R_4 = 2/3 для четвёрки", m4 / 1000.0))
    waterfall = koide_extend([mmu, mtau], 2.0 / 3.0)
    for m4 in waterfall:
        tag = "C  водопад Коиде (mu, tau, L_4)"
        if m4 < 1.0:
            print(f"Проверка алгебры: нижний корень водопада = {m4:.4f} МэВ, "
                  f"m_e = {me:.5f} МэВ, отклонение {abs(m4/me-1)*100:.2f} %")
            continue
        rows.append((tag, m4 / 1000.0))

    print()
    print(f"{'способ':>42} {'m_4, ГэВ':>10} {'статус':>14}")
    for name, m4 in sorted(rows, key=lambda r: r[1]):
        st = "ИСКЛЮЧЕНО" if m4 < LIMIT_L4_CHARGED else "разрешено"
        print(f"{name:>42} {m4:10.2f} {st:>14}")

    print()
    print("ВЫВОД: все способы продолжить лестницу дают массу ниже границы LEP.")
    print("Модель не предсказывает ОТСУТСТВИЯ четвёртого поколения — она")
    print("предсказывает четвёртый лептон в области 9-45 ГэВ, и это опровергнуто.")
    print("Отсутствие четвёртого поколения остаётся необъяснённым фактом.")
    return rows


# ----------------------------------------------------------------------------
# 10. Обратные решения Коиде и проверка точности соотношения
# ----------------------------------------------------------------------------

# Массы и погрешности PDG, МэВ
LEPTON_MASSES = {
    "e":   (0.51099895, 0.00000015),
    "mu":  (105.6583755, 0.0000023),
    "tau": (1776.86,     0.12),
}


def koide_solve_missing(known_masses, target=2.0 / 3.0):
    """Все положительные решения уравнения Коиде относительно недостающей массы."""
    return koide_extend(known_masses, target, pick="all")


def koide_inverse_report():
    """Решить Коиде поочерёдно относительно каждой массы и оценить, совместимо
    ли ТОЧНОЕ значение R = 2/3 с экспериментальными погрешностями.

    Существенная оговорка: это НЕ предсказание. Коиде — одно уравнение, и по
    любым двум массам третья определяется однозначно (с точностью до второго
    корня квадратного уравнения). Ценность расчёта в другом — в чувствительности:
    он показывает, какую массу соотношение фиксирует жёстче всего и можно ли
    списать наблюдаемое отклонение R от 2/3 на погрешность измерения.
    """
    me, d_me = LEPTON_MASSES["e"]
    mmu, d_mmu = LEPTON_MASSES["mu"]
    mtau, d_mtau = LEPTON_MASSES["tau"]
    R_exp = koide([me, mmu, mtau])
    dR = R_exp / KOIDE_TARGET - 1.0

    print("=" * 78)
    print("ОБРАТНЫЕ РЕШЕНИЯ КОИДЕ И ТОЧНОСТЬ СООТНОШЕНИЯ")
    print("=" * 78)
    print(f"R на измеренных массах = {R_exp:.9f};  отклонение от 2/3: {dR:+.3e}")
    print()
    print(f"{'ищем':>7} {'по паре':>11} {'предсказание':>16} {'измерено':>14}"
          f" {'откл.':>10} {'сигм':>10} {'усиление':>10}")

    cases = [("m_e", (mmu, mtau), me, d_me, "(mu,tau)"),
             ("m_mu", (me, mtau), mmu, d_mmu, "(e,tau)"),
             ("m_tau", (me, mmu), mtau, d_mtau, "(e,mu)")]
    out = []
    for name, known, obs, dobs, pair in cases:
        roots = koide_solve_missing(known)
        pred = min(roots, key=lambda r: abs(np.log(r / obs)))
        dev = (pred - obs) / obs * 100.0
        sig = (pred - obs) / dobs
        amp = abs((pred - obs) / obs) / abs(dR)
        print(f"{name:>7} {pair:>11} {pred:16.6f} {obs:14.6f} "
              f"{dev:+9.4f}% {sig:+10.2f} {amp:10.1f}")
        out.append((name, pred, obs, dobs, sig, amp, roots))

    print()
    print("Посторонние корни (второе решение квадратного уравнения):")
    for name, pred, obs, dobs, sig, amp, roots in out:
        extra = [r for r in roots if abs(r - pred) > 1e-9]
        if extra:
            print(f"  при поиске {name:>6}: {['%.3f МэВ' % r for r in extra]}"
                  f"  — реальным частицам не отвечают")

    print()
    print("КЛЮЧЕВОЙ ВЫВОД")
    pred_tau = out[2][1]
    print(f"  Точное Коиде требует m_tau = {pred_tau:.4f} МэВ")
    print(f"  PDG:                 m_tau = {mtau:.2f} +- {d_mtau:.2f} МэВ")
    print(f"  Расхождение = {pred_tau - mtau:+.4f} МэВ = "
          f"{(pred_tau - mtau)/d_mtau:+.2f} sigma")
    print("  Всё отклонение R от 2/3 списывается на погрешность измерения m_tau.")
    print("  => ТОЧНОЕ соотношение Коиде совместимо с данными.")
    print("     Вариант B пункта T-16 («Коиде лишь приближённое») данными")
    print("     НЕ подтверждается и как выход из противоречия не годится.")
    return out


# ----------------------------------------------------------------------------
# 11. Аудит подгонки (theory/04-9)
# ----------------------------------------------------------------------------
# Назначение: механически проверить, не подогнана ли схема под данные.
# Любой критик может запустить audit_report() и воспроизвести выводы.

PROVENANCE = [
    # (величина, происхождение, свободных параметров)
    ("alpha_0 = 1/(48 pi)", "топология: 3 оси x 2 полоид. обхода x 8pi", 0),
    ("cosh eta_0 = (2 pi / alpha_0)^(1/4)", "конформная фиксация из alpha_0", 0),
    ("швингеровский множитель 1 - alpha_0/2pi", "отдача, из alpha_0", 0),
    ("eta_eff", "следует из предыдущих", 0),
    ("tau_n (корни Лежандра)", "следуют из eta_eff", 0),
    ("Gamma_n (интегралы перекрытия)", "следуют из tau_n", 0),
    ("показатель закона масс (tau^3 * Gamma -> tau^2)", "вычислен в 4.5", 0),
    ("номера уровней (i, j, k)", "ПОДОБРАНЫ под данные", 3),
    ("масштабный множитель c", "ПОДОГНАН под данные", 1),
    ("аддитивный сдвиг d (энергия узла)", "ПОДОГНАН под данные", 1),
]


def check_alpha0_chain():
    """Проверить, что cosh eta_0 определяется через alpha_0 без свободы."""
    lhs = 96.0 * np.pi ** 2
    rhs = 2.0 * np.pi / ALPHA_0
    return lhs, rhs, abs(lhs / rhs - 1.0)


def fit_triple(targets, ladder=None, nmax=400):
    """Подобрать (i<j<k), масштаб c и сдвиг d так, чтобы c*L_n + d легли на targets.

    Возвращает (макс. отн. ошибка, i, j, k, c, d) или None.
    Именно эта процедура и подлежит проверке на поддельных данных.
    """
    if ladder is None:
        ladder = find_roots(ETA_EFF, n_roots=nmax) ** 2
    t = np.sort(np.asarray(targets, dtype=float))
    L = ladder[:nmax]
    best = None
    for i in range(len(L) - 2):
        for j in range(i + 1, len(L) - 1):
            denom = L[j] - L[i]
            if denom <= 0:
                continue
            c = (t[1] - t[0]) / denom
            d = t[0] - c * L[i]
            pred_k = (t[2] - d) / c
            if pred_k <= L[j]:
                continue
            k0 = int(np.searchsorted(L, pred_k))
            for k in (k0 - 1, k0, k0 + 1):
                if k <= j or k >= len(L):
                    continue
                err = abs((c * L[k] + d) / t[2] - 1.0)
                if best is None or err < best[0]:
                    best = (err, i + 1, j + 1, k + 1, c, d)
    return best


def null_test_fake_masses(n_trials=300, nmax=400, seed=20260813, ladder=None):
    """РЕШАЮЩАЯ ПРОВЕРКА: применить ту же процедуру подгонки к СЛУЧАЙНЫМ тройкам
    масс в том же диапазоне. Если поддельные данные ложатся не хуже настоящих,
    схема предсказательной силой не обладает.
    """
    if ladder is None:
        ladder = find_roots(ETA_EFF, n_roots=nmax) ** 2
    real = [LEPTON_MASSES["e"][0], LEPTON_MASSES["mu"][0], LEPTON_MASSES["tau"][0]]
    real_err = fit_triple(real, ladder, nmax)[0]
    rng = np.random.default_rng(seed)
    lo, hi = np.log(0.5), np.log(1800.0)
    errs = []
    for _ in range(n_trials):
        fake = np.sort(np.exp(rng.uniform(lo, hi, 3)))
        if fake[1] / fake[0] < 3 or fake[2] / fake[1] < 3:
            continue
        r = fit_triple(fake, ladder, nmax)
        if r:
            errs.append(r[0])
    return real_err, np.array(errs)


def zero_freedom_test(rules=None, nmax=60):
    """Тест БЕЗ подгонки номеров: правило задаёт (i,j,k), c и d фиксируются
    по (m_e, m_mu), масса тау-лептона ПРЕДСКАЗЫВАЕТСЯ."""
    if rules is None:
        rules = [("последовательные", (1, 2, 3)), ("нечётные", (1, 3, 5)),
                 ("квадраты", (1, 4, 9)), ("степени 3", (1, 3, 9)),
                 ("степени 4", (1, 4, 16)), ("треугольные", (1, 3, 6)),
                 ("простые", (2, 3, 5)), ("пятиугольные", (1, 5, 12))]
    L = find_roots(ETA_EFF, n_roots=nmax) ** 2
    me = LEPTON_MASSES["e"][0]
    mmu = LEPTON_MASSES["mu"][0]
    mtau = LEPTON_MASSES["tau"][0]
    out = []
    for name, (i, j, k) in rules:
        if k > len(L):
            continue
        c = (mmu - me) / (L[j - 1] - L[i - 1])
        d = me - c * L[i - 1]
        pred = c * L[k - 1] + d
        out.append((name, (i, j, k), pred, pred / mtau - 1.0))
    return out


def audit_report():
    """Полный аудит подгонки. Ответ на вопрос «не подогнано ли под данные?»"""
    print("=" * 78)
    print("АУДИТ ПОДГОНКИ: СКОЛЬКО ПАРАМЕТРОВ И СКОЛЬКО НАБЛЮДАЕМЫХ")
    print("=" * 78)

    lhs, rhs, rel = check_alpha0_chain()
    print("\nA. Цепочка параметров вакуума")
    print(f"   96*pi^2 = {lhs:.6f};  2*pi/alpha_0 = {rhs:.6f};  расхождение {rel:.2e}")
    print("   => cosh eta_0 выражается через alpha_0 тождественно, свободы НЕТ.")

    print("\nB. Происхождение всех величин")
    print(f"   {'величина':>44} {'своб.':>6}  происхождение")
    total = 0
    for name, origin, k in PROVENANCE:
        total += k
        print(f"   {name:>44} {k:6d}  {origin}")
    print(f"   {'ИТОГО подгоночных параметров':>44} {total:6d}")
    print(f"   {'наблюдаемых (массы e, mu, tau)':>44} {3:6d}")
    print(f"   {'чистых предсказаний':>44} {3-total:6d}   <- отрицательно")

    print("\nC. Решающая проверка на поддельных данных")
    real_err, errs = null_test_fake_masses()
    print(f"   ошибка подгонки реальных масс:  {real_err*100:.6f} %")
    print(f"   медиана по {len(errs)} поддельным тройкам: "
          f"{np.median(errs)*100:.6f} %")
    print(f"   доля подделок, севших ЛУЧШЕ реальных: {(errs < real_err).mean()*100:.1f} %")
    print("   => процедура ложится на ЛЮБЫЕ три массы. Предсказательной силы нет.")

    print("\nD. Тест без подгонки номеров (предсказание m_tau)")
    print(f"   {'правило':>18} {'(i,j,k)':>12} {'m_tau, МэВ':>12} {'откл.':>10}")
    for name, ijk, pred, dev in zero_freedom_test():
        print(f"   {name:>18} {str(ijk):>12} {pred:12.2f} {dev*100:+9.1f}%")
    print("   => ни одно простое правило не работает; провалы 23-83 %.")

    print("\nВЕРДИКТ: блок вывода поколений НЕ замкнут.")
    print("  Часть с параметрами вакуума чиста — подгонки в ней нет.")
    print("  Часть со спектром масс целиком подгоночная: 5 параметров на 3 массы.")
    print("  Условия замыкания — см. theory/04-9-audit-podgonki.md.")
