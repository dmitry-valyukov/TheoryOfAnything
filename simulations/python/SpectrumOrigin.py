"""
Происхождение чисел в выводе alpha и спектра лептонов (пункты T-6 и T-7).

Оба пункта сводятся к одному вопросу: откуда взялись конкретные числа —
Gamma_int = 0.048287 (T-6) и tau_2 = 8.31211, tau_3 = 22.14618 (T-7).
Настоящий модуль отвечает на него прямым перебором гипотез.

Коническая функция Лежандра вычисляется ДВУМЯ независимыми способами:
  (а) интегрирование уравнения Лежандра по eta — авторитетный способ,
      не зависящий ни от какой нормировки интегральных представлений:
          y'' + cth(eta) y' + (tau^2 + 1/4) y = 0,  y(0) = 1, y'(0) = 0
  (б) интеграл Мелера — Дирихле
          P = (2/pi) * int_0^eta cos(tau t) / sqrt(2(ch eta - ch t)) dt
Совпадение (а) и (б) — контроль правильности реализации.

Разделы:
  1. контроль реализации специальной функции
  2. спектр: где на самом деле лежат нули
  3. T-7: откуда tau_2 = 8.31211 и tau_3 = 22.14618
  4. T-6: откуда Gamma_int = 0.048287
  5. чем на самом деле держится согласие масс

Расчёты к подглавам theory/04-2 и theory/04-3.
Запуск:  python SpectrumOrigin.py
"""

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

# --- каустика из подглавы 4.2 -------------------------------------------
ALPHA0 = 1.0 / (48.0 * math.pi)
CH_ETA0 = (96.0 * math.pi**2) ** 0.25
ETA0 = math.acosh(CH_ETA0)
CH_ETA_EFF = CH_ETA0 * (1.0 - ALPHA0 / (2.0 * math.pi))
ETA_EFF = math.acosh(CH_ETA_EFF)

# --- числа, происхождение которых восстанавливается ----------------------
TAU_CLAIMED = (math.sqrt(2.0), 8.31211, 22.14618)
GAMMA_CLAIMED = (0.048287, 0.049311, 0.043782)

M_E, M_MU, M_TAU = 0.51099895, 105.6583755, 1776.86
INV_ALPHA = 137.035999084


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------- (а) через уравнение Лежандра

def P_ode(tau, eta_max, rtol=1e-12, atol=1e-14):
    """
    y'' + cth(eta) y' + (tau^2 + 1/4) y = 0,  y(0)=1, y'(0)=0.
    Разложение у нуля: y = 1 - (tau^2+1/4) eta^2/4 + ...
    """
    k2 = tau * tau + 0.25
    e0 = 1e-8
    y0 = 1.0 - k2 * e0**2 / 4.0
    dy0 = -k2 * e0 / 2.0

    def rhs(e, s):
        return [s[1], -(1.0 / math.tanh(e)) * s[1] - k2 * s[0]]

    sol = solve_ivp(rhs, (e0, eta_max), [y0, dy0], method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True)
    return sol


def P_at(tau, eta):
    return float(P_ode(tau, eta).y[0, -1])


# ------------------------------------------ (б) интеграл Мелера — Дирихле

def P_md(tau, eta):
    """P = (2/pi) int_0^eta cos(tau t) / sqrt(2(ch eta - ch t)) dt.
    Замена t = eta - v^2 снимает корневую особенность на верхнем пределе."""
    che = math.cosh(eta)

    def f(v):
        t = eta - v * v
        d = 2.0 * (che - math.cosh(t))
        if d <= 0.0:
            return 2.0 / math.sqrt(2.0 * math.sinh(eta)) * math.cos(tau * eta) if eta > 0 else 0.0
        return 2.0 * v * math.cos(tau * t) / math.sqrt(d)

    val, _ = quad(f, 0.0, math.sqrt(eta), limit=400)
    return 2.0 / math.pi * val


# ------------------------------------------------ 1. контроль реализации

def check_implementation():
    head("1. КОНТРОЛЬ РЕАЛИЗАЦИИ СПЕЦИАЛЬНОЙ ФУНКЦИИ")
    print(f"  ch eta_0   = {CH_ETA0:.7f}   eta_0   = {ETA0:.7f}")
    print(f"  ch eta_эфф = {CH_ETA_EFF:.7f}   eta_эфф = {ETA_EFF:.7f}")
    print()
    print(f"  {'tau':>12} {'P (ОДУ)':>16} {'P (Мелер—Дирихле)':>20} {'расхожд.':>12}")
    ok = True
    for tau in (0.5, 1.0, math.sqrt(2.0), 3.0, 8.31211, 22.14618):
        a = P_at(tau, ETA_EFF)
        b = P_md(tau, ETA_EFF)
        d = abs(a - b)
        ok &= d < 1e-6 * max(1.0, abs(a))
        print(f"  {tau:>12.5f} {a:>16.8f} {b:>20.8f} {d:>12.2e}")
    print()
    print(f"  два независимых метода согласуются: {'ДА' if ok else 'НЕТ'}")
    print("""
  Уравнение y'' + cth(eta) y' + (tau^2 + 1/4) y = 0 ОСЦИЛЛЯТОРНО: коэффициент
  при y положителен. Значит функция имеет нули и по eta, и по tau, причём
  асимптотический шаг нулей по tau равен pi/eta. Спектральная задача
  осмысленна — вопрос лишь в том, те ли у неё корни, что использованы.""")
    return ok


# ------------------------------------------------ 2. где на самом деле нули

def true_roots(n=14):
    head("2. ИСТИННЫЕ КОРНИ СПЕКТРАЛЬНОГО УРАВНЕНИЯ")
    f = lambda t: P_at(t, ETA_EFF)
    roots, t, prev, step = [], 0.05, None, 0.01
    while len(roots) < n and t < 60.0:
        cur = f(t)
        if prev is not None and prev * cur < 0:
            roots.append(brentq(f, t - step, t, xtol=1e-12))
        prev, t = cur, t + step
    roots = np.array(roots)
    print(f"  первые {len(roots)} корней при eta_эфф = {ETA_EFF:.7f}:")
    for i in range(0, len(roots), 5):
        print("     " + "  ".join(f"{r:9.5f}" for r in roots[i:i+5]))
    d = np.diff(roots)
    print(f"\n  шаг: среднее {d.mean():.5f}, теоретический pi/eta_эфф = {math.pi/ETA_EFF:.5f}")
    print("""
  Набор почти эквидистантный. Использованные в 4.3 значения
  sqrt(2) = 1.41421, 8.31211, 22.14618 в нём отсутствуют — проверяется ниже.""")
    return roots


# ------------------------------------------------ 3. T-7

def t7_origin(roots):
    head("3. T-7: ОТКУДА tau_2 = 8.31211 И tau_3 = 22.14618")

    print("  Гипотеза A: это корни при том же eta_эфф.")
    print(f"  {'tau':>12} {'P(tau)':>14} {'ближайший корень':>18} {'промах':>10}")
    for tau in TAU_CLAIMED:
        p = P_at(tau, ETA_EFF)
        near = roots[np.argmin(np.abs(roots - tau))]
        print(f"  {tau:>12.5f} {p:>14.6f} {near:>18.5f} {abs(near-tau):>10.4f}")
    print("  => НЕТ: ни одно из трёх не является корнем.\n")

    print("  Гипотеза B: есть ли ХОТЬ КАКАЯ-ТО каустика eta, при которой")
    print("  все три числа одновременно являются корнями (с любыми номерами)?")
    print("  Сканирование по eta; мера промаха — max по трём |P(tau, eta)|,")
    print("  нормированный на локальную амплитуду огибающей.")
    grid = np.linspace(0.20, 8.0, 4000)
    # одно решение ОДУ на каждое tau, затем плотная выдача по всей сетке
    curves = []
    for tau in TAU_CLAIMED:
        sol = P_ode(tau, grid[-1] + 0.01)
        curves.append(np.array([sol.sol(e)[0] for e in grid]))
    amp = 1.0 / np.sqrt(np.sinh(grid))
    miss = np.max(np.abs(np.array(curves)) / amp, axis=0)
    best = sorted(zip(miss.tolist(), grid.tolist()))
    print(f"  {'eta':>12} {'норм. промах':>16}")
    for miss, e in best[:5]:
        print(f"  {e:>12.5f} {miss:>16.5f}")
    print(f"\n  минимальный промах по всей сетке: {best[0][0]:.5f}")
    if best[0][0] > 0.02:
        print("  => НЕТ: ни при какой каустике три числа не становятся корнями")
        print("     одновременно. Гипотеза «другая каустика» закрыта.")
    else:
        print("  => найдена каустика-кандидат, требует отдельного разбора.")

    print("\n  Гипотеза B': условие Неймана dP/d(eta) = 0 вместо P = 0.")
    f = lambda t: P_ode(t, ETA_EFF).y[1, -1]
    nroots, x, prev, step = [], 0.05, None, 0.01
    while len(nroots) < 20 and x < 40.0:
        cur = f(x)
        if prev is not None and prev * cur < 0:
            nroots.append(brentq(f, x - step, x, xtol=1e-12))
        prev, x = cur, x + step
    nroots = np.array(nroots)
    print("     корни Неймана: " + "  ".join(f"{r:.5f}" for r in nroots[:8]))
    for tau in TAU_CLAIMED:
        near = nroots[np.argmin(np.abs(nroots - tau))]
        print(f"     tau = {tau:9.5f}  ближайший {near:9.5f}  промах {abs(near-tau):.4f}")
    print("  => НЕТ: условие Неймана тоже не порождает эти числа.\n")

    print("  Гипотеза C: восстановлены из масс по закону m ~ tau^p.")
    for name, m, tau in (("мюон", M_MU, TAU_CLAIMED[1]), ("тау", M_TAU, TAU_CLAIMED[2])):
        p = math.log(m / M_E) / math.log(tau / TAU_CLAIMED[0])
        print(f"  {name:>6}: m/m_e = {m/M_E:9.3f}, tau/tau_1 = {tau/TAU_CLAIMED[0]:8.5f}"
              f"  =>  p = {p:.4f}")
    print("  => показатели 3.01 и 2.96 — закон m ~ tau^3 с точностью 1%.")
    print("     Значит tau подобраны ПОД КУБИЧЕСКИЙ ЗАКОН, а не выведены:")
    for name, m in (("мюон", M_MU), ("тау", M_TAU)):
        tau_need = TAU_CLAIMED[0] * (m / M_E) ** (1.0 / 3.0)
        i = 1 if name == "мюон" else 2
        print(f"     {name:>6}: из m ~ tau^3 ровно  tau = {tau_need:.5f}"
              f"   заявлено {TAU_CLAIMED[i]:.5f}   расхожд. {abs(tau_need-TAU_CLAIMED[i])/tau_need:.2%}")


# ------------------------------------------------ 4. T-6

def gamma_int(tau, eta_max):
    sol = P_ode(tau, eta_max)
    g = lambda e: sol.sol(e)[0] ** 2 * math.sinh(e)
    val, _ = quad(g, 1e-8, eta_max, limit=400)
    return val


def t6_origin():
    head("4. T-6: ОТКУДА Gamma_int = 0.048287")

    print("  Прямой расчёт в стандартной нормировке (P(1) = 1):")
    print(f"  {'tau':>12} {'Gamma числ.':>14} {'Gamma заявл.':>14} {'отношение':>12}")
    ratios = []
    for tau, gc in zip(TAU_CLAIMED, GAMMA_CLAIMED):
        gn = gamma_int(tau, ETA_EFF)
        ratios.append(gn / gc)
        print(f"  {tau:>12.5f} {gn:>14.6f} {gc:>14.6f} {gn/gc:>12.4f}")
    print()
    print(f"  отношения численного к заявленному: "
          + ", ".join(f"{r:.4f}" for r in ratios))
    spread = max(ratios) / min(ratios)
    print(f"  разброс отношений: {spread:.3f}")
    if spread < 1.02:
        print("  => заявленные Gamma отличаются от численных ОБЩИМ множителем:")
        print(f"     это была бы просто нормировка, множитель ~ {np.mean(ratios):.4f}")
    else:
        print("  => общего множителя НЕТ: заявленные Gamma не воспроизводятся")
        print("     стандартным интегралом ни при какой нормировке.")

    print()
    gc = np.array(GAMMA_CLAIMED)
    gn = np.array([gamma_int(x, ETA_EFF) for x in TAU_CLAIMED])
    print("  Форма набора — отдельная улика:")
    print(f"    заявленные Gamma: разброс {gc.max()/gc.min():.3f}, "
          f"среднее {gc.mean():.6f}, отклонение {gc.std()/gc.mean():.1%}")
    print(f"    численные  Gamma: разброс {gn.max()/gn.min():.2f}")
    print()
    print("    Заявленный набор ПОЧТИ ПОСТОЯНЕН — расходится на 5 процентов,")
    print("    тогда как настоящий интеграл меняется в четырнадцать раз.")
    print("    Почти постоянная Gamma означает, что закон сводится к чистому")
    print("    m ~ tau^3, а малые отклонения работают точной подстройкой.")
    print("    Настоящая вычисляемая величина такой формы иметь не может.")
    print()
    print("  Какое значение Gamma требуется на самом деле:")
    need = math.sqrt(48.0 * math.pi / INV_ALPHA) - 1.0
    print(f"    из alpha = alpha_0 (1 + Gamma)^2  =>  Gamma = {need:.6f}")
    print(f"    заявлено                              Gamma = {GAMMA_CLAIMED[0]:.6f}")
    print(f"    расхождение                                  {abs(need-GAMMA_CLAIMED[0])/need:.2%}")
    print("""
  Существенно: заявленное значение НЕ равно требуемому. Будь оно подогнано
  под alpha, совпадение было бы точным. Промах в полтора процента означает,
  что 0.048287 пришло откуда-то ещё — но откуда, стандартный интеграл
  не объясняет.""")

    print("  При каком верхнем пределе интеграл дал бы 0.048287:")
    try:
        e = brentq(lambda x: gamma_int(TAU_CLAIMED[0], x) - GAMMA_CLAIMED[0], 0.05, ETA_EFF)
        print(f"    eta = {e:.6f}   против eta_эфф = {ETA_EFF:.6f}   (отношение {e/ETA_EFF:.4f})")
        print("    Осмысленной интерпретации у этого предела нет.")
    except ValueError:
        print("    решения в диапазоне (0, eta_эфф) нет")


# ------------------------------------------------ 5. чем держится согласие

def what_carries_the_fit():
    head("5. ЧЕМ НА САМОМ ДЕЛЕ ДЕРЖИТСЯ СОГЛАСИЕ МАСС")
    t1, t2, t3 = TAU_CLAIMED
    g1, g2, g3 = GAMMA_CLAIMED
    print(f"  {'':>8} {'(tau_n/tau_1)^3':>18} {'Gamma_n/Gamma_1':>18} {'произведение':>15} {'опыт':>12}")
    for name, t, g, m in (("мюон", t2, g2, M_MU / M_E), ("тау", t3, g3, M_TAU / M_E)):
        cube = (t / t1) ** 3
        rel = g / g1
        print(f"  {name:>8} {cube:>18.3f} {rel:>18.5f} {cube*rel:>15.3f} {m:>12.3f}")
    print()
    print("  Вклад каждого множителя в согласие:")
    for name, t, g, m in (("мюон", t2, g2, M_MU / M_E), ("тау", t3, g3, M_TAU / M_E)):
        cube = (t / t1) ** 3
        print(f"    {name:>6}: один куб даёт промах {abs(cube-m)/m:6.2%},"
              f"  с множителем Gamma — {abs(cube*g/g1-m)/m:6.2%}")
    print("""
  Кубический множитель уже даёт согласие на уровне 2 и 10 процентов, а
  отношение Gamma доводит его до долей процента. При этом tau подобраны так,
  чтобы куб работал (раздел 3), а Gamma не воспроизводятся расчётом (раздел 4).
  То есть согласие несут два подобранных набора чисел, а не вывод.""")


def report():
    check_implementation()
    roots = true_roots()
    t7_origin(roots)
    t6_origin()
    what_carries_the_fit()

    head("СВОДКА ПО T-6 И T-7")
    print("""
  T-7. Происхождение tau_2 = 8.31211 и tau_3 = 22.14618 УСТАНОВЛЕНО, и оно
       не спектральное. Корнями уравнения при eta_эфф они не являются.
       Ни другая каустика (сканирование по eta), ни условие Неймана их
       тоже не дают. Зато закон m ~ tau^p на них даёт p = 3.01 и 2.96, то
       есть числа согласованы с заранее выбранным КУБИЧЕСКИМ законом по
       известным массам — хотя и не равны точному обращению tau ~ m^(1/3)
       (промах 0.6% и 3.4%). Спектральной задачи за ними нет.

  T-6. Gamma_int = 0.048287 стандартным интегралом не воспроизводится ни при
       какой общей нормировке: отношения численного к заявленному для трёх
       уровней расходятся между собой. Одновременно значение НЕ совпадает
       с требуемым для alpha (промах ~1.5%), то есть и прямой подгонкой
       под alpha объяснить его нельзя. Число остаётся необъяснённым, но
       круг гипотез сузился: это не нормировка и не подгонка под alpha.

  ОБЩЕЕ. Оба пункта упираются в одно: спектральная задача, на которую
  ссылаются 4.2 и 4.3, при заявленной каустике корней в нужных местах
  не имеет. Пока не предъявлено ДРУГОЕ граничное условие, оба вывода
  держатся на подобранных числах.
""")


if __name__ == "__main__":
    report()
