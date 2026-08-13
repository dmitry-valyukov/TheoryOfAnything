"""ProtonColor.py — конечная скорость цвета, принцип Паули и геометрия протона.

Расчёты к подглаве theory/05-9-konechnaya-skorost-cveta.md.

Что здесь считается (и что НЕ считается):

  * Считается из модели: условие резонанса при конечной скорости передачи
    цвета, расстояние между кварками L, радиус кварковой триады, время
    цветового оборота, натяжение канала, расщепление N-Delta из счёта
    паулиевских конфликтов.

  * Берётся из эксперимента (и помечено PDG/CODATA): масса протона —
    единственный размерный вход; массы нейтрона, Delta и Omega, а также
    решёточное натяжение струны — только для СРАВНЕНИЯ, ни один из них
    в вывод не входит.

  * Ничего не подгоняется. Все безразмерные множители (3, 4*pi/3, 6, sqrt3)
    выписаны из геометрии, а не подобраны.

Провенанс каждой величины помечен в поле `src` таблиц вывода:
    [M] — выведено в модели
    [E] — эксперимент (PDG/CODATA), вход
    [C] — эксперимент, только для сравнения

Запуск:
    python ProtonColor.py
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

HBAR_C = 197.3269804          # МэВ*фм   [CODATA 2018, точное по определению c и h]
C_FM_S = 2.99792458e23        # фм/с     [точное]
HBAR_MEV_S = 6.582119569e-22  # МэВ*с    [CODATA 2018]

# --- экспериментальные массы, МэВ (PDG 2024) ---
M_PROTON = 938.27208816       # [E]  единственный размерный вход модели
M_NEUTRON = 939.56542052      # [C]
M_DELTA = 1232.0              # [C]  Delta(1232), Брейт-Вигнер
GAMMA_DELTA = 117.0           # [C]  полная ширина Delta(1232), МэВ
M_OMEGA = 1672.45             # [C]  Omega^-
M_PION_CH = 139.57039         # [C]  pi^+-

R_PROTON_EXP = 0.8409         # [C]  фм, зарядовый радиус, CODATA 2022
R_PROTON_ERR = 0.0004         # [C]  фм

# --- решёточное натяжение струны ---
SIGMA_LATTICE_GEV2 = 0.18     # [C]  ГэВ^2, стандартное значение

# --- геометрия модели ---
D_SPACE = 3                   # размерность пространства -> число цветов (5.1)
N_LINKS = 6                   # 3 внешних + 3 внутренних канала (5.6)


def lambda_bar_proton() -> float:
    """Приведённая комптоновская длина протона hbar/(M_p c), фм."""
    return HBAR_C / M_PROTON


# ---------------------------------------------------------------------------
# 1. Условие резонанса при конечной скорости цвета
# ---------------------------------------------------------------------------

def colour_resonance_length(phase_per_leg: float,
                            energy_fraction: float = 1.0) -> float:
    """Расстояние между кварками из условия замыкания цветового обхода.

    Цвет передаётся вдоль ребра треугольника со скоростью c, задержка
    tau = L/c. За задержку внутренняя фаза моды набегает на
    delta_phi = omega * tau. Требование замыкания за один цветовой
    обход (3 ребра) даёт 3*delta_phi = 4*pi (спинорное замыкание),
    то есть delta_phi = 4*pi/3.

    Из omega = f * M_p c^2 / hbar следует
        L = delta_phi * hbar c / (f * M_p c^2).

    Параметры
    ---------
    phase_per_leg : набег внутренней фазы за одно ребро, рад
    energy_fraction : доля энергии покоя, находящаяся в циркулирующей моде
    """
    return phase_per_leg * HBAR_C / (energy_fraction * M_PROTON)


def pauli_kinetic_length() -> float:
    """Оценка из запирания трёх ультрарелятивистских мод в области L.

    Каждая мода несёт импульс p ~ hbar/L, энергия E_i ~ hbar c / L,
    суммарно 3 hbar c / L = M_p c^2  =>  L = 3 * lambda_bar.
    Это паулиевская (обменная) оценка: моды не могут совпасть, значит
    каждая заперта в своей ячейке размера L.
    """
    return D_SPACE * lambda_bar_proton()


def equipartition_length() -> float:
    """Оценка подглавы 5.6: E_link = M_p c^2 / 6, L = hbar c / E_link."""
    return N_LINKS * lambda_bar_proton()


# ---------------------------------------------------------------------------
# 2. Производные величины
# ---------------------------------------------------------------------------

def light_speed_corrections() -> dict:
    """Насколько локальная скорость света внутри протона отличается от c.

    Две независимые оценки, обе пренебрежимо малы:

    1. Гравитационная. В модели гравитация есть размягчение модуля сдвига,
       и относительный сдвиг скорости равен обычному потенциалу
       Phi/c^2 = G M_p / (r_p c^2) — величина порядка 1e-39.

    2. Упругая (нелинейная). Акустоупругая поправка к скорости сдвиговой
       волны пропорциональна отношению деформации солитона к предсжатию
       среды, то есть отношению плотности энергии протона к планковской.
    """
    G = 6.67430e-11               # м^3/(кг с^2)   [CODATA 2018]
    M_P_KG = 1.67262192369e-27    # кг             [CODATA 2018]
    C2 = 2.99792458e8 ** 2        # м^2/с^2
    r_m = R_PROTON_EXP * 1e-15

    phi = G * M_P_KG / (r_m * C2)
    r_s_fm = 2.0 * G * M_P_KG / C2 * 1e15

    u_p = M_PROTON / ((4.0 / 3.0) * math.pi * R_PROTON_EXP ** 3)  # МэВ/фм^3
    e_planck = 1.220890e22        # МэВ           [планковская энергия]
    l_planck = 1.616255e-20       # фм            [планковская длина]
    u_planck = e_planck / l_planck ** 3

    return {
        "grav": phi,
        "r_schwarzschild_fm": r_s_fm,
        "u_proton": u_p,
        "u_planck": u_planck,
        "elastic": u_p / u_planck,
    }


def channel_speed_factor(mult: float) -> float:
    """Во сколько раз скорость передачи цвета ниже c для оценки L = mult * lambda.

    Из L = (4pi/3) * lambda * (v/c) следует v/c = mult / (4pi/3),
    то есть chi = c/v = (4pi/3) / mult. Значение chi < 1 означало бы
    сверхсветовую передачу и запрещено.
    """
    return (4.0 * math.pi / D_SPACE) / mult


def circumradius(side: float) -> float:
    """Радиус описанной окружности равностороннего треугольника."""
    return side / math.sqrt(3.0)


def colour_period(side: float) -> float:
    """Время полного цветового обхода треугольника, с."""
    return D_SPACE * side / C_FM_S


def string_tension_triangle(side: float) -> float:
    """Натяжение при укладке всей энергии покоя в три ребра, МэВ/фм."""
    return M_PROTON / (D_SPACE * side)


def string_tension_Y(side: float) -> float:
    """Натяжение при Y-конфигурации (три струны в точку Ферма).

    Для равностороннего треугольника суммарная длина Y-струны
    равна sqrt(3) * side.
    """
    return M_PROTON / (math.sqrt(3.0) * side)


def sigma_lattice_mev_fm() -> float:
    """Решёточное натяжение 0.18 ГэВ^2 в МэВ/фм: sigma / (hbar c)."""
    return SIGMA_LATTICE_GEV2 * 1e6 / HBAR_C


# ---------------------------------------------------------------------------
# 3. Принцип Паули как источник массы: счёт конфликтующих пар
# ---------------------------------------------------------------------------

def locked_pairs(n_cophased: int) -> int:
    """Число конфликтующих пар среди n_cophased синфазно захваченных мод.

    «Тождественность» в модели определяется НЕ флейвором, а состоянием
    моды: плоскостью намотки, знаком намотки и фазой. Мода, ушедшая в
    противофазу и на половинную частоту (d-маятник, 5.3), из конфликта
    выведена. Поэтому считаются пары среди синфазно захваченных мод.
    """
    return n_cophased * (n_cophased - 1) // 2


# Барионы: (метка, число синфазно захваченных мод, экспериментальная масса)
BARYONS = [
    ("p   (uud, S=1/2)", 2, M_PROTON),      # d ушёл в противофазу -> 1 пара
    ("n   (udd, S=1/2)", 2, M_NEUTRON),     # та же структура
    ("D++ (uuu, S=3/2)", 3, 1230.9),        # все три синфазны -> 3 пары
    ("D+  (uud, S=3/2)", 3, M_DELTA),       # спин 3/2: d тоже захвачен
    ("W-  (sss, S=3/2)", 3, M_OMEGA),       # 3 пары, но флейвор strange
]


def delta_prediction() -> dict:
    """Расщепление N-Delta из счёта паулиевских конфликтов.

    Гипотеза: каждый конфликт тождественных мод оплачивается энергией
    одного интерференционного плеча E_link = M_p c^2 / 6.
    Протон (uud): 1 конфликт. Delta^++ (uuu): 3 конфликта.
    Значит M(Delta) - M(N) = 2 * E_link = M_p c^2 / 3.
    """
    e_link = M_PROTON / N_LINKS
    n_p = locked_pairs(2)
    n_d = locked_pairs(3)
    split_pred = (n_d - n_p) * e_link
    split_exp = M_DELTA - M_NEUTRON / 2 - M_PROTON / 2  # среднее по нуклону
    return {
        "E_link": e_link,
        "pairs_p": n_p,
        "pairs_delta": n_d,
        "split_pred": split_pred,
        "split_exp": split_exp,
        "mass_delta_pred": M_PROTON + split_pred,
        "rel_err_split": (split_pred - split_exp) / split_exp,
        "rel_err_mass": (M_PROTON + split_pred - M_DELTA) / M_DELTA,
    }


# ---------------------------------------------------------------------------
# 4. Отчёт
# ---------------------------------------------------------------------------

def _row(name, value, unit, src, note=""):
    return f"  {name:<44s} {value:>12s} {unit:<7s} [{src}] {note}"


def report() -> None:
    lam = lambda_bar_proton()

    print("=" * 78)
    print("ProtonColor: конечная скорость цвета и геометрия протона")
    print("=" * 78)
    print()
    print("ВХОД (единственная размерная величина):")
    print(_row("масса протона M_p c^2", f"{M_PROTON:.5f}", "МэВ", "E", "PDG"))
    print(_row("приведённая комптоновская длина hbar/(M_p c)",
               f"{lam:.6f}", "фм", "M", "производная"))
    print()

    # --- 1. Три оценки расстояния между кварками ---
    print("-" * 78)
    print("1. РАССТОЯНИЕ МЕЖДУ КВАРКАМИ: три независимые оценки")
    print("-" * 78)
    print()

    phase_leg = 4.0 * math.pi / D_SPACE
    l_res = colour_resonance_length(phase_leg)
    l_res_half = colour_resonance_length(2.0 * math.pi / D_SPACE)
    l_pauli = pauli_kinetic_length()
    l_equi = equipartition_length()

    est = [
        ("(A) паулиевское запирание, 3 моды", l_pauli, "L = 3 * lambda"),
        ("(B) цветовой резонанс, delta_phi = 4pi/3", l_res, "L = (4pi/3) * lambda"),
        ("(C) равнораспределение по 6 каналам (5.6)", l_equi, "L = 6 * lambda"),
    ]
    print(f"  {'оценка':<44s} {'L, фм':>10s} {'L/lambda':>10s}   формула")
    for name, val, formula in est:
        print(f"  {name:<44s} {val:>10.4f} {val/lam:>10.4f}   {formula}")
    print()
    print(f"  побочная ветвь (B'), delta_phi = 2pi/3:      {l_res_half:10.4f} "
          f"{l_res_half/lam:10.4f}   замыкание за 2 обхода")
    print()
    print(f"  разброс оценок: {l_equi/l_pauli:.3f}x  (от {l_pauli:.3f} до {l_equi:.3f} фм)")
    print()

    # --- 1b. Скорость передачи цвета ---
    print("-" * 78)
    print("1b. ЧТО ЗА СКОРОСТЬ СТОИТ В ФОРМУЛЕ")
    print("-" * 78)
    print()
    corr = light_speed_corrections()
    print("  Поправки к локальной скорости света внутри протона:")
    print(_row("гравитационная, Phi/c^2 = GM_p/(r_p c^2)",
               f"{corr['grav']:.3e}", "", "M"))
    print(_row("радиус Шварцшильда протона",
               f"{corr['r_schwarzschild_fm']:.3e}", "фм", "M"))
    print(_row("упругая, u_proton / u_Planck",
               f"{corr['elastic']:.3e}", "", "M"))
    print(f"  (плотность энергии протона {corr['u_proton']:.1f} МэВ/фм^3 "
          f"против планковской {corr['u_planck']:.3e})")
    print()
    print("  Точность модели в этой главе — 6.7% (расщепление N-Delta) и")
    print("  множитель 2 (T-23). Поправки на 37 и 77 порядков ниже: локальная")
    print("  скорость света внутри протона неотличима от c.")
    print()
    print("  Но в формулу входит не скорость света в среде, а скорость сигнала")
    print("  ВДОЛЬ УДЕРЖИВАЮЩЕГО КАНАЛА: v_c = c / chi, где chi >= 1 — фактор")
    print("  извилистости и волноводной дисперсии. Это эффект геометрии канала,")
    print("  а не свойства среды, и он порядка единицы.")
    print()
    print(f"  {'оценка':<44s} {'chi = c/v':>10s} {'v/c':>8s}   статус")
    for name, val, _ in est:
        mult = val / lam
        chi = channel_speed_factor(mult)
        status = "допустимо" if chi >= 1.0 else "СВЕРХСВЕТОВАЯ — запрещено"
        print(f"  {name:<44s} {chi:>10.4f} {1/chi:>8.4f}   {status}")
    print()
    print(f"  Следствие: v_c <= c даёт верхнюю границу L <= {l_res:.4f} фм.")
    print("  Вариант (C) требует передачи цвета со скоростью 1.43 c и отпадает.")
    print()

    # --- 2. Сопоставление с зарядовым радиусом ---
    print("-" * 78)
    print("2. ЭТО НЕ РАДИУС ПРОТОНА")
    print("-" * 78)
    print()
    r_q = circumradius(l_res)
    print(_row("сторона треугольника L (вариант B)", f"{l_res:.4f}", "фм", "M"))
    print(_row("радиус триады R_q = L/sqrt(3)", f"{r_q:.4f}", "фм", "M"))
    print(_row("зарядовый радиус r_p (эксперимент)",
               f"{R_PROTON_EXP:.4f}", "фм", "C", f"+- {R_PROTON_ERR}"))
    print(_row("отношение R_q / r_p", f"{r_q/R_PROTON_EXP:.4f}", "", "M",
               "кварки внутри зарядового облака" if r_q < R_PROTON_EXP
               else "ПРОТИВОРЕЧИЕ: кварки вне облака"))
    print()
    print("  Числовое наблюдение (НЕ вывод): r_p / lambda = "
          f"{R_PROTON_EXP/lam:.4f}, то есть r_p ~ 4 * lambda.")
    print(f"  4 * lambda = {4*lam:.5f} фм против {R_PROTON_EXP} фм — "
          f"отклонение {100*(4*lam-R_PROTON_EXP)/R_PROTON_EXP:+.3f}% "
          f"({abs(4*lam-R_PROTON_EXP)/R_PROTON_ERR:.2f} sigma).")
    print("  Это переформулировка наблюдения T-8 (множитель 3/2), а не новый результат.")
    print()

    # --- 3. Время цветового оборота ---
    print("-" * 78)
    print("3. ВРЕМЯ ЦВЕТОВОГО ОБОРОТА И ВРЕМЯ ЖИЗНИ Delta")
    print("-" * 78)
    print()
    for name, val, _ in est:
        t = colour_period(val)
        print(f"  {name:<44s} T = {t:.4e} с")
    print()
    t_col = colour_period(l_res)
    tau_delta = HBAR_MEV_S / GAMMA_DELTA
    print(_row("T (цветовой оборот, вариант B)", f"{t_col:.4e}", "с", "M"))
    print(_row("tau(Delta) = hbar / Gamma", f"{tau_delta:.4e}", "с", "C",
               f"Gamma = {GAMMA_DELTA} МэВ"))
    print(_row("tau(Delta) / T", f"{tau_delta/t_col:.4f}", "", "M",
               "< 1: распад раньше первого замыкания"
               if tau_delta < t_col else "> 1: успевает замкнуться"))
    print()

    # --- 4. Натяжение канала ---
    print("-" * 78)
    print("4. НАТЯЖЕНИЕ ЦВЕТОВОГО КАНАЛА")
    print("-" * 78)
    print()
    sig_lat = sigma_lattice_mev_fm()
    print(f"  решёточное значение: sigma = {SIGMA_LATTICE_GEV2} ГэВ^2 = "
          f"{sig_lat:.1f} МэВ/фм   [C]")
    print()
    print(f"  {'оценка L':<44s} {'треугольник':>14s} {'Y-струна':>14s}")
    for name, val, _ in est:
        s_tri = string_tension_triangle(val)
        s_y = string_tension_Y(val)
        print(f"  {name:<44s} {s_tri:>10.1f} МэВ/фм {s_y:>10.1f} МэВ/фм")
    print()
    for name, val, _ in est:
        s_y = string_tension_Y(val)
        print(f"  {name:<44s} Y / решётка = {s_y/sig_lat:.3f}")
    print()
    print("  Оговорка: сюда уложена ВСЯ энергия покоя, тогда как часть её")
    print("  кинетическая. Поэтому модельное sigma — верхняя оценка, и совпадение")
    print("  на уровне единиц процентов переоценивать нельзя.")
    print()

    # --- 5. Энергия связи ---
    print("-" * 78)
    print("5. ЭНЕРГИЯ СВЯЗИ НА КАНАЛ")
    print("-" * 78)
    print()
    e_link = M_PROTON / N_LINKS
    e_leg = M_PROTON / D_SPACE
    print(_row("E_link = M_p c^2 / 6 (6 каналов, 5.6)", f"{e_link:.3f}", "МэВ", "M"))
    print(_row("E_leg  = M_p c^2 / 3 (3 внешних ребра)", f"{e_leg:.3f}", "МэВ", "M"))
    print(_row("Lambda_QCD^(3) (MS-bar, для масштаба)", "~332", "МэВ", "C",
               "схемозависима"))
    print(_row("m(pi+) (порог разрыва канала)", f"{M_PION_CH:.3f}", "МэВ", "C"))
    print()

    # --- 6. Паулиевский счёт: N-Delta ---
    print("-" * 78)
    print("6. ПРИНЦИП ПАУЛИ КАК ИСТОЧНИК МАССЫ: счёт конфликтующих пар")
    print("-" * 78)
    print()
    d = delta_prediction()
    print("  «Тождественность» задаётся состоянием моды (плоскость, знак намотки,")
    print("  фаза), а не флейвором. d-маятник протона в противофазе и на половинной")
    print("  частоте (5.3) — из конфликта выведен; при спине 3/2 он захвачен синфазно.")
    print()
    print(f"  {'барион':<20s} {'синфазн.':>9s} {'пар':>5s} "
          f"{'предсказ., МэВ':>15s} {'опыт, МэВ':>11s} {'откл.':>8s}")
    m_core = M_PROTON - locked_pairs(2) * d["E_link"]
    for label, n_co, m_exp in BARYONS:
        npair = locked_pairs(n_co)
        m_pred = m_core + npair * d["E_link"]
        print(f"  {label:<20s} {n_co:>9d} {npair:>5d} "
              f"{m_pred:>15.1f} {m_exp:>11.2f} {100*(m_pred-m_exp)/m_exp:>+7.2f}%")
    print()
    print(f"  «ядро» M_core = M_p - 1 * E_link = {m_core:.2f} МэВ  [M]")
    print()
    print(_row("квант конфликта = E_link", f"{d['E_link']:.3f}", "МэВ", "M"))
    print(_row("предсказание M(Delta) - M(N)", f"{d['split_pred']:.3f}", "МэВ", "M"))
    print(_row("эксперимент  M(Delta) - M(N)", f"{d['split_exp']:.3f}", "МэВ", "C"))
    print(_row("относительная ошибка расщепления",
               f"{100*d['rel_err_split']:+.2f}", "%", "M"))
    print()
    print(_row("предсказание M(Delta)", f"{d['mass_delta_pred']:.3f}", "МэВ", "M",
               "= (4/3) M_p"))
    print(_row("эксперимент  M(Delta)", f"{M_DELTA:.3f}", "МэВ", "C"))
    print(_row("относительная ошибка массы",
               f"{100*d['rel_err_mass']:+.2f}", "%", "M"))
    print()
    print("  Что правило даёт и чего не даёт:")
    print(f"    + вырождение p/n: предсказано точное, опыт "
          f"{100*(M_NEUTRON-M_PROTON)/M_PROTON:+.3f}%;")
    print("    + вырождение всего Delta-мультиплета (D++ и D+ на одном уровне);")
    print(f"    + расщепление N-Delta: {100*d['rel_err_split']:+.2f}%;")
    print(f"    - Omega^-(sss): {100*(m_core + 3*d['E_link'] - M_OMEGA)/M_OMEGA:+.1f}% "
          f"— правило нарушено.")
    print()
    print("  Провал на Omega^- диагностичен: у модели нет флейвора (странности),")
    print("  а счёт конфликтов по построению от флейвора не зависит. Правило")
    print("  применимо только внутри u/d-сектора. См. T-25.")
    print()

    # --- 6b. Эквивалентность условия резонанса и периода d-маятника ---
    print("-" * 78)
    print("6b. РЕЗОНАНС: период d-маятника и время цветового оборота")
    print("-" * 78)
    print()
    omega = M_PROTON / HBAR_MEV_S            # рад/с, из E = hbar*omega
    omega_d = omega / 2.0                    # 5.3: omega_d = omega_u / 2
    t_pendulum = 2.0 * math.pi / omega_d
    print(_row("omega = M_p c^2 / hbar", f"{omega:.4e}", "рад/с", "M"))
    print(_row("omega_d = omega / 2 (5.3)", f"{omega_d:.4e}", "рад/с", "M"))
    print(_row("период d-маятника T_d = 2pi / omega_d", f"{t_pendulum:.4e}", "с", "M"))
    print(_row("время цветового оборота T = 3L/c", f"{t_col:.4e}", "с", "M"))
    print(_row("T_d / T", f"{t_pendulum/t_col:.6f}", "", "M",
               "тождественно 1"))
    print()
    print("  Это не второй независимый вывод, а переформулировка условия")
    print("  delta_phi = 4pi/3: замыкание спинорной фазы за один цветовой обход")
    print("  эквивалентно тому, что d-маятник делает ровно один взмах за оборот.")
    print()

    # --- 7. Резюме ---
    print("=" * 78)
    print("РЕЗЮМЕ")
    print("=" * 78)
    print(f"  расстояние между кварками   L   = {l_res:.3f} фм "
          f"(диапазон модели {l_pauli:.3f}...{l_equi:.3f})")
    print(f"  радиус кварковой триады     R_q = {r_q:.3f} фм")
    print(f"  зарядовый радиус (экспер.)  r_p = {R_PROTON_EXP:.4f} фм")
    print(f"  время цветового оборота     T   = {t_col:.3e} с")
    print(f"  энергия на канал            E   = {e_link:.1f}...{e_leg:.1f} МэВ")
    print(f"  натяжение канала (Y)        sig = {string_tension_Y(l_res):.0f} МэВ/фм "
          f"против решёточных {sig_lat:.0f}")
    print()


# ---------------------------------------------------------------------------
# 10. Фокусы на намотанной нити: вырождение, отдача и форма треугольника
# ---------------------------------------------------------------------------

import itertools


def wound_curve(theta, phase_shift=0.0, R_over_r=math.sqrt(3.0), r=1.0):
    """Точка намотанной нити: phi = theta/2 + phase_shift (мёбиусов шаг)."""
    R = R_over_r * r
    ph = 0.5 * theta + phase_shift
    rad = R + r * math.cos(ph)
    return (rad * math.cos(theta), rad * math.sin(theta), r * math.sin(ph))


def _dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def foci_geometry_report() -> None:
    print("-" * 78)
    print("10. ФОКУСЫ НА НАМОТАННОЙ НИТИ")
    print("-" * 78)
    print()

    ths = [4.0 * math.pi * i / 4000 for i in range(4001)]

    # --- 1. вырождение ---
    worst = max(_dist(wound_curve(t, math.pi),
                      wound_curve(t + 2.0 * math.pi, 0.0)) for t in ths)
    print(_row("max|C(th; +pi) - C(th+2pi; 0)|", f"{worst:.3e}", "", "M",
             "сдвиг фазы на pi = сдвиг theta на 2pi"))
    print()

    # --- 2. синфазная против дифференциальной отдачи ---
    print("  Отдача от d-моды: синфазная (как в Proton.py) против дифференциальной")
    print()
    print(f"  {'eps':>8s} {'синфазная':>14s} {'дифференциальная':>20s}")
    for eps in (0.0, 0.05, 0.10, 0.20, 0.39):
        same = max(_dist(wound_curve(t, math.pi + eps),
                         wound_curve(t + 2.0 * math.pi, eps)) for t in ths)
        diff = max(_dist(wound_curve(t, math.pi - eps),
                         wound_curve(t + 2.0 * math.pi, eps)) for t in ths)
        print(f"  {eps:>8.2f} {same:>14.3e} {diff:>20.4f}")
    print()
    print("  В Proton.py записано shift_u1 = u_recoil, shift_u2 = pi + u_recoil,")
    print("  то есть разность сдвигов тождественно равна pi при любом t. Такая")
    print("  отдача СИНФАЗНА и вырождения не снимает ни при какой амплитуде.")
    print("  Дифференциальная отдача разводит нити примерно на 2*eps*r.")
    print()

    # --- 3. форма треугольника при равном шаге по theta ---
    print("  Три фокуса при равном шаге по theta: равносторонний ли треугольник?")
    print()
    print(f"  {'R/r':>8s} {'стороны (в единицах r)':>34s} {'max/min':>9s} {'разброс':>9s}")
    for ratio in (math.sqrt(3.0), 2.0, 3.0, 5.0, 10.0, 30.0):
        P = [wound_curve(t, 0.0, ratio)
             for t in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)]
        d = [_dist(P[i], P[j]) for i, j in itertools.combinations(range(3), 2)]
        mean = sum(d) / 3.0
        s = " ".join(f"{x:6.3f}" for x in sorted(d, reverse=True))
        print(f"  {ratio:>8.3f} {s:>34s} {max(d)/min(d):>9.3f} "
              f"{100*(max(d)-min(d))/mean:>8.1f}%")
    print()
    print("  Равносторонним треугольник становится только при R >> r. Модель же")
    print("  принимает R = sqrt(3) r, где разброс сторон 37%.")
    print()

    # --- 4. что это меняет в числах 5.9 ---
    P = [wound_curve(t) for t in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)]
    d = [_dist(P[i], P[j]) for i, j in itertools.combinations(range(3), 2)]
    a, b, c = d
    per = a + b + c
    mean_edge = per / 3.0
    s = per / 2.0
    K = math.sqrt(s * (s - a) * (s - b) * (s - c))
    R_circ = a * b * c / (4.0 * K)
    axis_dist = [math.hypot(p[0], p[1]) for p in P]

    L_FM = 4.0 * math.pi / 3.0 * lambda_bar_proton()
    scale = L_FM / mean_edge          # среднее ребро отождествляем с L из 5.9

    print("  Пересчёт величин подглавы 5.9 на реальную (неравностороннюю) форму:")
    print()
    print(_row("среднее ребро = периметр/3", f"{mean_edge:.4f}", "r", "M",
             f"= L = {L_FM:.4f} фм"))
    print(_row("радиус описанной окружности", f"{R_circ:.4f}", "r", "M",
             f"= {R_circ*scale:.4f} фм"))
    print(_row("то же для равностороннего, L/sqrt3", f"{mean_edge/math.sqrt(3):.4f}",
             "r", "M", f"= {mean_edge/math.sqrt(3)*scale:.4f} фм"))
    print(_row("отклонение", f"{100*(R_circ-mean_edge/math.sqrt(3))/(mean_edge/math.sqrt(3)):+.1f}",
             "%", "M"))
    print()
    print("  Но расстояния фокусов от ОСИ тора вообще не равны:")
    for i, ad in enumerate(sorted(axis_dist, reverse=True), 1):
        print(f"    фокус {i}: {ad:.4f} r = {ad*scale:.4f} фм")
    print(f"    отношение max/min = {max(axis_dist)/min(axis_dist):.3f}")
    print()
    print(f"  Против зарядового радиуса r_p = {R_PROTON_EXP} фм: все три фокуса")
    print("  лежат внутри облака, но «радиус триады» как ОДНО число не определён —")
    print("  корректная запись есть диапазон "
          f"{min(axis_dist)*scale:.2f}...{max(axis_dist)*scale:.2f} фм.")
    print()
    print("  Качественный вывод 5.9 (фокусы внутри зарядового облака) устоял;")
    print("  количественная запись R_q = 0.509 фм требует замены на диапазон.")
    print()



if __name__ == "__main__":
    report()
    foci_geometry_report()
