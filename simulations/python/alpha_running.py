"""Бег постоянной тонкой структуры 1/alpha по эпохам Вселенной.

ЧТО ИЗМЕНИЛОСЬ ПО СРАВНЕНИЮ СО СТАРОЙ ВЕРСИЕЙ (см. CODE_REVIEW.md §2.1)
----------------------------------------------------------------------
Старый скрипт на каждой эпохе решал k из условия 1/alpha == CODATA:

    k_adapted = sqrt(deficit / qcd_base)

Так как вклад КХД пропорционален k**2 по построению, эта подстановка
АЛГЕБРАИЧЕСКИ гарантировала точное совпадение при любых входных данных —
и вывод это подтверждал: 137.035999 во всех шести эпохах до последнего знака.
Плюс УФ-член сокращался сам с собой (C_Gouy = sqrt3*ln(M_P/M_EW) - 105.81,
после вычитания оставалась голая константа 105.81).

Здесь ровно один свободный параметр — аддитивная калибровка C_CALIB, —
и он фиксируется ОДИН РАЗ по сегодняшней эпохе (a = 1). Все остальные
эпохи после этого предсказываются, а не подгоняются.

Ключевое наблюдение: величина Delta alpha / alpha не зависит ни от C_CALIB,
ни от УФ-члена, ни от фермионной суммы, ни от бозонных вкладов — все
статические слагаемые сокращаются в разности. Предсказание модели целиком
определяется двумя слагаемыми, зависящими от масштабного фактора:
ИК-добегом по горизонту и термическим экранированием КХД-конденсата.
Именно поэтому его можно честно сравнить с наблюдениями.

Запуск:
    python alpha_running.py                # таблица + график
    python alpha_running.py --no-plot      # только таблица
    python alpha_running.py --theme dark --save alpha_running.png
"""

import argparse
import math

# --- Фундаментальные константы ---
M_EW = 246.22e9          # Электрослабый масштаб вакуума, эВ
M_P = 1.22089e19         # Планковская масса, эВ
LAMBDA_QCD = 200e6       # Масштаб конфайнмента, эВ
R_PLANCK = 1.616255e-35  # Планковская длина, м
R_HUBBLE = 1.374e26      # Хаббловский радиус сегодня, м

# CODATA 2022
INV_ALPHA_CODATA = 137.035999177

# Постоянная Хаббла, 1/год (H0 = 67.4 км/с/Мпк)
H0_PER_YEAR = 67.4e3 / 3.0857e22 * 3.1557e7

DELTA_GLOBAL = math.radians(11.099)     # глобальная фаза модели
THETA_W = math.radians(28.71)           # угол Вайнберга
COS2_THETA_W = math.cos(THETA_W) ** 2
SQRT3 = math.sqrt(3.0)

# 3+1D объёмная нормировка ИК-добега
C_COSMO = 1.0 / (2.0 * math.pi ** 2)

T_NOW_EV = 2.348e-4      # Реликтовое излучение сегодня (~2.725 K), эВ
T_CRITICAL_EV = 150e6    # Температура КХД-перехода (~150 МэВ), эВ

# Полный фермионный спектр (массы: PDG, МэВ)
FERMIONS = [
    {"name": "e",   "m_MeV": 0.510998,  "n": 0, "q": -1.0,   "type": "lepton"},
    {"name": "mu",  "m_MeV": 105.65837, "n": 1, "q": -1.0,   "type": "lepton"},
    {"name": "tau", "m_MeV": 1776.86,   "n": 2, "q": -1.0,   "type": "lepton"},
    {"name": "u",   "m_MeV": 2.16,      "n": 1, "q": +2 / 3, "type": "quark"},
    {"name": "d",   "m_MeV": 4.67,      "n": 2, "q": -1 / 3, "type": "quark"},
    {"name": "s",   "m_MeV": 93.4,      "n": 0, "q": -1 / 3, "type": "quark"},
    {"name": "c",   "m_MeV": 1270.0,    "n": 2, "q": +2 / 3, "type": "quark"},
    {"name": "b",   "m_MeV": 4180.0,    "n": 0, "q": -1 / 3, "type": "quark"},
    {"name": "t",   "m_MeV": 172760.0,  "n": 1, "q": +2 / 3, "type": "quark"},
]

M_W = 80.38e9
M_Z = 91.19e9


# --------------------------------------------------------------------------
# Койде: аналитическое решение вместо перебора по 3600 углам
# --------------------------------------------------------------------------

def koide_fit(m1: float, m2: float, m3: float) -> tuple[float, float]:
    """Возвращает (невязка, фаза) наилучшей подгонки к окружности Койде.

    x_i = sqrt(m_i) / sum(sqrt(m)) сравнивается с cos(phi + 2*pi*i/3) / NORM.

    Перебор не нужен. Сумма квадратов трёх косинусов, разнесённых на 120
    градусов, тождественно равна 3/2, поэтому NORM = sqrt(3/2) — константа, а

        sum_i x_i * cos(phi + 2*pi*i/3) = |(A, B)| * cos(phi + atan2(B, A)),
        A = sum_i x_i*cos(2*pi*i/3),  B = sum_i x_i*sin(2*pi*i/3)

    максимальна при phi = -atan2(B, A). Совпадает с прежним перебором
    до 1e-11 и считается примерно в 1e5 раз быстрее.
    """
    sq = [math.sqrt(m) for m in (m1, m2, m3)]
    s = sum(sq)
    if s == 0.0:
        return 0.0, 0.0
    x = [v / s for v in sq]

    norm = math.sqrt(1.5)
    a = sum(xi * math.cos(2 * math.pi * i / 3) for i, xi in enumerate(x))
    b = sum(xi * math.sin(2 * math.pi * i / 3) for i, xi in enumerate(x))

    amp = math.hypot(a, b)
    err = sum(xi * xi for xi in x) - 2.0 * amp / norm + 1.0
    phi = (-math.atan2(b, a)) % (2 * math.pi)
    return math.sqrt(max(err, 0.0)), phi


def build_delta_f_map() -> dict:
    """Фазовая поправка на фермион = невязка Койде его поколения, делённая на 3.

    ВНИМАНИЕ: используется именно НЕВЯЗКА подгонки, а не фаза. Так было и в
    исходной версии; сохранено намеренно, но стоит проверить, что это то,
    что задумывалось (см. CODE_REVIEW.md §2.1в).
    """
    groups = {
        "leptons": ("e", "mu", "tau"),
        "up_quarks": ("u", "c", "t"),
        "down_quarks": ("d", "s", "b"),
    }
    by_name = {f["name"]: f for f in FERMIONS}
    out = {}
    for names in groups.values():
        residual, _phase = koide_fit(*(by_name[n]["m_MeV"] for n in names))
        for n in names:
            out[n] = residual / 3.0
    return out


DELTA_F = build_delta_f_map()


# --------------------------------------------------------------------------
# Статические (не зависящие от эпохи) слагаемые
# --------------------------------------------------------------------------

def uv_term() -> float:
    """УФ-добег от планковского масштаба до электрослабого."""
    return SQRT3 * math.log(M_P / M_EW)


def fermion_term() -> float:
    """Сумма по фермионным петлям с фазами Койде."""
    total = 0.0
    for f in FERMIONS:
        phase = DELTA_GLOBAL + 2.0 * math.pi * f["n"] / 3.0 + DELTA_F[f["name"]]
        n_c = 3 if f["type"] == "quark" else 1
        total += 2.0 * SQRT3 * n_c * (f["q"] ** 2) * math.cos(phase)
    return total


def weak_boson_term(m_boson: float) -> float:
    """Вклад слабого бозона. Для Z (m > 85 ГэВ) добавляется cos^2(theta_W)."""
    c_a_su2 = 2.0
    weight = c_a_su2 * (COS2_THETA_W if m_boson > 85e9 else 1.0)
    return weight * math.cos(DELTA_GLOBAL) * math.log(M_EW / m_boson)


STATIC_TERM = uv_term() + fermion_term() + weak_boson_term(M_W) + weak_boson_term(M_Z)


# --------------------------------------------------------------------------
# Слагаемые, зависящие от масштабного фактора a = 1/(1+z)
# --------------------------------------------------------------------------

def ir_horizon_term(a: float, coeff: float = 1.0) -> float:
    """ИК-добег по горизонту. R(a) = a * R_Hubble, поэтому член линеен по ln(a).

    coeff — множитель к C_COSMO, нужен только для оценки, насколько слабее
    должна быть нормировка, чтобы модель прошла наблюдательные пределы.
    """
    return (coeff * C_COSMO * SQRT3 * math.tan(THETA_W)
            * math.log(a * R_HUBBLE / R_PLANCK))


def _trace_omega_unit() -> float:
    """След неабелевой связности при k = 1. Считается один раз при импорте.

    tr_Omega пропорционален k**2, поэтому масштабировать матрицы на каждом
    вызове (как делала старая версия) не нужно.
    """
    omega = {
        1: [[0.0, 0.5, -0.2], [-0.5, 0.0, 0.1], [0.2, -0.1, 0.0]],
        2: [[0.0, -0.1, 0.4], [0.1, 0.0, -0.3], [-0.4, 0.3, 0.0]],
        3: [[0.0, 0.3, 0.1], [-0.3, 0.0, 0.5], [-0.1, -0.5, 0.0]],
    }
    total = 0.0
    for idx in omega:
        nxt = (idx % 3) + 1
        prv = ((idx - 2) % 3) + 1
        # ковариантная производная тождественно 0 для антисимметричных матриц,
        # остаётся только коммутаторный член
        total += sum(omega[nxt][i][j] * omega[prv][j][i]
                     for i in range(3) for j in range(3))
    return total


TRACE_OMEGA_UNIT = _trace_omega_unit()

QCD_BASE_UNIT = (abs(TRACE_OMEGA_UNIT) * (2.0 / math.pi)
                 * math.cos(DELTA_GLOBAL) * math.log(M_EW / LAMBDA_QCD))


def qcd_thermal_term(a: float, k: float = 1.0) -> float:
    """КХД-вклад с термическим экранированием конденсата. Пропорционален k**2.

    T(a) = T_now / a; при T -> T_critical конденсат плавится и вклад гаснет.
    """
    t_current = T_NOW_EV / a
    screening = math.exp(-(t_current ** 2) / (T_CRITICAL_EV ** 2))
    return QCD_BASE_UNIT * (k ** 2) * screening


# --------------------------------------------------------------------------
# Модель целиком
# --------------------------------------------------------------------------

K_QCD = 1.0   # ЗАФИКСИРОВАН. Раньше решался из ответа на каждой эпохе.

#: Единственный свободный параметр: аддитивная калибровка по сегодняшней эпохе.
C_CALIB = INV_ALPHA_CODATA - (STATIC_TERM
                              + ir_horizon_term(1.0)
                              + qcd_thermal_term(1.0, K_QCD))


def inv_alpha(a: float, k: float = K_QCD, ir_coeff: float = 1.0) -> float:
    """1/alpha при масштабном факторе a = 1/(1+z)."""
    return (C_CALIB + STATIC_TERM
            + ir_horizon_term(a, ir_coeff)
            + qcd_thermal_term(a, k))


def delta_inv_alpha(a: float, k: float = K_QCD, ir_coeff: float = 1.0) -> float:
    """1/alpha(a) - 1/alpha(1) — величина, свободная от калибровки.

    Все статические слагаемые и C_CALIB сокращаются тождественно, остаются
    только два зависящих от эпохи члена. Именно это и есть предсказание
    модели в чистом виде; оно строго линейно по ir_coeff.
    """
    return ((ir_horizon_term(a, ir_coeff) - ir_horizon_term(1.0, ir_coeff))
            + (qcd_thermal_term(a, k) - qcd_thermal_term(1.0, k)))


def delta_alpha_over_alpha(a: float, k: float = K_QCD, ir_coeff: float = 1.0) -> float:
    """(alpha(a) - alpha_today) / alpha_today — наблюдаемая величина.

    Формула точная, без линеаризации. От калибровки зависит только через
    общую нормировку 1/alpha ~ 137 в знаменателе (C_CALIB зафиксирован по
    CODATA), тогда как ФОРМА зависимости от эпохи задаётся исключительно
    delta_inv_alpha() и никак не подгоняется.
    """
    inv_today = inv_alpha(1.0, k, ir_coeff)
    return inv_today / (inv_today + delta_inv_alpha(a, k, ir_coeff)) - 1.0


def drift_rate_today(ir_coeff: float = 1.0) -> float:
    """d(alpha)/dt / alpha сегодня, 1/год.

    d(Delta a/a)/dt = -(1/inv_alpha) * d(ir_term)/dln(a) * H0.
    """
    slope = ir_coeff * C_COSMO * SQRT3 * math.tan(THETA_W)
    return -(slope / INV_ALPHA_CODATA) * H0_PER_YEAR


# --------------------------------------------------------------------------
# Наблюдательные пределы на |Delta alpha / alpha|
# --------------------------------------------------------------------------

OBSERVATIONS = [
    # (метка, z, предел |Delta alpha/alpha| (1 sigma), источник)
    ("Оклo (реактор)", 0.14, 1.0e-8,
     "Davis & Hamdan, arXiv:1510.00856 — < ~10 ppb при z ~ 0.14"),
    # Одна квазарная точка, а не две: одиночное измерение по HE 0515-4414
    # (1.3 +- 1.3 ppm, z = 1.15) входит в комбинированную выборку, поэтому
    # ставить обе — двойной учёт.
    ("Квазары (29 систем)", 2.4, 0.6e-6,
     "Murphy et al. 2022, A&A 658 A123 — -0.5 +- 0.5stat +- 0.4sys ppm, z = 0.6-2.4"),
    ("CMB (Planck)", 1100.0, 3.7e-3,
     "Planck int. XXIV, arXiv:1406.7482 — Delta a/a = (3.6 +- 3.7)e-3"),
    ("BBN", 4.0e8, 1.2e-3,
     "Deal & Martins 2020, A&A 633 A91 — (0.2 +- 1.2)e-3 без лития"),
]

#: Предел на дрейф сегодня: 1.0(1.1)e-18 /год (Lange et al., PRL 126 011102).
CLOCK_LIMIT_PER_YEAR = 1.1e-18


def evaluate() -> list:
    """Сравнивает предсказание с каждым пределом."""
    rows = []
    for label, z, bound, source in OBSERVATIONS:
        a = 1.0 / (1.0 + z)
        pred = delta_alpha_over_alpha(a)
        ratio = abs(pred) / bound
        rows.append({"label": label, "z": z, "pred": pred, "bound": bound,
                     "ratio": ratio, "ok": ratio <= 1.0, "source": source})
    return rows


def max_allowed_ir_coeff(rows) -> float:
    """Во сколько раз слабее должна быть нормировка ИК-члена, чтобы пройти всё.

    Delta alpha/alpha линейна по coeff, поэтому достаточно взять
    минимум 1/ratio по всем пределам (включая часы).
    """
    limits = [1.0 / r["ratio"] for r in rows if r["ratio"] > 0]
    clock_ratio = abs(drift_rate_today()) / CLOCK_LIMIT_PER_YEAR
    if clock_ratio > 0:
        limits.append(1.0 / clock_ratio)
    return min(limits) if limits else float("inf")


def _times(x: float) -> str:
    return f"x{x:,.0f}".replace(",", " ")


def report() -> list:
    rows = evaluate()

    print("=" * 78)
    print("БЕГ 1/alpha ПО ЭПОХАМ: ПРЕДСКАЗАНИЕ И СРАВНЕНИЕ С НАБЛЮДЕНИЯМИ")
    print("=" * 78)
    print(f"Статические слагаемые   : {STATIC_TERM:.6f}")
    print(f"  из них УФ-добег       : {uv_term():.6f}")
    print(f"  фермионные петли      : {fermion_term():.6f}")
    print(f"  W + Z                 : {weak_boson_term(M_W) + weak_boson_term(M_Z):.6f}")
    print(f"ИК-добег по горизонту   : {ir_horizon_term(1.0):.6f}  (сегодня)")
    print(f"КХД при k = {K_QCD:<12g}: {qcd_thermal_term(1.0, K_QCD):.6f}  (сегодня)")
    print(f"Калибровка C_CALIB      : {C_CALIB:.6f}   <-- единственный свободный параметр")
    print(f"Итог сегодня            : 1/alpha = {inv_alpha(1.0):.9f}")
    print(f"CODATA 2022             : 1/alpha = {INV_ALPHA_CODATA:.9f}")
    print()
    print("Ниже — ПРЕДСКАЗАНИЕ. Ни одна из этих строк не подгонялась: статические")
    print("слагаемые и C_CALIB сокращаются в разности Delta alpha / alpha.")
    print()

    print(f"{'эпоха':<20}{'z':>12}{'предсказание':>16}{'предел':>12}{'превышение':>14}")
    print("-" * 78)
    for r in rows:
        verdict = "OK" if r["ok"] else _times(r["ratio"])
        print(f"{r['label']:<20}{r['z']:>12.4g}{r['pred']:>16.3e}"
              f"{r['bound']:>12.1e}{verdict:>14}")

    rate = drift_rate_today()
    clock_ratio = abs(rate) / CLOCK_LIMIT_PER_YEAR
    clock_verdict = "OK" if clock_ratio <= 1 else _times(clock_ratio)
    print("-" * 78)
    print(f"{'Атомные часы, 1/год':<20}{0:>12.4g}{rate:>16.3e}"
          f"{CLOCK_LIMIT_PER_YEAR:>12.1e}{clock_verdict:>14}")
    print()

    failed = [r for r in rows if not r["ok"]]
    if failed or clock_ratio > 1:
        worst = max([r["ratio"] for r in rows] + [clock_ratio])
        print(f"ВЫВОД: модель в текущей нормировке ИСКЛЮЧЕНА. "
              f"Худшее превышение: {_times(worst)}")
        print()
        print("Источник расхождения — ИК-добег: он линеен по ln(a), то есть alpha")
        print("дрейфует на всём протяжении космологической истории, тогда как")
        print("лабораторные и квазарные данные требуют почти полной стабильности.")
        print("КХД-член включается только при T ~ T_c (z > 1e10) и на низких z")
        print("не работает вовсе, компенсировать дрейф ему нечем.")
        print()
        allowed = max_allowed_ir_coeff(rows)
        print("Чтобы пройти ВСЕ пределы, коэффициент ИК-члена должен быть не более")
        print(f"{allowed:.2e} от нынешнего, то есть C_cosmo <= {C_COSMO * allowed:.3e}")
        print(f"вместо 1/(2*pi^2) = {C_COSMO:.6f}.")
        print()
        print("Физическая альтернатива: если под R в ИК-члене понимать СОПУТСТВУЮЩИЙ")
        print("горизонт (фиксированный), а не собственный (растущий как a), член")
        print("перестаёт зависеть от эпохи и все пределы выполняются тождественно —")
        print("но тогда модель не предсказывает и никакого бега alpha.")
    else:
        print("ВЫВОД: модель совместима со всеми учтёнными пределами.")

    print()
    print("Источники пределов:")
    for r in rows:
        print(f"  {r['label']:<20} {r['source']}")
    print(f"  {'Атомные часы':<20} Lange et al. 2021, PRL 126 011102 — "
          "1.0(1.1)e-18 /год")
    return rows


# --------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------

# Палитра проверена валидатором (light и dark):
# CVD dE 23.8 / 25.7, normal-vision dE 31.6 / 31.9, контраст >= 3:1
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "pred": "#2a78d6", "bound": "#d03b3b"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "pred": "#3987e5", "bound": "#d03b3b"},
}


#: короткие подписи для графика (полные — в таблице stdout)
SHORT_LABELS = {
    "Оклo (реактор)": "Оклo",
    "Квазары (29 систем)": "квазары",
    "CMB (Planck)": "CMB",
    "BBN": "BBN",
}


def make_plot(rows, theme: str = "light", out_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    t = THEMES[theme]

    z_grid = np.logspace(-2, 12, 600)
    pred = np.array([abs(delta_alpha_over_alpha(1.0 / (1.0 + z))) for z in z_grid])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    ax.plot(z_grid, pred, color=t["pred"], lw=2.0, zorder=4,
            label=r"Предсказание модели  $|\Delta\alpha/\alpha|$")

    z_b = np.array([r["z"] for r in rows])
    b_b = np.array([r["bound"] for r in rows])
    ax.plot(z_b, b_b, linestyle="none", marker="v", markersize=9,
            color=t["bound"], zorder=5,
            label="Наблюдательный предел (модель должна быть ниже)")

    # Вертикальный вынос от предела до кривой — это и есть превышение.
    # Подписи разносим по высоте, поэтому они не сталкиваются даже у
    # двух близких квазарных точек.
    for i, r in enumerate(rows):
        p = abs(delta_alpha_over_alpha(1.0 / (1.0 + r["z"])))
        if not r["ok"]:
            ax.vlines(r["z"], r["bound"], p, color=t["bound"],
                      linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.8, zorder=3)
            ax.annotate(f"×{r['ratio']:,.0f}".replace(",", " "),
                        xy=(r["z"], math.sqrt(r["bound"] * p)),
                        xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9,
                        color=t["bound"], zorder=6)
        # имя предела — под маркером, со ступенчатым отступом
        ax.annotate(SHORT_LABELS.get(r["label"], r["label"]),
                    xy=(r["z"], r["bound"]),
                    xytext=(0, -13),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9, color=t["muted"], zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 2e12)
    ax.set_ylim(1e-10, 3e-1)
    ax.set_xlabel("Красное смещение z", color=t["muted"], fontsize=10)
    ax.set_ylabel(r"$|\Delta\alpha/\alpha|$", color=t["muted"], fontsize=10)
    ax.set_title("Бег постоянной тонкой структуры: модель против наблюдений",
                 color=t["ink"], fontsize=13, pad=14)

    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=t["grid"], linewidth=0.4, alpha=0.5, zorder=0)
    ax.tick_params(colors=t["muted"], labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for text in leg.get_texts():
        text.set_color(t["ink"])

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=t["surface"])
        print(f"\nГрафик сохранён: {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Бег 1/alpha по эпохам: предсказание против наблюдений")
    parser.add_argument("--no-plot", action="store_true", help="только таблица")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--save", metavar="PATH", help="сохранить график в файл")
    args = parser.parse_args()

    rows = report()

    if not args.no_plot:
        import matplotlib.pyplot as plt
        make_plot(rows, theme=args.theme, out_path=args.save)
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
