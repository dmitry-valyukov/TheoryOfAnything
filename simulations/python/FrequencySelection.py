"""Правило отбора частоты (T-15): что выделяет точки в окне Q-шара.

После 2.8 отбор частоты — единственный оставшийся кандидат на механизм
различия поколений, и у него впервые есть вычислимый носитель: однопара-
метрическое семейство солитонов (Q, E(Q), mu(Q)) в секторах намотки.
Модуль строит эту кривую по результатам TorusBVP.py и проверяет ЧЕСТНЫЙ
вопрос: есть ли на кривой естественные выделенные точки, отношения
энергий которых дают лептонные 206.77 и 3477.23.

Аналитические якоря (g = 2, h = 1):
    плотность плато        rho_d  = 3g/4h            = 1.5
    толстостенный предел   mu_inf = -g rho_d + h rho_d^2 = -3g^2/16h = -0.75
    окно частот            mu in (mu_inf, 0)
    в пределе Q -> inf     E/Q -> mu_inf  (нулевое давление капли)

Кандидаты в выделенные точки ("ориентиры"):
    A  порог связывания сектора n=1:  E_1(Q) = 0
    B  пересечение уровней Q_x:       E_1 = E_0 (намотка = основное состояние)
    C  складка Вахитова - Колоколова: экстремум Q(mu), если есть
    D  положение mu_n лестницы секторов в окне (данные 2.8 при Q = 740)

Разделы:
  1. кривые E(Q), mu(Q) по секторам, контроль асимптоты mu -> mu_inf
  2. ориентиры на кривой и энергии в них
  3. решётка отношений |E| и dE между ориентирами против 206.77 / 3477.23
  4. вердикт и что остаётся правилу отбора (мост к T-31)

Требует накопленных записей в torusbvp_results.json (скан по Q делает
TorusBVP.relax при подменённом Q_FIX; см. freqsel_scan.log).

Запуск:  python FrequencySelection.py            — анализ + текстовый отчёт
         python FrequencySelection.py --plot     — плюс theory/images/freqsel-window.png
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "torusbvp_results.json")

G_CUB, H_QUI = 2.0, 1.0
RHO_D = 3 * G_CUB / (4 * H_QUI)
MU_INF = -G_CUB * RHO_D + H_QUI * RHO_D ** 2          # = -3 g^2 / 16 h
M_MU_ME, M_TAU_ME = 206.768, 3477.23


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def load_curves():
    data = json.load(open(RESULTS, encoding="utf-8"))
    best = {}
    for r in data.values():
        key = (int(round(r["Q"])), r["n"])
        if key not in best or r["E"] < best[key]["E"]:
            best[key] = r
    curves = {}
    for n in (0, 1):
        pts = sorted((q, r) for (q, m), r in best.items() if m == n)
        curves[n] = dict(Q=np.array([q for q, _ in pts], float),
                         E=np.array([r["E"] for _, r in pts]),
                         mu=np.array([r["mu"] for _, r in pts]))
    return curves


def cross_zero(x, y, mask=None):
    """Корни y(x) = 0 линейной интерполяцией; mask сужает поиск."""
    roots = []
    for i in range(len(x) - 1):
        if mask is not None and not (mask[i] and mask[i + 1]):
            continue
        if y[i] * y[i + 1] < 0:
            t = y[i] / (y[i] - y[i + 1])
            roots.append(x[i] + t * (x[i + 1] - x[i]))
    return roots


def interp(x, y, x0):
    return float(np.interp(x0, x, y))


def section_curves(c):
    head("1. КРИВЫЕ E(Q) И mu(Q); КОНТРОЛЬ АСИМПТОТЫ")
    print(f"\n  аналитически: rho_d = {RHO_D}, окно mu in ({MU_INF}, 0), "
          f"E/Q -> {MU_INF} при Q -> inf\n")
    print(f"  {'Q':>6} {'E0':>10} {'mu0':>8} {'E0/Q':>8}   "
          f"{'E1':>10} {'mu1':>8} {'E1/Q':>8}")
    for i, q in enumerate(c[0]["Q"]):
        j = np.searchsorted(c[1]["Q"], q)
        has1 = j < len(c[1]["Q"]) and c[1]["Q"][j] == q
        e1 = f"{c[1]['E'][j]:10.2f} {c[1]['mu'][j]:8.4f} " \
             f"{c[1]['E'][j]/q:8.4f}" if has1 else " " * 28
        print(f"  {q:6.0f} {c[0]['E'][i]:10.2f} {c[0]['mu'][i]:8.4f} "
              f"{c[0]['E'][i]/q:8.4f}   {e1}")
    for n in (0, 1):
        mu_last = c[n]["mu"][-1]
        frac = mu_last / MU_INF
        print(f"\n  сектор n={n}: mu({c[n]['Q'][-1]:.0f}) = {mu_last:.4f} — "
              f"{frac:.0%} глубины окна (медленный подход ~ Q^(-1/3))")


def section_landmarks(c):
    head("2. ОРИЕНТИРЫ НА КРИВОЙ")
    L = {}
    # A: порог связывания n=1 (E1 = 0)
    for qA in cross_zero(c[1]["Q"], c[1]["E"])[:1]:
        L["A: порог E1=0"] = dict(Q=qA, E1=0.0,
                                  E0=interp(c[0]["Q"], c[0]["E"], qA),
                                  mu1=interp(c[1]["Q"], c[1]["mu"], qA))
    # B: пересечение уровней в СВЯЗАННОЙ области (оба E < 0);
    #    кроссинги в несвязанной области — артефакты порога, не берём
    qs = np.intersect1d(c[0]["Q"], c[1]["Q"])
    e0s = np.array([interp(c[0]["Q"], c[0]["E"], q) for q in qs])
    e1s = np.array([interp(c[1]["Q"], c[1]["E"], q) for q in qs])
    bound = (e0s < 0) & (e1s < 0)
    for qB in cross_zero(qs, e1s - e0s, mask=bound)[:1]:
        L["B: пересечение Q_x"] = dict(Q=qB,
                                       E1=interp(c[1]["Q"], c[1]["E"], qB),
                                       E0=interp(c[0]["Q"], c[0]["E"], qB),
                                       mu1=interp(c[1]["Q"], c[1]["mu"], qB))
    # E: порог локализации капли без намотки (mu0 = 0): ниже него
    #    локализован ТОЛЬКО намотанный тор
    for qE in cross_zero(c[0]["Q"], c[0]["mu"])[:1]:
        L["E: локализация n=0"] = dict(Q=qE,
                                       E1=interp(c[1]["Q"], c[1]["E"], qE),
                                       E0=interp(c[0]["Q"], c[0]["E"], qE),
                                       mu1=interp(c[1]["Q"], c[1]["mu"], qE))
    # C: складка ВК — экстремум Q(mu): ищем смену знака dQ/dmu
    mu, Q = c[1]["mu"], c[1]["Q"]
    order = np.argsort(mu)
    dQdmu = np.gradient(Q[order], mu[order])
    sign_change = np.where(np.diff(np.sign(dQdmu)))[0]
    if len(sign_change):
        k = sign_change[0]
        L["C: складка ВК"] = dict(Q=float(Q[order][k]),
                                  E1=float(c[1]["E"][order][k]),
                                  E0=float("nan"),
                                  mu1=float(mu[order][k]))
        vk = "ЕСТЬ смена знака dQ/dmu"
    else:
        vk = ("dQ/dmu не меняет знак на охваченном диапазоне — "
              "складки нет, ветвь монотонна")
    print(f"\n  Вахитов - Колоколов: {vk}")
    print(f"\n  {'ориентир':>22} {'Q':>8} {'E1':>10} {'E0':>10} {'mu1':>8} "
          f"{'mu1/mu_inf':>11}")
    for name, v in L.items():
        print(f"  {name:>22} {v['Q']:8.1f} {v['E1']:10.2f} "
              f"{v['E0']:10.2f} {v['mu1']:8.4f} {v['mu1']/MU_INF:11.2%}")
    # D: лестница mu_n при Q=740 в долях окна (из 2.8)
    print("\n  D: положение лестницы секторов в окне (Q = 740, из 2.8.5):")
    for n, mu_n in ((0, -0.4768), (1, -0.3816), (2, -0.3346), (3, -0.2132)):
        print(f"     n={n}: mu = {mu_n:7.4f}  ->  {mu_n/MU_INF:6.1%} глубины")
    return L


def section_ratios(c, L):
    head("3. РЕШЁТКА ОТНОШЕНИЙ МЕЖДУ ОРИЕНТИРАМИ ПРОТИВ 206.77 / 3477.2")
    # честная решётка: ТОЛЬКО энергии в настоящих ориентирах.
    # Q и обрезки скана в отношения не входят: заряд с энергией
    # несоизмерим, а |E|max зависит от того, где остановили скан.
    vals = {}
    for name, v in L.items():
        for field in ("E1", "E0"):
            x = abs(v[field])
            if np.isfinite(x) and x > 1e-9:
                vals[f"|{field}|@{name.split(':')[0]}"] = x

    names = list(vals)
    print(f"\n  величины: " + ", ".join(f"{k}={v:.3g}" for k, v in vals.items()))
    best = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hi, lo = max(vals[a], vals[b]), min(vals[a], vals[b])
            if lo <= 0:
                continue
            r = hi / lo
            for target, lbl in ((M_MU_ME, "206.77"), (M_TAU_ME, "3477.2"),
                                (M_TAU_ME / M_MU_ME, "16.82")):
                miss = abs(math.log(r / target))
                best.append((miss, f"{a}/{b} = {r:9.3f} против {lbl}"))
    best.sort()
    print("\n  ближайшие попадания (по |ln(r/цель)|):")
    for miss, s in best[:6]:
        print(f"    промах e^{miss:5.2f} = {math.exp(miss):7.2f}x   {s}")
    return best


def verdict(best):
    head("4. ВЕРДИКТ")
    closest = math.exp(best[0][0]) if best else float("inf")
    print(f"""
  Лучшее из всех отношений между естественными ориентирами кривой
  промахивается в {closest:.1f} раза. Ни одна пара выделенных точек
  однопараметрического семейства (порог, пересечение уровней, складка,
  края окна) лептонных отношений не даёт.

  ЗНАЧИТ: правило отбора частоты НЕ живёт на кривой одиночного солитона.
  Кривая E(Q), mu(Q) гладкая, ориентиров на ней мало, и их отношения —
  числа порядка единиц-десятков. Чтобы выделить в окне ТРИ точки с
  отношениями 1 : 207 : 3477, нужен внешний масштаб — спектр самой среды.

  Оставшийся кандидат конкретизировался: баланс поглощения и излучения
  с шумовым термостатом среды (T-31) — уровень устойчив там, где приток
  энергии из нулевых колебаний равен утечке в излучение. Это уравнение
  на mu с ВНЕШНИМ спектром J(omega), и только оно способно внести в окно
  масштабы, которых нет у самой капли.
""")


def make_plot(c, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FG, BG, MUT = "#111111", "#ffffff", "#555555"
    COL = {0: "#0072B2", 1: "#E69F00"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), facecolor=BG)
    for n in (0, 1):
        ax1.plot(c[n]["Q"], c[n]["E"], "-o", color=COL[n], lw=2, ms=5,
                 mfc=BG, mew=1.4)
        ax1.annotate(f"$E_{n}$", (c[n]["Q"][-1], c[n]["E"][-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=COL[n], fontsize=11)
        ax2.plot(c[n]["Q"], c[n]["mu"], "-o", color=COL[n], lw=2, ms=5,
                 mfc=BG, mew=1.4)
        ax2.annotate(f"$\\mu_{n}$", (c[n]["Q"][-1], c[n]["mu"][-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=COL[n], fontsize=11)
    ax1.axhline(0, color=MUT, lw=0.6)
    for name, v in L.items():
        ax1.axvline(v["Q"], color=MUT, lw=0.8, ls="--")
        ax1.text(v["Q"], 0.98, name.split(":")[0],
                 transform=ax1.get_xaxis_transform(), ha="center",
                 va="top", color=FG, fontsize=9)
    ax1.set_xlabel("заряд $Q$", color=FG)
    ax1.set_ylabel("$E_n$", color=FG)
    ax1.set_title("Ориентиры на кривой семейства", color=FG)
    ax2.axhline(0, color=MUT, lw=0.8)
    ax2.axhline(MU_INF, color="#CC3311", lw=1.0, ls="--")
    ax2.text(0.02, MU_INF, "  $\\mu_\\infty = -3g^2/16h$", color="#CC3311",
             fontsize=9, va="bottom", transform=ax2.get_yaxis_transform())
    ax2.set_ylim(MU_INF * 1.08, 0.03)
    ax2.set_xlabel("заряд $Q$", color=FG)
    ax2.set_ylabel("$\\mu_n$", color=FG)
    ax2.set_title("Окно частот: гладкий подход к $\\mu_\\infty$, "
                  "выделенных точек нет", color=FG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUT)
        for s in ax.spines.values():
            s.set_color(MUT)
    fig.text(0.005, 0.005,
             "TorusBVP.py / FrequencySelection.py — кривая семейства "
             "секторов n = 0, 1 (T-15)", color=MUT, fontsize=8)
    out = os.path.join(HERE, "..", "..", "theory", "images",
                       "freqsel-window.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    print("\nнаписан", out)


if __name__ == "__main__":
    curves = load_curves()
    section_curves(curves)
    landmarks = section_landmarks(curves)
    ratios = section_ratios(curves, landmarks)
    verdict(ratios)
    if "--plot" in sys.argv:
        make_plot(curves, landmarks)
