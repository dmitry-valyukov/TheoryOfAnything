"""TorusBVPViz.py — статические иллюстрации к подглаве 2.8 из РЕШЕНИЙ TorusBVP.py.

В отличие от render_figures.py, который рисует принятый в модели анзац,
здесь рендерятся ПОСЧИТАННЫЕ минимайзеры секторов: плотность |W|^2 на
меридиональном сечении, диаграмма пересечения уровней E_n(Q) и провал
лестницы возбуждений против лептонных отношений.

Данные берутся из torusbvp_results.json и npz-файлов, которые пишет
TorusBVP.py; если каких-то записей нет, соответствующая панель пропускается
с предупреждением. Файлы кладутся в theory/images/.

Рисует:
    torusbvp-sectors.png   — сечения плотности минимайзеров n = 0..3 (Q = 740)
                             с разбиением момента на спин и орбиту;
    torusbvp-crossing.png  — пересечение уровней E_0/E_1 по Q (намотанный тор
                             становится основным состоянием) и лестница
                             возбуждений против требуемой лептонной.

Запуск:
    python TorusBVPViz.py [--out ../../theory/images]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DPI = 150
FG = "#111111"
BG = "#ffffff"
MUT = "#555555"

# фиксированный категориальный порядок секторов (палитра Окабэ - Ито,
# устойчива к цветовым аномалиям зрения; идентичность дублируется подписями)
SECTOR_COLOR = {0: "#0072B2", 1: "#E69F00", 2: "#009E73", 3: "#CC79A7"}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "torusbvp_results.json")

M_MU_ME, M_TAU_ME = 206.768, 3477.23


def load():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def rec(data, Q, n):
    """Лучшая (минимальная по E) запись сектора n при заряде Q."""
    best = None
    for r in data.values():
        if int(round(r["Q"])) == Q and r["n"] == n:
            if best is None or r["E"] < best["E"]:
                best = r
    return best


def npz_path(Q, n, seed):
    return os.path.join(HERE, f"torusbvp_results_Q{Q}_n{n}_{seed}.npz")


# ---------------------------------------------------------------------------
# Рис. 1: сечения плотности секторов
# ---------------------------------------------------------------------------

def fig_sectors(data, out):
    Q = 740
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.6), facecolor=BG)
    fig.suptitle(
        "Минимумы секторов намотки $n$: плотность $|W|^2$ на меридиональном "
        "сечении  (TorusBVP.py, $Q = 740$)",
        color=FG, fontsize=13, y=0.98)

    vmax = 1.8
    im = None
    for k, n in enumerate((0, 1, 2, 3)):
        ax = axes[k]
        ax.set_facecolor(BG)
        r = rec(data, Q, n)
        if r is None:
            ax.text(0.5, 0.5, "нет данных", transform=ax.transAxes,
                    ha="center", color=MUT)
            continue
        p = npz_path(Q, n, r["seed"])
        if not os.path.exists(p):
            ax.text(0.5, 0.5, "нет npz", transform=ax.transAxes,
                    ha="center", color=MUT)
            continue
        z = np.load(p)
        rho_d, R, Z = z["rho"], z["R"], z["Z"]
        # зеркалим сечение на -rho: полный меридиональный разрез тора
        dens = np.vstack([rho_d[::-1], rho_d])
        x = np.concatenate([-R[::-1, 0], R[:, 0]])
        im = ax.pcolormesh(x, Z[0], dens.T, cmap="magma",
                           vmin=0.0, vmax=vmax, shading="auto",
                           rasterized=True)
        # керновое кольцо
        ax.plot([r["R_core"], -r["R_core"]], [r["z_core"], r["z_core"]],
                ls="none", marker="o", ms=5, mfc="none", mec="#66ccff",
                mew=1.4)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-7, 7)
        ax.set_aspect("equal")
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color(SECTOR_COLOR[n])
            s.set_linewidth(1.6)
        ax.set_xlabel(r"$\rho$", color=MUT, fontsize=9)
        if k == 0:
            ax.set_ylabel(r"$z$", color=MUT, fontsize=9)

        Jz = n * r["Q"]
        if n > 0:
            spin = r["S_z"] / Jz
            split = f"спин {spin:.0%} · орбита {1 - spin:.0%}"
        else:
            split = "полоидальная циркуляция"
        ax.set_title(
            f"$n = {n}$   $E = {r['E']:.1f}$\n"
            f"$R/r_{{\\rm eff}} = {r['aspect']:.2f}$   {split}",
            color=SECTOR_COLOR[n], fontsize=10)

    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label(r"$|W|^2$  (плато $\rho_* = 1.5$)", color=MUT, fontsize=9)
        cb.ax.tick_params(colors=MUT, labelsize=8)

    fig.text(0.005, 0.01,
             "посчитанные минимайзеры секторно-точной релаксации (2.8.5), "
             "не анзац; кольца — положение кернового максимума",
             color=MUT, fontsize=8)
    path = os.path.join(out, "torusbvp-sectors.png")
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("написан", path)


# ---------------------------------------------------------------------------
# Рис. 2: пересечение уровней и лестница возбуждений
# ---------------------------------------------------------------------------

def fig_crossing(data, out):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.2, 5.2), facecolor=BG,
        gridspec_kw=dict(width_ratios=(1.25, 1.0), wspace=0.28))

    # --- слева: E0(Q), E1(Q), пересечение
    qs = sorted({int(round(r["Q"])) for r in data.values()})
    E = {n: [] for n in (0, 1)}
    q_ok = []
    for q in qs:
        r0, r1 = rec(data, q, 0), rec(data, q, 1)
        if r0 is None or r1 is None:
            continue
        q_ok.append(q)
        E[0].append(r0["E"])
        E[1].append(r1["E"])
    q_arr = np.array(q_ok, float)
    e0, e1 = np.array(E[0]), np.array(E[1])

    # точка пересечения — линейная интерполяция разности
    d = e1 - e0
    q_x = None
    for i in range(len(q_arr) - 1):
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            q_x = q_arr[i] + t * (q_arr[i + 1] - q_arr[i])
    axL.set_facecolor(BG)
    if q_x is not None:
        axL.axvspan(min(q_arr) - 20, q_x, color="#E69F00", alpha=0.08)
        axL.axvline(q_x, color=MUT, lw=1.0, ls="--")
        axL.text(q_x + 10, 0.88, f"$Q_\\times \\approx {q_x:.0f}$",
                 transform=axL.get_xaxis_transform(),
                 ha="left", va="top", color=FG, fontsize=10)
        axL.text(min(q_arr) - 10, 0.03,
                 "намотанный тор —\nосновное состояние",
                 transform=axL.get_xaxis_transform(),
                 ha="left", va="bottom", color="#8a6100", fontsize=9)
    for n, e in ((0, e0), (1, e1)):
        axL.plot(q_arr, e, "-o", color=SECTOR_COLOR[n], lw=2.0, ms=6,
                 mfc=BG, mew=1.6)
        axL.annotate(f"$E_{n}$" + ("  (без намотки)" if n == 0
                                   else "  (намотка $n=1$)"),
                     (q_arr[-1], e[-1]), xytext=(-6, 12 if n else -16),
                     textcoords="offset points", ha="right",
                     color=SECTOR_COLOR[n], fontsize=10)
    axL.axhline(0.0, color=MUT, lw=0.6)
    axL.set_xlabel("заряд $Q$", color=FG)
    axL.set_ylabel("энергия сектора $E_n$", color=FG)
    axL.set_title("Пересечение уровней: при $Q < Q_\\times$ узел ниже капли",
                  color=FG, fontsize=12)
    axL.tick_params(colors=MUT)
    for s in axL.spines.values():
        s.set_color(MUT)

    # --- справа: лестница возбуждений против лептонной (лог-шкала)
    axR.set_facecolor(BG)
    r0 = rec(data, 740, 0)
    d1 = rec(data, 740, 1)["E"] - r0["E"]
    got = [rec(data, 740, n)["E"] - r0["E"] for n in (1, 2, 3)]
    got = np.array(got) / d1
    need = np.array([1.0, M_MU_ME, M_TAU_ME])
    ns = np.array([1, 2, 3])

    axR.plot(ns, need, "-s", color="#666666", lw=1.8, ms=7, mfc=BG, mew=1.6)
    axR.plot(ns, got, "-o", color="#0072B2", lw=2.0, ms=7, mfc=BG, mew=1.8)
    axR.set_yscale("log")
    axR.set_xticks([1, 2, 3])
    axR.set_xlim(0.8, 3.45)
    axR.set_xlabel("номер сектора $n$", color=FG)
    axR.set_ylabel(r"$\Delta E_n / \Delta E_1$", color=FG)
    axR.set_title("Лестница секторов — не лептонная", color=FG, fontsize=12)
    axR.annotate("требуется\n(массы лептонов)", (3, need[2]),
                 xytext=(-12, -26), textcoords="offset points",
                 ha="right", color="#666666", fontsize=10)
    axR.annotate("посчитано\n($\\Delta E_n \\sim n^{1.3}$)", (3, got[2]),
                 xytext=(0, 12), textcoords="offset points",
                 ha="right", color="#0072B2", fontsize=10)
    # разрыв
    axR.annotate("", xy=(3, got[2]), xytext=(3, need[2]),
                 arrowprops=dict(arrowstyle="<->", color="#CC3311", lw=1.4))
    axR.text(3.06, float(np.sqrt(got[2] * need[2])),
             f"$\\times{need[2] / got[2]:.0f}$", color="#CC3311", fontsize=11,
             va="center")
    axR.tick_params(colors=MUT, which="both")
    for s in axR.spines.values():
        s.set_color(MUT)

    fig.text(0.005, 0.005,
             "TorusBVP.py, секторно-точная релаксация (2.8.5); слева также "
             "видно, что подгонка dE2/dE1 у пересечения не спасает: "
             "dE3/dE2 = 1.34 против 16.82",
             color=MUT, fontsize=8)
    path = os.path.join(out, "torusbvp-crossing.png")
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("написан", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..",
                                                  "theory", "images"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    data = load()
    fig_sectors(data, out)
    fig_crossing(data, out)


if __name__ == "__main__":
    main()
