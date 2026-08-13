"""Локальная модель корреляций пары спинов на торе против квантовой предсказки.

ЧТО ДОБАВЛЕНО (CODE_REVIEW.md §2.3)
------------------------------------
Старый скрипт складывал корреляции в список с именем `cshs_ab` (опечатка
в "CHSH"), но сам параметр CHSH нигде не вычислял — то есть главный вопрос
"нарушает ли модель неравенство Белла" оставался без ответа. Теперь S
считается и печатается, а ключ --scan ищет максимум |S| по всем четвёркам
углов, а не только по каноническим.

Физическая часть модели (вид функции torus_spin) НЕ менялась.

РЕЗУЛЬТАТ
---------
Модель воспроизводит ФОРМУ cos(x) с систематическим отклонением до ~9%,
но S = 2.000000 ровно, причём с НУЛЕВОЙ дисперсией: комбинация CHSH равна
+2 на КАЖДОЙ отдельной реализации скрытой переменной, а не в среднем.
Так и должно быть. Для любых детерминированных исходов +-1 комбинация
A*B - A*B' + A'*B + A'*B' = A(B - B') + A'(B + B') алгебраически может
принимать только значения +-2: одна из скобок всегда обнуляется. Среднее
по любому распределению скрытой переменной поэтому лежит в [-2, 2].

Это не случайность выбора углов: перебор по сетке 96^4 даёт максимум
|S| = 2.000000. Квантовый предел Цирельсона 2*sqrt(2) ~ 2.828 недостижим
для любой локальной модели такого типа.

Запуск:
    python bell.py
    python bell.py --scan            # поиск максимума |S| по углам (небыстро)
    python bell.py --samples 200000 --no-plot
"""

import argparse
import math

import numpy as np

SEED = 20260811          # фиксированный seed: результат воспроизводим

#: Канонические углы CHSH (a, a' | b, b')
CANONICAL = (0.0, math.pi / 2, math.pi / 4, 3 * math.pi / 4)

TSIRELSON = 2.0 * math.sqrt(2.0)
CLASSICAL_BOUND = 2.0


def torus_spin(phase, x):
    """Детерминированный локальный исход (+-1) при скрытой переменной phase.

    phase — общая для обеих сторон скрытая переменная (ZPF),
    x — локальная настройка прибора. Ни одна сторона не видит настройку другой.
    """
    phase2 = np.cos(phase / 2) * (np.pi / 4)
    return np.sign(np.sin(phase - x / 2) ** 2
                   + 2 * np.cos(phase2 / 2 - x / 2) ** 2
                   - 1.5)


def outcomes(phase, angles):
    """Матрица исходов (len(angles), len(phase)) для набора настроек."""
    x = np.asarray(angles, dtype=float)[:, None]
    p = phase[None, :]
    p2 = np.cos(p / 2) * (np.pi / 4)
    return np.sign(np.sin(p - x / 2) ** 2 + 2 * np.cos(p2 / 2 - x / 2) ** 2 - 1.5)


def chsh(phase, angles=CANONICAL, return_samples: bool = False):
    """Возвращает (S, погрешность среднего) для четвёрки (a, a', b, b').

    Для детерминированных исходов +-1 величина
    A*B - A*B' + A'*B + A'*B' = A(B - B') + A'(B + B')
    на каждой отдельной реализации может быть только +-2 (одна из скобок
    обнуляется). Отсюда и следует классический предел |S| <= 2.
    """
    a, ap, b, bp = angles
    A, Ap, B, Bp = outcomes(phase, (a, ap, b, bp))
    per_sample = A * B - A * Bp + Ap * B + Ap * Bp
    s = float(per_sample.mean())
    sem = float(per_sample.std(ddof=1) / math.sqrt(len(phase)))
    if return_samples:
        return s, sem, per_sample
    return s, sem


def scan_max_chsh(phase, n_angles: int = 96) -> tuple:
    """Максимум |S| по сетке из n_angles^4 четвёрок углов.

    Считается через корреляционную матрицу C[i,j] = <A_i B_j>: она строится
    один раз (одно матричное умножение), после чего
    S(i,j,k,l) = C[i,k] - C[i,l] + C[j,k] + C[j,l] — уже просто индексация.
    Полный перебор "в лоб" был бы на 4-5 порядков медленнее.
    """
    grid = np.linspace(0.0, 4 * np.pi, n_angles, endpoint=False)
    o = outcomes(phase, grid).astype(np.float32)
    corr = (o @ o.T) / len(phase)

    best, best_idx = -1.0, None
    for i in range(n_angles):
        # по осям (j, k, l)
        tensor = (corr[i][None, :, None] - corr[i][None, None, :]
                  + corr[:, :, None] + corr[:, None, :])
        mag = np.abs(tensor)
        idx = np.unravel_index(int(np.argmax(mag)), mag.shape)
        if float(mag[idx]) > best:
            best = float(mag[idx])
            best_idx = (i,) + tuple(int(v) for v in idx)
    return best, tuple(grid[list(best_idx)])


def correlation_curve(phase, x_vals):
    """<A(0) B(x)> как функция относительного угла x."""
    a = torus_spin(phase, 0.0)
    b = outcomes(phase, x_vals)
    return (a[None, :] * b).mean(axis=1)


# --------------------------------------------------------------------------

def report(phase, do_scan: bool):
    x_vals = np.linspace(0, 4 * np.pi, 200)
    model = correlation_curve(phase, x_vals)
    theory = np.cos(x_vals)

    print("=" * 74)
    print("ЛОКАЛЬНАЯ МОДЕЛЬ НА ТОРЕ: КОРРЕЛЯЦИИ И НЕРАВЕНСТВО БЕЛЛА")
    print("=" * 74)
    print(f"Выборок скрытой переменной : {len(phase):,}".replace(",", " "))
    print(f"Максимальное |модель - cos(x)| : {np.abs(model - theory).max():.4f}")
    print()

    s, sem, per_sample = chsh(phase, CANONICAL, return_samples=True)
    print("Параметр CHSH на канонических углах (0, pi/2 | pi/4, 3pi/4):")
    print(f"  S = {s:.6f} +- {sem:.6f}")
    uniq = np.unique(per_sample)
    print(f"  значения по отдельным реализациям: {uniq}")
    if len(uniq) == 1:
        print("  (комбинация равна пределу на КАЖДОЙ реализации, не только в среднем)")
    print(f"  классический предел (Белл)  : {CLASSICAL_BOUND:.6f}")
    print(f"  квантовый предел (Цирельсон): {TSIRELSON:.6f}")
    print()

    if s > CLASSICAL_BOUND + 3 * sem:
        print("  ВЫВОД: неравенство Белла НАРУШЕНО (более 3 сигма).")
    else:
        print("  ВЫВОД: неравенство Белла НЕ нарушено — S упирается в классический")
        print("  предел 2, но не превышает его. Квантовая корреляция не получена.")
    print()

    if do_scan:
        print("Поиск максимума |S| по сетке углов (не только каноническим)...")
        best, angles = scan_max_chsh(phase)
        print(f"  max |S| = {best:.6f}")
        print(f"  углы    = {np.round(angles, 4)}")
        print()
        if best <= CLASSICAL_BOUND + 1e-6:
            print("  Максимум достигается ровно на классическом пределе. Это не")
            print("  особенность выбора углов: детерминированная локальная модель с")
            print("  общей скрытой переменной не может дать |S| > 2 ни при каких")
            print("  настройках — теорема Белла.")
    else:
        print("(ключ --scan ищет максимум |S| по всем четвёркам углов)")

    return x_vals, model, theory, s, sem


def make_plot(x_vals, model, theory, s, sem, theme="light", out_path=None):
    import matplotlib.pyplot as plt
    from lib.theme import THEMES, style_axes, legend

    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=t["surface"])

    ax.plot(x_vals, theory, color=t["s1"], lw=2.0, zorder=3,
            label=r"Квантовая теория:  $\cos(x)$")
    ax.plot(x_vals, model, color=t["s2"], lw=1.6, zorder=4,
            label="Локальная модель на торе (общий ZPF)")

    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.15, 1.80)
    ax.set_xticks(np.arange(0, 4.1 * np.pi, np.pi))
    ax.set_xticklabels(["0", "π", "2π", "3π", "4π"])
    style_axes(ax, t,
               xlabel="Относительный угол x, рад",
               ylabel=r"$\langle AB \rangle$",
               title="Корреляции пары спинов: локальная модель против квантовой")

    verdict = "нарушено" if s > CLASSICAL_BOUND + 3 * sem else "НЕ нарушено"
    ax.annotate(f"CHSH:  S = {s:.4f} ± {sem:.4f}\n"
                f"классический предел 2,  Цирельсон {TSIRELSON:.3f}\n"
                f"неравенство Белла {verdict}",
                xy=(0.015, 0.975), xycoords="axes fraction",
                ha="left", va="top", fontsize=9.5,
                color=t["muted"], linespacing=1.5)

    legend(ax, t, loc="upper right")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=t["surface"])
        print(f"\nГрафик сохранён: {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Локальная модель корреляций против неравенства Белла")
    p.add_argument("--samples", type=int, default=200000,
                   help="число выборок скрытой переменной (по умолчанию 200000)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--scan", action="store_true",
                   help="искать максимум |S| по сетке углов 96^4")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--theme", choices=("light", "dark"), default="light")
    p.add_argument("--save", metavar="PATH")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    phase = rng.uniform(0, 4 * np.pi, args.samples)

    x_vals, model, theory, s, sem = report(phase, args.scan)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        make_plot(x_vals, model, theory, s, sem,
                  theme=args.theme, out_path=args.save)
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
