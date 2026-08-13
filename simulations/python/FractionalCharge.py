"""FractionalCharge.py — дробный заряд как дефицит спинорного обхода.

Расчёты к подглаве theory/05-11-deficit-obhoda.md.

Гипотеза (объединяет 5.2, 5.3 и 5.9):

  * Полный спинорный обход есть 4*pi (замыкание солитона, 2.1).
  * Условие замыкания цветового контура (5.9) даёт набег delta_phi = 4*pi/3
    на одно ребро — то есть ОДНО РЕБРО ЦВЕТОВОГО КОНТУРА РОВНО ТРЕТЬ
    полного спинорного обхода. Отсюда знаменатель 3.
  * Мода, вращающаяся в двух ортогональных плоскостях из трёх (u), набирает
    за цветовой оборот вдвое больше фазы, чем мода, запертая в одной (d);
    это то же самое, что omega_d = omega_u / 2 из 5.3.
  * Заряд есть доля полного обхода, пройденная модой, со знаком намотки.

Отсюда без подгонки: Q_u = +2/3, Q_d = -1/3, и всё остальное — следствия.

Ничего, кроме нормировки «заряд протона равен +1», не подставляется.

Запуск:
    python FractionalCharge.py
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement

# --- шаг цветового контура в единицах полного спинорного обхода 4*pi ---
LEG = Fraction(1, 3)          # delta_phi = 4pi/3, то есть 1/3 от 4pi (5.9)

# --- сколько плоскостей вращения занимает мода (5.2) ---
PLANES = {"u": 2, "d": 1}
# --- знак намотки ---
SIGN = {"u": +1, "d": -1}


def charge(flavour: str, anti: bool = False) -> Fraction:
    """Заряд как доля спинорного обхода со знаком намотки."""
    q = SIGN[flavour] * PLANES[flavour] * LEG
    return -q if anti else q


def content_charge(content) -> Fraction:
    return sum(charge(f, a) for f, a in content)


def closes(content) -> bool:
    """Замыкается ли суммарный обход: полный заряд обязан быть целым."""
    return content_charge(content).denominator == 1


def baryon_number(content) -> Fraction:
    n_q = sum(1 for _, a in content if not a)
    n_a = sum(1 for _, a in content if a)
    return Fraction(n_q - n_a, 3)


# ---------------------------------------------------------------------------

def q(s: str):
    """Разбор строки вида 'uud' или 'u d~' (тильда = антикварк)."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in " ":
            i += 1
            continue
        anti = i + 1 < len(s) and s[i + 1] == "~"
        out.append((c, anti))
        i += 2 if anti else 1
    return out


HADRONS = [
    ("p", "uud", 1), ("n", "udd", 0),
    ("D++", "uuu", 2), ("D+", "uud", 1), ("D0", "udd", 0), ("D-", "ddd", -1),
    ("pi+", "ud~", 1), ("pi-", "du~", -1), ("pi0", "uu~", 0),
    ("p~", "u~u~d~", -1),
]


def report() -> None:
    print("=" * 78)
    print("FractionalCharge: дробный заряд как дефицит спинорного обхода")
    print("=" * 78)
    print()

    print("-" * 78)
    print("1. ОТКУДА ЗНАМЕНАТЕЛЬ 3")
    print("-" * 78)
    print()
    print("  Условие замыкания цветового контура (5.9): 3 * delta_phi = 4 pi,")
    print("  откуда delta_phi = 4 pi / 3. В единицах полного спинорного обхода")
    print(f"  4 pi одно ребро цветового контура составляет ровно {LEG}.")
    print()
    print("  Знаменатель 3 приходит не из подгонки под наблюдаемые заряды,")
    print("  а из числа рёбер контура — оно же число цветов, оно же")
    print("  размерность пространства (5.1).")
    print()

    print("-" * 78)
    print("2. ЗАРЯДЫ КВАРКОВ")
    print("-" * 78)
    print()
    print(f"  {'мода':<6s} {'плоскостей':>11s} {'знак':>6s} {'доля обхода':>13s} {'заряд':>8s}")
    for f in ("u", "d"):
        print(f"  {f:<6s} {PLANES[f]:>11d} {SIGN[f]:>+6d} "
              f"{str(PLANES[f]*LEG):>13s} {str(charge(f)):>8s}")
    print()
    print("  Отношение |Q_d| / |Q_u| = 1/2 есть то же самое, что omega_d =")
    print("  omega_u / 2 из 5.3: мода в одной плоскости набирает фазу вдвое")
    print("  медленнее моды в двух. Это ОДНО допущение, записанное дважды.")
    print()

    print("-" * 78)
    print("3. ЗАРЯДЫ АДРОНОВ: ни одного подгоночного параметра")
    print("-" * 78)
    print()
    print(f"  {'адрон':<8s} {'состав':<10s} {'предсказание':>13s} {'опыт':>7s} {'B':>6s} {'':>4s}")
    bad = 0
    for name, s, exp in HADRONS:
        c = content_charge(q(s))
        ok = (c == exp)
        bad += 0 if ok else 1
        print(f"  {name:<8s} {s:<10s} {str(c):>13s} {exp:>7d} "
              f"{str(baryon_number(q(s))):>6s} {'OK' if ok else 'МИМО':>4s}")
    print()
    print(f"  Расхождений: {bad}")
    print()

    print("-" * 78)
    print("4. УСЛОВИЕ ЗАМЫКАНИЯ: какие комбинации вообще возможны")
    print("-" * 78)
    print()
    print("  Свободное состояние обязано замкнуть обход целиком, то есть иметь")
    print("  ЦЕЛЫЙ суммарный заряд. Перебор всех комбинаций из u, d и их")
    print("  античастиц до пяти составляющих:")
    print()
    kinds = [("u", False), ("d", False), ("u", True), ("d", True)]
    print(f"  {'состав':<10s} {'кварков-антикварков':>21s} {'замыкается':>12s}")
    seen = {}
    for n in range(1, 6):
        allowed = 0
        total = 0
        for combo in combinations_with_replacement(kinds, n):
            total += 1
            if closes(combo):
                allowed += 1
        seen[n] = (allowed, total)
    for n, (a, t) in seen.items():
        print(f"  {n} составляющих: {a} из {t} комбинаций замыкаются")
    print()
    print("  Разбор по структуре (N_q - N_qbar):")
    for label, combo in (("q       (одиночный кварк)", [("u", False)]),
                         ("qq      (дикварк)", [("u", False), ("d", False)]),
                         ("qqq     (барион)", [("u", False)] * 3),
                         ("q qbar  (мезон)", [("u", False), ("d", True)]),
                         ("qqqq    (тетра из кварков)", [("u", False)] * 4),
                         ("qq qbqb (тетракварк)", [("u", False)] * 2 + [("d", True)] * 2),
                         ("qqqq qb (пентакварк)", [("u", False)] * 4 + [("d", True)])):
        nq = sum(1 for _, a in combo if not a)
        na = sum(1 for _, a in combo if a)
        print(f"    {label:<28s} N_q - N_qb = {nq-na:+d}  "
              f"{'ЗАМЫКАЕТСЯ' if closes(combo) else 'запрещено'}")
    print()
    print("  Условие сводится к N_q - N_qbar = 0 (mod 3), то есть к целому")
    print("  барионному числу. Оно РАЗРЕШАЕТ барион, мезон, тетракварк и")
    print("  пентакварк — все наблюдаемые типы — и ЗАПРЕЩАЕТ одиночный кварк,")
    print("  дикварк и четвёркукварков, которых не наблюдают.")
    print()
    print("  Оговорка: это правило ОТБОРА, а не механизм. Почему незамкнутая")
    print("  конфигурация не может существовать динамически — вопрос главы 6")
    print("  (эффект Пуассона в несжимаемом канале), и здесь он не решается.")
    print()

    print("=" * 78)
    print("ЧТО ЗДЕСЬ ВЫВЕДЕНО, А ЧТО ПРЕДПОЛОЖЕНО")
    print("=" * 78)
    print("  предположено (гипотеза 5.2/5.3, не выводится):")
    print("     u занимает 2 плоскости вращения, d — одну")
    print("     знак намотки d обратен знаку u")
    print("  выведено (следствия, ни одного подгоночного параметра):")
    print("     знаменатель 3 — из условия замыкания цветового контура (5.9)")
    print("     Q_u = +2/3, Q_d = -1/3")
    print("     заряды всех u/d-адронов, включая нейтральность нейтрона")
    print("     правило отбора: барион, мезон, тетра- и пентакварк разрешены;")
    print("        одиночный кварк и дикварк запрещены")
    print()


if __name__ == "__main__":
    report()
