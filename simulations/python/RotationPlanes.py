"""
Счёт задействованных плоскостей вращения: из слова в вычислимую величину.

Подглава 5.2 говорит, что модуль дробного заряда есть |q| = n_П / D, где n_П —
число задействованных модой ортогональных плоскостей вращения, D = 3. До сих
пор n_П было словом. Здесь оно определяется как **ранг** — величина дискретная
по построению и потому пригодная для точно квантованного заряда.

Определение.
    Параметр порядка среды — локальный репер Картана (1.3). Его вращение за
    внутренний период описывается аксиальным вектором угловой скорости
    omega(t) (в D = 3 мгновенное вращение всегда одноплоскостное: у любой
    антисимметричной матрицы 3x3 ранг 0 или 2, то есть ось одна). Разные
    «плоскости» появляются оттого, что ось за период СМЕЩАЕТСЯ.

    Поэтому считать надо не мгновенную плоскость, а размерность пространства,
    которое ось заметает за период:

        M = <omega omega^T>,   n_П = rank M

    - ось стоит на месте          -> rank 1  -> одна плоскость (d)
    - ось ходит по дуге           -> rank 2  -> две плоскости  (u)
    - ось заметает телесный угол  -> rank 3  -> три плоскости  (лептон)

    Ранг — целое число, устойчивое к малым возмущениям. Именно это и нужно:
    заряд адрона квантован точнее 1e-21, а метрическая величина (отношение
    длин) такой точности не даёт.

Знак.
    - при rank 1 знак задаётся направлением суммарного вращения, sign(<omega>·e1);
    - при rank >= 2 — ориентацией заметания оси, знаком площади, заметённой
      проекцией omega на плоскость двух старших собственных векторов.

ЧЕГО ЗДЕСЬ НЕТ. Функция omega(t) для кварковой моды из уравнений поля не
получена: это и есть содержание пункта T-36. Ниже процедура определена и
проверена на модельных входах; подстановка настоящего omega(t) остаётся долгом.

Запуск:  python RotationPlanes.py
"""

from __future__ import annotations

import math

import numpy as np

TOL = 0.05          # порог значимости собственного значения (доля от следа)


def correlation_matrix(omega: np.ndarray) -> np.ndarray:
    """M = <omega omega^T> по выборке omega формы (N, 3)."""
    omega = np.asarray(omega, dtype=float)
    return omega.T @ omega / len(omega)


def plane_spectrum(omega: np.ndarray) -> dict:
    """Собственные значения M, эффективный и целый ранг."""
    M = correlation_matrix(omega)
    w = np.linalg.eigvalsh(M)[::-1]
    w = np.clip(w, 0.0, None)
    tr = float(w.sum())
    frac = w / tr if tr > 0 else w
    rank = int(np.sum(frac > TOL))
    # участие: гладкий аналог ранга, (sum w)^2 / sum w^2
    part = float(tr ** 2 / np.sum(w ** 2)) if tr > 0 else 0.0
    return {"eigenvalues": w, "fractions": frac, "rank": rank,
            "participation": part}


def sweep_vector(omega: np.ndarray) -> np.ndarray:
    """
    Векторный инвариант обхода: <omega> + <omega x d(omega)>.

    Первое слагаемое ловит суммарное вращение при неподвижной оси (rank 1),
    второе — ориентацию заметания, когда ось ходит (rank >= 2). Складывать их
    можно: при rank 1 второе обращается в нуль, при rank 2 первое.
    """
    omega = np.asarray(omega, dtype=float)
    d = np.roll(omega, -1, axis=0) - omega
    return np.mean(omega, axis=0) + np.mean(np.cross(omega, d), axis=0) * len(omega)


def sweep_sign(omega: np.ndarray, reference: np.ndarray) -> dict:
    """
    Знак обхода относительно ВНЕШНЕЙ опорной оси.

    Абсолютного знака у хиральности нет: правое и левое различимы лишь
    относительно чего-то. В модели опорой служит ориентация самого солитона.
    Это же и есть причина, по которой заряд кварка определён относительно
    заряда лептона, а не сам по себе.
    """
    v = sweep_vector(omega)
    ref = np.asarray(reference, dtype=float)
    ref = ref / np.linalg.norm(ref)
    proj = float(v @ ref)
    return {"invariant": v, "projection": proj,
            "sign": int(np.sign(proj)) if abs(proj) > 1e-12 else 0}


def charge_from_rotation(omega: np.ndarray, reference, D: int = 3) -> dict:
    """|q| = n_П / D, знак — из ориентации обхода относительно опоры (5.2)."""
    spec = plane_spectrum(omega)
    sgn = sweep_sign(omega, reference)
    return {**spec, **sgn, "charge": sgn["sign"] * spec["rank"] / D}


# ---------------------------------------------------------------------------
# Мост к полю: omega из поля смещений
# ---------------------------------------------------------------------------


def omega_from_displacement(u_field, x0, times, h: float = 1e-4) -> np.ndarray:
    """
    Угловая скорость локального репера из поля смещений среды.

    В модели уже есть отождествление (3.1): B = rot u = 2 omega, то есть
    локальный поворот ячейки равен половине ротора смещения. Значит угловая
    скорость репера есть

        omega(t) = (1/2) rot (du/dt)

    и никакой новой сущности вводить не нужно: omega вычисляется из того же
    поля u(x, t), которым модель уже пользуется в главе 3.

    Отсюда рецепт: любой анзац или решение для u(x, t) немедленно даёт
    omega(t) в точке фокуса, а `plane_spectrum` — число задействованных
    плоскостей. Задача «получить omega» сводится к задаче «получить u».
    """
    x0 = np.asarray(x0, dtype=float)
    out = []
    for t in np.asarray(times, dtype=float):
        def vel(x):
            return (np.asarray(u_field(x, t + h)) - np.asarray(u_field(x, t - h))) / (2 * h)
        curl = np.zeros(3)
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            ej, ek = np.zeros(3), np.zeros(3)
            ej[j], ek[k] = h, h
            dvk_dj = (vel(x0 + ej)[k] - vel(x0 - ej)[k]) / (2 * h)
            dvj_dk = (vel(x0 + ek)[j] - vel(x0 - ek)[j]) / (2 * h)
            curl[i] = dvk_dj - dvj_dk
        out.append(0.5 * curl)
    return np.array(out)


# --- пробные поля смещений ---

def field_rigid_rotation(axis=(0.0, 0.0, 1.0), rate: float = 1.0):
    """u = (Omega x x) t — жёсткий поворот вокруг неподвижной оси."""
    ax = np.asarray(axis, dtype=float) * rate

    def u(x, t):
        return np.cross(ax, np.asarray(x, dtype=float)) * t
    return u


def field_linear_wave(amp: float = 1.0, k: float = 1.0, om: float = 1.0):
    """Линейно поляризованная сдвиговая волна вдоль z, смещение по x."""
    def u(x, t):
        x = np.asarray(x, dtype=float)
        return np.array([amp * math.cos(k * x[2] - om * t), 0.0, 0.0])
    return u


def field_circular_wave(hand: int = +1, amp: float = 1.0, k: float = 1.0,
                        om: float = 1.0):
    """Циркулярно поляризованная сдвиговая волна вдоль z."""
    def u(x, t):
        x = np.asarray(x, dtype=float)
        ph = k * x[2] - om * t
        return np.array([amp * math.cos(ph), hand * amp * math.sin(ph), 0.0])
    return u


def field_three_circular(amp: float = 1.0):
    """
    Наложение трёх циркулярных волн по трём осям с несоизмеримыми частотами.

    Двух волн мало: при равных частотах траектория omega(t) остаётся плоской
    (эллипс), и ранг выходит 2, а не 3. Третье измерение появляется только
    при несоизмеримости — это, между прочим, содержательное наблюдение:
    полный трёхплоскостной обход требует не просто трёх волн, а трёх
    НЕЗАВИСИМЫХ частот.
    """
    base = field_circular_wave(+1, amp, 1.0, 1.0)

    def u(x, t):
        x = np.asarray(x, dtype=float)
        p2, p3 = x[0] - 2.0 * t, x[1] - 3.0 * t
        return (np.asarray(base(x, t))
                + np.array([0.0, amp * math.cos(p2), amp * math.sin(p2)])
                + np.array([amp * math.sin(p3), 0.0, amp * math.cos(p3)]))
    return u


FIELD_CASES = (
    ("жёсткий поворот", field_rigid_rotation(), (0, 0, 1), 1),
    ("линейная поляризация", field_linear_wave(), (0, 1, 0), 1),
    ("циркулярная, правая", field_circular_wave(+1), (0, 0, 1), 2),
    ("циркулярная, левая", field_circular_wave(-1), (0, 0, 1), 2),
    ("три циркулярных", field_three_circular(), (0, 0, 1), 3),
)


def field_report() -> None:
    print("\n" + "=" * 78)
    print("МОСТ К ПОЛЮ: omega = (1/2) rot(du/dt), отождествление 3.1")
    print("=" * 78)
    print(f"\n  {'поле смещений':<24} {'доли собств. знач.':<26} {'rank':>5} "
          f"{'знак':>5} {'|q|':>6}")
    ok = True
    times = np.linspace(0.0, 2 * math.pi, 600, endpoint=False)
    for name, fld, ref, want in FIELD_CASES:
        om = omega_from_displacement(fld, (0.3, 0.2, 0.1), times)
        res = charge_from_rotation(om, ref)
        fr = " ".join(f"{x:.3f}" for x in res["fractions"])
        print(f"  {name:<24} {fr:<26} {res['rank']:>5} {res['sign']:>5d} "
              f"{res['rank'] / 3:>6.3f}")
        ok = ok and res["rank"] == want
    print(f"\n  все пробные поля распознаны верно: {ok}")
    print()
    print("  Читается прямо, и это проверяемое утверждение о моде:")
    print("    ЛИНЕЙНАЯ поляризация  -> одна плоскость -> |q| = 1/3  (d)")
    print("    ЦИРКУЛЯРНАЯ           -> две плоскости  -> |q| = 2/3  (u)")
    print("    три циркулярных       -> три плоскости  -> |q| = 1    (лептон)")
    print("  Двух циркулярных волн МАЛО: при равных частотах omega(t) остаётся")
    print("  плоской и ранг выходит 2. Полный обход требует трёх независимых")
    print("  частот — отдельное содержательное следствие процедуры.")
    print("  Знак заряда есть направление циркулярности — то же, что различает")
    print("  электрон и позитрон. Гипотеза 5.2 переводится в вопрос о")
    print("  ПОЛЯРИЗАЦИИ моды, а он решается тем же радиальным BVP (2.5).")


# ---------------------------------------------------------------------------
# Модельные входы для проверки процедуры
# ---------------------------------------------------------------------------


def sample_axis_fixed(sense: int = +1, num: int = 2000) -> np.ndarray:
    """Ось стоит: чистое вращение в одной плоскости."""
    t = np.linspace(0.0, 2 * math.pi, num, endpoint=False)
    return np.stack([np.zeros_like(t), np.zeros_like(t),
                     sense * np.ones_like(t)], axis=1)


def sample_axis_arc(sense: int = +1, num: int = 2000) -> np.ndarray:
    """Ось ходит по дуге: задействованы две плоскости."""
    t = np.linspace(0.0, 2 * math.pi, num, endpoint=False)
    return np.stack([np.cos(t), np.zeros_like(t), sense * np.sin(t)], axis=1)


def sample_axis_solid(num: int = 4000) -> np.ndarray:
    """Ось заметает телесный угол: задействованы все три плоскости."""
    t = np.linspace(0.0, 2 * math.pi, num, endpoint=False)
    return np.stack([np.cos(t), np.sin(t), np.cos(2 * t) * 0.9], axis=1)


#: опорная ось для знака: та, вокруг которой определяется хиральность
CASES = (
    ("ось стоит, +", sample_axis_fixed(+1), (0, 0, 1), 1, +1),
    ("ось стоит, −", sample_axis_fixed(-1), (0, 0, 1), 1, -1),
    ("ось по дуге, +", sample_axis_arc(+1), (0, -1, 0), 2, +1),
    ("ось по дуге, −", sample_axis_arc(-1), (0, -1, 0), 2, -1),
    ("ось по углу", sample_axis_solid(), (0, 0, 1), 3, None),
)


def report() -> None:
    print("=" * 78)
    print("СЧЁТ ПЛОСКОСТЕЙ ВРАЩЕНИЯ КАК РАНГ")
    print("=" * 78)
    print("\n  M = <omega omega^T>,  n_П = rank M,  |q| = n_П / 3")
    print(f"\n  {'вход':<18} {'доли собств. знач.':<28} {'rank':>5} "
          f"{'участие':>8} {'знак':>5} {'q':>7}")
    ok = True
    for name, om, ref, want_rank, want_sign in CASES:
        res = charge_from_rotation(om, ref)
        fr = " ".join(f"{x:.3f}" for x in res["fractions"])
        q = res["charge"]
        print(f"  {name:<18} {fr:<28} {res['rank']:>5} "
              f"{res['participation']:>8.3f} {res['sign']:>5d} {q:>+7.3f}")
        if res["rank"] != want_rank:
            ok = False
        if want_sign is not None and res["sign"] != want_sign:
            ok = False
    print(f"\n  все модельные входы распознаны верно: {ok}")

    print("\n  Соответствие модели (5.2):")
    print("    d      — ось стоит      -> rank 1 -> |q| = 1/3")
    print("    u      — ось по дуге    -> rank 2 -> |q| = 2/3")
    print("    лептон — ось по углу    -> rank 3 -> |q| = 1  (и цвета нет:")
    print("             выделять нечего, задействованы все три плоскости)")

    print("\n  Устойчивость ранга к шуму (почему заряд точно квантован):")
    rng = np.random.default_rng(12345)
    base = sample_axis_arc(+1)
    print(f"    {'шум':>8} {'rank':>6} {'участие':>9}")
    for eps in (0.0, 0.01, 0.05, 0.10, 0.20):
        noisy = base + eps * rng.standard_normal(base.shape)
        r = plane_spectrum(noisy)
        print(f"    {eps:>8.2f} {r['rank']:>6d} {r['participation']:>9.3f}")
    print(f"    порог значимости собственного значения TOL = {TOL}")
    print("    ранг остаётся 2: целое число не сползает от возмущений,")
    print("    в отличие от отношения длин (5.11.6)")

    field_report()

    print("\n  ЧЕГО НЕТ: решения для u(x, t) кварковой моды. Процедура и мост")
    print("  к полю определены и проверены; не хватает самого поля — T-36.")


if __name__ == "__main__":
    report()
