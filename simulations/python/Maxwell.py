"""Maxwell.py — численный аудит вывода уравнений Максвелла из упругой среды.

Расчёты к подглаве theory/03-5-audit-elektrodinamiki.md.

Проверяется НЕ «похоже ли», а буквально: строится явное поле смещений
упругой среды, все производные берутся конечными разностями 4-го порядка,
и считаются невязки всех четырёх уравнений Максвелла.

Что проверяется:
  1. отождествление A ~ u (смещение) даёт все 4 уравнения с верными знаками;
  2. отождествление 3.1 (E ~ +dPsi/dt) даёт закон Фарадея с ОБРАТНЫМ знаком;
  3. добавление продольной компоненты ломает и Гаусса, и Ампера
     -> несжимаемость есть не отдельное допущение, а требование Максвелла;
  4. сопоставление плотностей энергии: что именно фиксировано, а что нет;
  5. вейлевский множитель sqrt(rho c^2 / G) тождественно равен 1;
  6. размерности масштабного мостика K и инварианта E.B.

Запуск:
    python Maxwell.py
"""

from __future__ import annotations

import numpy as np

C = 1.0            # скорость поперечной волны в условных единицах
H = 1e-3           # шаг конечных разностей


# ---------------------------------------------------------------------------
# 1. Явное поле смещений
# ---------------------------------------------------------------------------

def make_waves(seed: int = 20260813, n: int = 4, longitudinal: bool = False):
    """Набор плоских волн. По умолчанию строго поперечных (e . k = 0).

    longitudinal=True добавляет одну продольную волну (e || k) —
    именно то, что запрещено пределом несжимаемости.
    """
    rng = np.random.default_rng(seed)
    waves = []
    for _ in range(n):
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        k *= rng.uniform(0.7, 1.6)
        # поляризация, ортогональная k
        tmp = rng.normal(size=3)
        e = np.cross(k, tmp)
        e /= np.linalg.norm(e)
        waves.append((k, e, rng.uniform(0.5, 1.5), rng.uniform(0, 2 * np.pi)))
    if longitudinal:
        k = np.array([0.0, 0.0, 1.1])
        e = k / np.linalg.norm(k)          # e || k — продольная мода
        waves.append((k, e, 1.0, 0.3))
    return waves


def displacement(waves):
    """Возвращает функцию u(t, r) -> вектор смещения."""
    def u(t, r):
        out = np.zeros(3)
        for k, e, a, ph in waves:
            omega = C * np.linalg.norm(k)
            out += a * e * np.cos(np.dot(k, r) - omega * t + ph)
        return out
    return u


# ---------------------------------------------------------------------------
# 2. Конечные разности 4-го порядка
# ---------------------------------------------------------------------------

_W4 = ((1, 8.0 / 12.0), (2, -1.0 / 12.0))


def d_dt(f, t, r, h=H):
    out = np.zeros(3)
    for n, w in _W4:
        out += w * (f(t + n * h, r) - f(t - n * h, r)) / h
    return out


def d_dx(f, t, r, axis, h=H):
    out = np.zeros(3)
    for n, w in _W4:
        rp = r.copy(); rp[axis] += n * h
        rm = r.copy(); rm[axis] -= n * h
        out += w * (f(t, rp) - f(t, rm)) / h
    return out


def jacobian(f, t, r):
    """J[i][j] = d f_j / d x_i."""
    return np.array([d_dx(f, t, r, i) for i in range(3)])


def divergence(f, t, r):
    J = jacobian(f, t, r)
    return J[0][0] + J[1][1] + J[2][2]


def curl(f, t, r):
    J = jacobian(f, t, r)   # J[i][j] = d_i f_j
    return np.array([J[1][2] - J[2][1],
                     J[2][0] - J[0][2],
                     J[0][1] - J[1][0]])


def curl_field(f):
    return lambda t, r: curl(f, t, r)


# ---------------------------------------------------------------------------
# 3. Проверка уравнений Максвелла
# ---------------------------------------------------------------------------

def sample_points(n=12, seed=7):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0, 5), rng.uniform(-3, 3, size=3)) for _ in range(n)]


def maxwell_residuals(u, sign_E=-1.0):
    """Невязки четырёх уравнений при E = sign_E * du/dt, B = rot u.

    Возвращает словарь: для каждого уравнения — относительная невязка,
    то есть max|левая - правая| / max(|левая|, |правая|).
    """
    E = lambda t, r: sign_E * d_dt(u, t, r)
    B = curl_field(u)

    res = {k: [0.0, 1e-300] for k in ("gauss_E", "gauss_B", "faraday", "ampere")}

    def acc(key, lhs, rhs, scale=None):
        res[key][0] = max(res[key][0], np.max(np.abs(np.atleast_1d(lhs - rhs))))
        cand = [np.max(np.abs(np.atleast_1d(lhs))), np.max(np.abs(np.atleast_1d(rhs)))]
        if scale is not None:
            cand.append(np.max(np.abs(np.atleast_1d(scale))))
        res[key][1] = max(res[key][1], *cand)

    for t, r in sample_points():
        # для уравнений вида div F = 0 нормируем на масштаб самих производных,
        # иначе деление на ноль превращает машинный нуль в невязку 1.0
        JE, JB = jacobian(E, t, r), jacobian(B, t, r)
        acc("gauss_E", divergence(E, t, r), 0.0, scale=JE)
        acc("gauss_B", divergence(B, t, r), 0.0, scale=JB)
        # rot E = -dB/dt
        acc("faraday", curl(E, t, r), -d_dt(B, t, r))
        # rot B = (1/c^2) dE/dt
        acc("ampere", curl(B, t, r), d_dt(E, t, r) / C ** 2)

    return {k: v[0] / v[1] for k, v in res.items()}


# ---------------------------------------------------------------------------
# 4. Плотности энергии
# ---------------------------------------------------------------------------

def energy_densities(u, rho=1.0, G=1.0, n=60000, seed=11):
    """Средние по объёму и времени плотности кинетической и сдвиговой энергии.

    Проверяется заодно тождество <G * u_ij u_ij> = <(G/2) |rot u|^2>
    для бездивергентного поля (верно в среднем, не поточечно).
    """
    rng = np.random.default_rng(seed)
    kin = shear_curl = shear_strain = 0.0
    for _ in range(n):
        t = rng.uniform(0, 400)
        r = rng.uniform(-200, 200, size=3)
        v = d_dt(u, t, r)
        J = jacobian(u, t, r)
        eps = 0.5 * (J + J.T)
        kin += 0.5 * rho * np.dot(v, v)
        cu = np.array([J[1][2] - J[2][1], J[2][0] - J[0][2], J[0][1] - J[1][0]])
        shear_curl += 0.5 * G * np.dot(cu, cu)
        shear_strain += G * np.sum(eps * eps)
    return kin / n, shear_curl / n, shear_strain / n


# ---------------------------------------------------------------------------
# 5. Размерности
# ---------------------------------------------------------------------------

# базис (кг, м, с, А)
DIM = {
    "kg": np.array([1, 0, 0, 0]), "m": np.array([0, 1, 0, 0]),
    "s": np.array([0, 0, 1, 0]), "A": np.array([0, 0, 0, 1]),
    "mu0": np.array([1, 1, -2, -2]),        # Гн/м
    "eps0": np.array([-1, -3, 4, 2]),       # Ф/м
    "e": np.array([0, 0, 1, 1]),            # Кл = А*с
    "Pa": np.array([1, -1, -2, 0]),         # Па = Дж/м^3
    "E": np.array([1, 1, -3, -1]),          # В/м
    "B": np.array([1, 0, -2, -1]),          # Тл
    "G": np.array([1, -1, -2, 0]),          # модуль сдвига, Па
    "rho": np.array([1, -3, 0, 0]),         # кг/м^3
    "A_pot": np.array([1, 1, -2, -1]),      # В*с/м = Тл*м
    "Ohm": np.array([1, 2, -3, -2]),
}


def fmt_dim(v):
    names = ("кг", "м", "с", "А")
    parts = [f"{n}^{int(p)}" for n, p in zip(names, v) if p != 0]
    return "·".join(parts) if parts else "безразмерно"


# ---------------------------------------------------------------------------
# 6. Отчёт
# ---------------------------------------------------------------------------

def report() -> None:
    print("=" * 78)
    print("Maxwell: численный аудит вывода уравнений Максвелла из упругой среды")
    print("=" * 78)
    print()

    waves_T = make_waves()
    u_T = displacement(waves_T)

    print("-" * 78)
    print("1. ОТОЖДЕСТВЛЕНИЕ A ~ u:  E = -du/dt,  B = rot u")
    print("-" * 78)
    print()
    print("  Поле: суперпозиция 4 поперечных плоских волн (e . k = 0),")
    print("  точное решение волнового уравнения, div u = 0 тождественно.")
    print()
    r = maxwell_residuals(u_T, sign_E=-1.0)
    names = {"gauss_E": "div E = 0", "gauss_B": "div B = 0",
             "faraday": "rot E = -dB/dt", "ampere": "rot B = (1/c^2) dE/dt"}
    for k, label in names.items():
        ok = "OK" if r[k] < 1e-6 else "НЕ ВЫПОЛНЕНО"
        print(f"  {label:<28s} относительная невязка = {r[k]:.3e}   {ok}")
    print()
    print("  Все четыре уравнения выполняются с точностью схемы.")
    print("  Знаки верны БЕЗ подгонки: они следуют из E = -du/dt.")
    print()

    print("-" * 78)
    print("2. ЗНАК: что даёт запись 3.1 (E ~ +d/dt, без минуса)")
    print("-" * 78)
    print()
    r2 = maxwell_residuals(u_T, sign_E=+1.0)
    for k, label in names.items():
        ok = "OK" if r2[k] < 1e-6 else "НЕ ВЫПОЛНЕНО"
        print(f"  {label:<28s} относительная невязка = {r2[k]:.3e}   {ok}")
    print()
    print("  При E = +du/dt закон Фарадея и закон Ампера получают обратный знак")
    print("  (невязка равна 2, то есть левая часть равна минус правой).")
    print("  Знак «~» в подглаве 3.1 скрывает обязательный минус.")
    print()

    print("-" * 78)
    print("3. НЕСЖИМАЕМОСТЬ: что будет, если оставить продольную ветвь")
    print("-" * 78)
    print()
    u_L = displacement(make_waves(longitudinal=True))
    r3 = maxwell_residuals(u_L, sign_E=-1.0)
    for k, label in names.items():
        ok = "OK" if r3[k] < 1e-6 else "НЕ ВЫПОЛНЕНО"
        print(f"  {label:<28s} относительная невязка = {r3[k]:.3e}   {ok}")
    print()
    print("  Добавлена одна продольная волна (e || k). Ломаются ДВА уравнения:")
    print("  закон Гаусса (div E != 0) и закон Ампера — последний потому, что")
    print("  rot rot u = grad(div u) - lap u, и при div u != 0 появляется")
    print("  лишний градиентный член. Причём продольная волна взята здесь даже")
    print("  с той же скоростью c; при реальном c_L != c_T разрушение сильнее.")
    print()
    print("  Вывод: div E = 0 в вакууме ЕСТЬ несжимаемость среды div u = 0.")
    print("  Это не калибровка и не отдельный постулат, а условие K -> бесконечность.")
    print()

    print("-" * 78)
    print("4. ПЛОТНОСТИ ЭНЕРГИИ И РАЗДЕЛЕНИЕ eps0 / mu0")
    print("-" * 78)
    print()
    kin, sh_curl, sh_strain = energy_densities(u_T)
    print(f"  <кинетическая>  (rho/2)|du/dt|^2      = {kin:.6f}")
    print(f"  <сдвиговая>     (G/2)|rot u|^2        = {sh_curl:.6f}")
    print(f"  <сдвиговая>     G u_ij u_ij           = {sh_strain:.6f}")
    print(f"  отношение двух записей сдвиговой      = {sh_strain/sh_curl:.6f}")
    print(f"  равнораспределение (кин./сдвиг.)      = {kin/sh_curl:.6f}")
    print()
    print("  Обе записи сдвиговой энергии совпадают в среднем (для div u = 0),")
    print("  и выполняется равнораспределение, как и должно быть для бегущей волны.")
    print()
    print("  Теперь о разделении eps0 и mu0. Пусть A = kappa * u. Тогда")
    print("    E = -kappa * v,   B = kappa * rot u,")
    print("    (eps0/2)E^2 = (rho/2)v^2      =>  eps0 = rho / kappa^2")
    print("    (1/2mu0)B^2 = (G/2)|rot u|^2  =>  mu0  = kappa^2 / G")
    print("  Произведение eps0*mu0 = rho/G — ИНВАРИАНТНО, kappa сокращается.")
    print("  Разделение по отдельности от kappa ЗАВИСИТ:")
    print()
    print(f"  {'kappa':<22s} {'eps0':<14s} {'mu0':<14s} что получается")
    print(f"  {'1 (E = скорость)':<22s} {'rho':<14s} {'1/G':<14s} "
          f"противоположно записи 3.2")
    print(f"  {'sqrt(rho*G) = rho*c':<22s} {'1/G':<14s} {'rho':<14s} "
          f"как в подглаве 3.2")
    print()
    print("  Вывод: модель фиксирует ТОЛЬКО произведение. Утверждение 3.2")
    print("  «mu0 ~ rho, eps0 ~ 1/G» есть выбор нормировки kappa = rho*c,")
    print("  а не следствие. При этом онтология 3.1 («E — скорость сдвига»)")
    print("  требует kappa = 1 и даёт ОБРАТНОЕ разделение. Два места главы")
    print("  согласуются только если одно из них ослабит формулировку.")
    print()

    print("-" * 78)
    print("5. ВЕЙЛЕВСКАЯ КАЛИБРОВКА ПОДГЛАВЫ 3.3")
    print("-" * 78)
    print()
    for rho, G in ((1.0, 1.0), (3.7, 11.2), (1e97, 1e113)):
        c2 = G / rho
        factor = np.sqrt(rho * c2 / G)
        print(f"  rho = {rho:<10.4g} G = {G:<10.4g} "
              f"sqrt(rho*c^2/G) = {factor:.15g}")
    print()
    print("  Множитель тождественно равен 1: подстановка c^2 = G/rho даёт")
    print("  rho*(G/rho)/G = 1 при ЛЮБЫХ значениях. Калибровка 3.3 не делает")
    print("  ничего и, в частности, не может превратить безразмерное Psi")
    print("  в размерное A.")
    print()
    kappa_dim = 0.5 * (DIM["mu0"] + DIM["G"])
    print(f"  Размерно верный мостик: A = sqrt(mu0 * G_shear) * u")
    print(f"    [sqrt(mu0*G)] = {fmt_dim(kappa_dim)}  (это Тесла)")
    print(f"    [sqrt(mu0*G) * u] = {fmt_dim(kappa_dim + DIM['m'])}"
          f"  против [A] = {fmt_dim(DIM['A_pot'])}")
    same = np.array_equal(kappa_dim + DIM["m"], DIM["A_pot"])
    print(f"    совпадает: {same}")
    print()

    print("-" * 78)
    print("6. РАЗМЕРНОСТИ МАСШТАБНОГО МОСТИКА K И ИНВАРИАНТА E·B")
    print("-" * 78)
    print()
    K_dim = -(DIM["mu0"] + 2 * DIM["e"])
    print(f"  K = 1/(mu0 e^2) * (dx0/l_Pl)^2")
    print(f"    [mu0 * e^2] = {fmt_dim(DIM['mu0'] + 2*DIM['e'])}")
    print(f"    [K]         = {fmt_dim(K_dim)}")
    print(f"    требуется   = безразмерно (так как sigma и T^EM оба в Па)")
    print(f"    ВЫВОД: {'совпадает' if not K_dim.any() else 'РАЗМЕРНОСТЬ НЕ СХОДИТСЯ'}")
    print()
    EB = DIM["E"] + DIM["B"]
    print(f"  [E·B]     = {fmt_dim(EB)}")
    print(f"  [Дж/м^3]  = {fmt_dim(DIM['Pa'])}")
    print(f"  разность  = {fmt_dim(EB - DIM['Pa'])}")
    print(f"  это в точности Ом: {np.array_equal(EB - DIM['Pa'], DIM['Ohm'])}")
    print()
    print("  То есть [E·B] = (Дж/м^3) * Ом. Утверждение 3.3 верно в гауссовой")
    print("  системе и неверно в СИ; система единиц должна быть оговорена.")
    print()

    print("=" * 78)
    print("РЕЗЮМЕ")
    print("=" * 78)
    print("  + свободные уравнения Максвелла выводятся точно, при A ~ u")
    print("  + продольная ветвь ломает Гаусса И Ампера — несжимаемость обязательна")
    print("  + div E = 0 получает физический смысл: несжимаемость среды")
    print("  - в 3.1 пропущен обязательный минус в E = -du/dt")
    print("  - в 3.1 смешаны смещение u и поле вращений Omega")
    print("  - разделение eps0 / mu0 в 3.2 зависит от нормировки, а не выведено")
    print("  - вейлевский множитель 3.3 тождественно равен 1")
    print("  - множитель K в 3.3 размерно несостоятелен")
    print("  - [E·B] = Дж/м^3 верно только в гауссовой системе")
    print()


if __name__ == "__main__":
    report()
