"""
T-42, первый приступ: релаксация кубично-квинтического функционала для
ВЕКТОРНОГО СОЛЕНОИДАЛЬНОГО комплексного поля при фиксированном заряде Q,
в секторах намотки n.

Постановка ([2.7], [2.8], [T-42]): минимизируется

    E[W] = int [ |grad W|^2 - (g/2)|W|^4 + (h/3)|W|^6 ] dV,
    div W = 0,   Q = int |W|^2 dV = const,

где W — комплексное трёхкомпонентное поле смещений (физическое поле
u = Re[W e^{-i omega t}]), mu = dE/dQ — множитель Лагранжа.

СЕКТОР НАМОТКИ ЗАДАН ТОЧНО, ПО ПОСТРОЕНИЮ:

    W = [V_rho(rho,z) e_rho + V_phi e_phi + V_z e_z] e^{i n phi},

то есть поле трёх комплексных функций на ДВУМЕРНОЙ сетке (rho, z) —
ровно та размерность, что была обещана в 2.6.5. Число n нумерует сектор
полного момента J_z = n: орбитальная намотка и поляризация (спин)
складываются, и внутри сектора спуск свободен перераспределять момент
между ними («наложение мод простого вращения и тороидально-полоидального»).

Урок, оплаченный счётом: первая, трёхмерная декартова версия этого
расчёта сектор НЕ удержала — машинный шум на уровне 1e-16 засевает
чужие секторы, и за ~10^3 шагов мнимого времени спуск скатывает любое
n в основной шар n = 0 (усиление ~ e^{|mu| T}). Секторную задачу надо
решать в секторно-точном представлении, иное — самообман.

Метод: полуявная схема по мнимому времени. Линейная часть (векторный
лапласиан с центробежными членами (n^2+1)/rho^2 и связью +-2in/rho^2
между V_rho и V_phi) и штраф beta |div W|^2 сидят В НЕЯВНОЙ матрице
(sparse LU, факторизация один раз), нелинейность (g rho - h rho^2) W —
явно. Соленоидальность держится штрафом, невязка контролируется.

Измеряется: E_n, mu_n, спиральность H_n, разбиение момента на спин и
орбиту, положение кернового кольца R, эффективный радиус r, R/r;
отношения E_n против 206.77 и 3477.2.

Запуск:  python TorusBVP.py all            — секторы n = 0..3, обе затравки
         python TorusBVP.py relax n [seed] — один сектор (seed: tube | ball)
         python TorusBVP.py report         — сводка
Результаты — torusbvp_results.json.
"""

import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------------ параметры
G_CUB, H_QUI = 2.0, 1.0        # плато плотности 3g/4h = 1.5, стенка ~0.6
Q_FIX = 740.0
NR, NZ = 88, 120               # сетка (rho, z)
HH = 0.15                      # шаг
BETA_DIV = 200.0               # штраф несоленоидальности
DT = 0.05
MAX_IT = 30000
TOL = 1e-11
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "torusbvp_results.json")


def grids():
    rho = (np.arange(NR) + 0.5) * HH          # ось не содержит точку rho=0
    z = (np.arange(NZ) - NZ / 2 + 0.5) * HH
    R, Z = np.meshgrid(rho, z, indexing="ij")
    w = 2.0 * math.pi * R * HH * HH           # мера объёма
    return rho, z, R, Z, w


def d_matrix(n_pts, h):
    """Центральная производная, Дирихле снаружи."""
    e = np.ones(n_pts)
    D = sp.diags([-e[1:], e[:-1]], [-1, 1]) / (2 * h)
    return sp.csr_matrix(D)


def build_operators(n):
    """Блочная матрица L: квадратичная часть градиента энергии
    плюс beta * D^H D (штраф дивергенции)."""
    rho, z, R, Z, w = grids()
    Ir, Iz = sp.identity(NR), sp.identity(NZ)
    Dr, Dz = d_matrix(NR, HH), d_matrix(NZ, HH)

    # скалярный радиальный лапласиан: (1/rho) d/drho (rho d/drho) + d2/dz2
    e = np.ones(NR)
    main = -2.0 * e / HH ** 2
    up = (1.0 + 0.5 * HH / rho[:-1]) / HH ** 2
    dn = (1.0 - 0.5 * HH / rho[1:]) / HH ** 2
    Lr = sp.diags([dn, main, up], [-1, 0, 1])
    ez = np.ones(NZ)
    Lz = sp.diags([ez[1:], -2 * ez, ez[:-1]], [-1, 0, 1]) / HH ** 2
    Lsc = sp.kron(Lr, Iz) + sp.kron(Ir, Lz)

    R2 = sp.kron(sp.diags(1.0 / rho ** 2), Iz)

    Zb = sp.csr_matrix((NR * NZ, NR * NZ))
    L_rr = -Lsc + (n ** 2 + 1) * R2
    L_pp = -Lsc + (n ** 2 + 1) * R2
    L_zz = -Lsc + (n ** 2) * R2
    L_rp = (2j * n) * R2
    L_pr = (-2j * n) * R2

    L = sp.bmat([[L_rr, L_rp, Zb],
                 [L_pr, L_pp, Zb],
                 [Zb, Zb, L_zz]], format="csc")

    # дивергенция: D = [ d/drho + 1/rho , in/rho , d/dz ]
    inv_r = sp.kron(sp.diags(1.0 / rho), Iz)
    Div = sp.hstack([sp.kron(Dr, Iz) + inv_r,
                     (1j * n) * inv_r,
                     sp.kron(Ir, Dz)], format="csc")
    L = (L + BETA_DIV * (Div.getH() @ Div)).tocsc()
    return L, Div, w, R, Z


def seed_tube(n, R, Z, R0=4.0, rt=2.5, edge=0.6):
    """Гирационная трубка: круговая поляризация в полоидальной плоскости."""
    S = np.sqrt((R - R0) ** 2 + Z ** 2)
    F = 0.5 * (1.0 - np.tanh((S - rt) / edge))
    V = np.zeros((3, NR, NZ), dtype=np.complex128)
    V[0] = F / math.sqrt(2.0)          # e_rho
    V[2] = 1j * F / math.sqrt(2.0)     # e_z
    return V


def seed_ball(n, R, Z, a=5.0, edge=0.6):
    """Спин-шар: (e_rho + i e_phi) e^{in phi} = (x^ + i y^) e^{i(n-1) phi}."""
    r_sph = np.sqrt(R ** 2 + Z ** 2)
    F = 0.5 * (1.0 - np.tanh((r_sph - a) / edge))
    V = np.zeros((3, NR, NZ), dtype=np.complex128)
    V[0] = F / math.sqrt(2.0)
    V[1] = 1j * F / math.sqrt(2.0)
    return V


def gradient_energy(V, n, R, w):
    Dr, Dz = d_matrix(NR, HH), d_matrix(NZ, HH)
    E2 = 0.0
    for c in range(3):
        gr_r = Dr @ V[c]
        gr_z = V[c] @ Dz.T
        E2 += float(((np.abs(gr_r) ** 2 + np.abs(gr_z) ** 2) * w).sum())
    a = 1j * n * V[0] - V[1]
    b = 1j * n * V[1] + V[0]
    E2 += float((((np.abs(a) ** 2 + np.abs(b) ** 2
                   + (n ** 2) * np.abs(V[2]) ** 2) / R ** 2) * w).sum())
    return E2


def observables(V, n, Div, w, R, Z):
    rho_d = (np.abs(V) ** 2).sum(axis=0)
    Q = float((rho_d * w).sum())
    E2 = gradient_energy(V, n, R, w)
    E = E2 - (G_CUB / 2.0) * float((rho_d ** 2 * w).sum()) \
        + (H_QUI / 3.0) * float((rho_d ** 3 * w).sum())
    Dr, Dz = d_matrix(NR, HH), d_matrix(NZ, HH)
    curl_r = (1j * n / R) * V[2] - (V[1] @ Dz.T)
    curl_p = (V[0] @ Dz.T) - (Dr @ V[2])
    curl_z = (Dr @ V[1]) + V[1] / R - (1j * n / R) * V[0]
    H = 0.5 * float((np.real(np.conj(V[0]) * curl_r + np.conj(V[1]) * curl_p
                             + np.conj(V[2]) * curl_z) * w).sum())
    S_z = float((2.0 * np.imag(np.conj(V[0]) * V[1]) * w).sum())
    dv = (Div @ V.reshape(3 * NR * NZ)).reshape(NR, NZ)
    div2 = float((np.abs(dv) ** 2 * w).sum())
    imax = np.unravel_index(np.argmax(rho_d), rho_d.shape)
    R_core, z_core = float(R[imax]), float(Z[imax])
    s2 = (R - R_core) ** 2 + (Z - z_core) ** 2
    r_eff = math.sqrt(float((rho_d * s2 * w).sum() / (rho_d * w).sum()))
    return dict(Q=Q, E=float(E), E2=float(E2), H=H, S_z=S_z, L_z=n * Q - S_z,
                div2=div2, div_rel=math.sqrt(div2 / max(E2, 1e-30)),
                R_core=R_core, z_core=z_core, r_eff=r_eff,
                aspect=R_core / r_eff if r_eff > 0 else float("nan"),
                rho_peak=float(rho_d.max()))


def relax(n, seed_kind="tube", verbose=True):
    L, Div, w, R, Z = build_operators(n)
    Ident = sp.identity(3 * NR * NZ, format="csc", dtype=np.complex128)
    solver = spla.splu((Ident + DT * L).tocsc())
    V = seed_tube(n, R, Z) if seed_kind == "tube" else seed_ball(n, R, Z)
    q = float(((np.abs(V) ** 2).sum(axis=0) * w).sum())
    V *= math.sqrt(Q_FIX / q)
    wf = np.concatenate([w.ravel()] * 3)
    E_prev, hits = np.inf, 0
    for it in range(MAX_IT):
        rho_d = (np.abs(V) ** 2).sum(axis=0)
        NL = (G_CUB * rho_d - H_QUI * rho_d ** 2) * V
        rhs = (V + DT * NL).reshape(3 * NR * NZ)
        Vn = solver.solve(rhs)
        q = float((np.abs(Vn) ** 2 * wf).sum())
        V = (Vn * math.sqrt(Q_FIX / q)).reshape(3, NR, NZ)
        if it % 100 == 99:
            obs = observables(V, n, Div, w, R, Z)
            if abs(obs["E"] - E_prev) < TOL * max(abs(obs["E"]), 1e-12):
                hits += 1
                if hits >= 3:
                    break
            else:
                hits = 0
            E_prev = obs["E"]
    obs = observables(V, n, Div, w, R, Z)
    rho_d = (np.abs(V) ** 2).sum(axis=0)
    act = (L @ V.reshape(3 * NR * NZ)) \
        - ((G_CUB * rho_d - H_QUI * rho_d ** 2) * V).reshape(3 * NR * NZ)
    mu = float((np.real(np.conj(V.reshape(-1)) * act) * wf).sum()) / Q_FIX
    rec = dict(n=n, seed=seed_kind, iters=it + 1, mu=mu, **obs)
    data = {}
    if os.path.exists(RESULTS):
        data = json.load(open(RESULTS))
    data[f"Q{int(Q_FIX)}_n{n}_{seed_kind}"] = rec
    json.dump(data, open(RESULTS, "w"), indent=1)
    np.savez_compressed(RESULTS.replace(".json", f"_Q{int(Q_FIX)}_n{n}_{seed_kind}.npz"),
                        rho=rho_d, R=R, Z=Z)
    if verbose:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in rec.items()}, ensure_ascii=False))
    return rec


def report():
    data = json.load(open(RESULTS))
    print(f"{'сектор':>10} {'iters':>6} {'E':>11} {'mu':>8} {'H':>9} "
          f"{'S_z/Q':>7} {'L_z/Q':>7} {'R':>6} {'r_eff':>6} {'R/r':>6} "
          f"{'rho_pk':>7} {'div_rel':>9}")
    for k in sorted(data):
        r = data[k]
        print(f"{k:>10} {r['iters']:>6} {r['E']:>11.3f} {r['mu']:>8.4f} "
              f"{r['H']:>9.3f} {r['S_z']/r['Q']:>7.3f} {r['L_z']/r['Q']:>7.3f} "
              f"{r['R_core']:>6.2f} {r['r_eff']:>6.2f} {r['aspect']:>6.2f} "
              f"{r['rho_peak']:>7.3f} {r['div_rel']:>9.2e}")
    qs = {}
    for r in data.values():
        qs.setdefault(int(round(r["Q"])), {})
        n, cur = r["n"], qs[int(round(r["Q"]))]
        if n not in cur or r["E"] < cur[n]["E"]:
            cur[n] = r
    for q in sorted(qs):
        best = qs[q]
        if 0 not in best or 1 not in best:
            continue
        print(f"\nQ = {q}: возбуждения dE_n = E_n - E_0 и отношения")
        d1 = best[1]["E"] - best[0]["E"]
        for n in sorted(best):
            if n == 0:
                continue
            d = best[n]["E"] - best[0]["E"]
            extra = f"   dE_n/dE_1 = {d/d1:9.3f}" if abs(d1) > 1e-12 else ""
            print(f"  n={n}: E = {best[n]['E']:10.3f}  dE = {d:9.3f}{extra}")
        if 2 in best and 3 in best:
            d2 = best[2]["E"] - best[0]["E"]
            d3 = best[3]["E"] - best[0]["E"]
            print(f"  контроль поколений: dE3/dE2 = {d3/d2:.3f}  (нужно 3477.2/206.77 = 16.82)")
    print("\nтребуется для поколений: dE2/dE1 = 206.77, dE3/dE1 = 3477.2")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "relax":
        nn = int(sys.argv[2])
        kind = sys.argv[3] if len(sys.argv) > 3 else "tube"
        relax(nn, kind)
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        for nn in (0, 1, 2, 3):
            for kind in ("tube", "ball"):
                relax(nn, kind)
    else:
        report()
