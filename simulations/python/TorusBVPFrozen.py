"""T-42, остаток: релаксация с ВМОРОЖЕННОЙ спиральностью.

Спуск по мнимому времени в TorusBVP.py — релаксация тейлоровского типа:
он сохраняет заряд Q, но СТЕКАЕТ спиральность H, поэтому найденные там
минимумы секторов — нижние границы. Среда же бездиссипативна, и
топология каждой линии вморожена (2.8.2): истинное состояние узла — минимум
энергии при ДВУХ связях, Q = const и H = const.

Здесь та же секторно-точная схема W = V(rho,z) e^{i n phi}, но после
каждого неявного шага делается проекция на многообразие {Q, H} = const:
Ньютон по двум множителям (a, b) в подстановке V -> a (V + b C V),
где C — оператор ротора (H = (1/2) Re <V, C V> — эрмитова квадратичная
форма). Первый порядок по b, две итерации на шаг; невязки Q и H
контролируются.

Что даёт:
  1. E_1(H) при фиксированном Q — ЦЕНА ТОПОЛОГИИ, монотонна ли;
  2. R/r истинного узла против минимума сектора;
  3. выживает ли пересечение уровней Q_x при вмороженной H;
  4. сравнение с нижними границами 2.8 (при H = H_min должно совпасть).

Запуск:  python TorusBVPFrozen.py scan [Q]     — E(H) для n=1 при заряде Q
         python TorusBVPFrozen.py report
Результаты — torusbvp_frozen.json.
"""

import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import TorusBVP as T

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "torusbvp_frozen.json")


def curl_operator(n):
    """Оператор ротора в секторе n на трёх компонентах (rho, phi, z)."""
    rho, z, R, Z, w = T.grids()
    Ir, Iz = sp.identity(T.NR), sp.identity(T.NZ)
    Dr, Dz = T.d_matrix(T.NR, T.HH), T.d_matrix(T.NZ, T.HH)
    inv_r = sp.kron(sp.diags(1.0 / rho), Iz)
    DR = sp.kron(Dr, Iz)
    DZ = sp.kron(Ir, Dz)
    Zb = sp.csr_matrix((T.NR * T.NZ, T.NR * T.NZ))
    # curl_r = (i n / rho) V_z - dz V_phi
    # curl_p = dz V_rho - dr V_z
    # curl_z = dr V_phi + V_phi / rho - (i n / rho) V_rho
    C = sp.bmat([[Zb, -DZ, (1j * n) * inv_r],
                 [DZ, Zb, -DR],
                 [(-1j * n) * inv_r, DR + inv_r, Zb]], format="csc")
    return C, w


def qh(Vf, C, wf):
    q = float((np.abs(Vf) ** 2 * wf).sum())
    h = 0.5 * float((np.real(np.conj(Vf) * (C @ Vf)) * wf).sum())
    return q, h


def project_qh(Vf, C, wf, Q0, H0, iters=3):
    """Проекция на {Q = Q0, H = H0}: V -> a (V + b C V)."""
    for _ in range(iters):
        CV = C @ Vf
        q0, h0 = qh(Vf, C, wf)
        # производные по b при a = 1 (первый порядок):
        # dQ/db = 2 Re<V, CV>_w = 4 h0 ; dH/db = Re<CV, C V>_w = <CV,CV>_w
        dQdb = 4.0 * h0
        dHdb = float((np.abs(CV) ** 2 * wf).sum())
        # решаем: a^2 (q0 + dQdb b) = Q0 ; a^2 (h0 + dHdb b) = H0
        # => (h0 + dHdb b)/(q0 + dQdb b) = H0/Q0
        r = H0 / Q0
        denom = dHdb - r * dQdb
        if abs(denom) < 1e-14:
            break
        b = (r * q0 - h0) / denom
        b = max(-0.5, min(0.5, b))
        Vf = Vf + b * CV
        q1, _ = qh(Vf, C, wf)
        Vf = Vf * math.sqrt(Q0 / q1)
    return Vf


def relax_frozen(n, Q0, H0, seed_kind="tube", max_it=8000, verbose=True):
    L, Div, w, R, Z = T.build_operators(n)
    C, _ = curl_operator(n)
    N3 = 3 * T.NR * T.NZ
    Ident = sp.identity(N3, format="csc", dtype=np.complex128)
    solver = spla.splu((Ident + T.DT * L).tocsc())
    V = T.seed_tube(n, R, Z) if seed_kind == "tube" else T.seed_ball(n, R, Z)
    wf = np.concatenate([w.ravel()] * 3)
    Vf = V.reshape(N3)
    q = float((np.abs(Vf) ** 2 * wf).sum())
    Vf = Vf * math.sqrt(Q0 / q)
    Vf = project_qh(Vf, C, wf, Q0, H0, iters=20)
    E_prev, hits = np.inf, 0
    for it in range(max_it):
        V = Vf.reshape(3, T.NR, T.NZ)
        rho_d = (np.abs(V) ** 2).sum(axis=0)
        NL = (T.G_CUB * rho_d - T.H_QUI * rho_d ** 2) * V
        Vn = solver.solve((V + T.DT * NL).reshape(N3))
        Vf = project_qh(Vn, C, wf, Q0, H0)
        if it % 100 == 99:
            obs = T.observables(Vf.reshape(3, T.NR, T.NZ), n, Div, w, R, Z)
            if abs(obs["E"] - E_prev) < 1e-9 * max(abs(obs["E"]), 1e-12):
                hits += 1
                if hits >= 3:
                    break
            else:
                hits = 0
            E_prev = obs["E"]
    V = Vf.reshape(3, T.NR, T.NZ)
    obs = T.observables(V, n, Div, w, R, Z)
    q_f, h_f = qh(Vf, C, wf)
    rec = dict(n=n, Q0=Q0, H0=H0, seed=seed_kind, iters=it + 1,
               Q_res=q_f - Q0, H_res=h_f - H0, **obs)
    data = {}
    if os.path.exists(RESULTS):
        data = json.load(open(RESULTS))
    data[f"Q{int(Q0)}_n{n}_H{H0:+.1f}"] = rec
    json.dump(data, open(RESULTS, "w"), indent=1)
    if verbose:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in rec.items()}, ensure_ascii=False))
    return rec


def report():
    data = json.load(open(RESULTS))
    print(f"{'ключ':>16} {'iters':>6} {'E':>11} {'H':>8} {'H_res':>9} "
          f"{'S_z/Q':>7} {'R':>6} {'r_eff':>6} {'R/r':>6} {'div_rel':>9}")
    for k in sorted(data, key=lambda s: (int(s.split('_')[0][1:]),
                                          float(s.split('_H')[1]))):
        r = data[k]
        print(f"{k:>16} {r['iters']:>6} {r['E']:>11.3f} {r['H']:>8.3f} "
              f"{r['H_res']:>9.1e} {r['S_z']/r['Q']:>7.3f} {r['R_core']:>6.2f} "
              f"{r['r_eff']:>6.2f} {r['aspect']:>6.2f} {r['div_rel']:>9.2e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        Q0 = float(sys.argv[2]) if len(sys.argv) > 2 else 740.0
        Hs = [float(x) for x in sys.argv[3:]] or [0.0, 5.0, 10.0, 20.0, 40.0]
        for H0 in Hs:
            relax_frozen(1, Q0, H0)
    else:
        report()
