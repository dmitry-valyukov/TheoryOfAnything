"""T-4, звено 2: перенос дрожания по спаянной заклиненной сети — коэффициент диффузии колебательной
энергии D(omega) по Аллену — Фельдману (гармоническая сеть, классический предел).

Постановка. Механизм B (7.4, T-4) требует стационарной диффузии сохраняющейся «величины дрожания» от
узла: delta<u^2> = S/(4 pi D r). Здесь на спаянной сети (BondedNet, 1.11/11.2) считается сам коэффициент
переноса: модовые диффузивности D_i = pi/(3 w_i^2) sum_j |S_ij|^2 delta(w_i - w_j),
S_ij^a = 1/2 (w_i + w_j) sum_{ab} R^a_ab e_i^T Phi_ab e_j (Allen, Feldman 1993; массы 1, k = 1/sigma^2),
дельта — гауссиан ширины eta = n_eta * (средний межуровневый интервал). Выдаётся D(omega) в единицах
d^2 w_D и в единицах a c (c = sqrt(mu/rho) по гессиановому mu), длина пробега l = 3D/c,
частота Иоффе — Регеля (l = d), средневзвешенное D по спектру.

Запуск: python JitterTransport.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2] [--neta 4]
        python JitterTransport.py --analyse jitter_transport_N512.npz
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from StateDrift import BondedNet, assemble_full, moduli_full  # noqa: E402


def allen_feldman(p, n_eta=4.0):
    """Модовые диффузивности спаянной сети. Возвращает w (частоты, без нулевых мод), D_i, ell_i, а также
    (mu, rho, c, w_D)."""
    H, Xis, aff, V, pr = assemble_full(p)
    N = p.N
    w2, E = np.linalg.eigh(H)
    keep = w2 > 1e-8 * w2.max()
    w = np.sqrt(w2[keep]); E = E[:, keep]
    # блоки K_ab и R_ab по активным парам
    i, j, dx, r, ov = p.contacts()
    sig = p.rad[i] + p.rad[j]
    dVdr = -ov / sig; d2V = 1.0 / sig ** 2
    n = dx / r[:, None]
    K = (d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :] + (dVdr / r)[:, None, None] * np.eye(3)[None]
    # Phi_R^alpha: блок (a,b) = R^alpha_ab * Phi_ab, где Phi_ab = -K_ab (a != b), R_ab = x_b - x_a = dx
    S = []
    for al in range(3):
        M = np.zeros((3 * N, 3 * N))
        Kr = K * dx[:, al][:, None, None]      # R^al_ij K_ij для пары (i -> j): R_ij = dx
        for a_ in range(3):
            for b_ in range(3):
                # Phi_ij = -K_ij ; R_ij = +dx ; Phi_ji = -K_ij ; R_ji = -dx
                np.add.at(M, (3 * i + a_, 3 * j + b_), -Kr[:, a_, b_])
                np.add.at(M, (3 * j + a_, 3 * i + b_), +Kr[:, a_, b_])
        S.append(E.T @ M @ E)
    S = np.array(S)                              # (3, n, n)
    # |S_ij|^2 ∝ |M_ij|^2 (w_i + w_j)^2 / (4 w_i w_j)  -> ~ |M_ij|^2 при w_j ~ w_i
    S2 = (S ** 2).sum(axis=0) * (w[:, None] + w[None, :]) ** 2 / (4.0 * w[:, None] * w[None, :])
    # гауссова дельта с шириной по локальной плотности уровней
    ws = np.sort(w)
    spacing = np.gradient(ws)                    # локальный интервал
    eta_i = n_eta * np.interp(w, ws, spacing)
    dw = w[:, None] - w[None, :]
    G = np.exp(-0.5 * (dw / eta_i[:, None]) ** 2) / (math.sqrt(2 * math.pi) * eta_i[:, None])
    np.fill_diagonal(G, 0.0)
    # Аллен-Фельдман: D_i = pi V^2/(3 hbar^2 w_i^2) sum_j |S_ij|^2 delta(w_i-w_j),
    # S_ij = hbar (w_i+w_j)/(4 V sqrt(w_i w_j)) M_ij  ->  D_i = pi/(12 w_i^2) sum_j |M_ij|^2 (w_i+w_j)^2/(4 w_i w_j) delta
    D = math.pi / (12 * w ** 2) * (S2 * G).sum(axis=1)
    mus, B, pr2, _ = moduli_full(p)
    mu = mus.mean(); rho = N / V; c = math.sqrt(mu / rho)
    ell = 3 * D / c
    return w, D, ell, dict(mu=mu, rho=rho, c=c, wD=w.max(), B=B, p=pr)


def run_sample(N, seed, dphis, n_eta, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed} ---", flush=True)
    p0, phi_j = find_phi_j(N, seed, False, verbose=verbose)
    out = []
    for dphi in dphis:
        p = PinnedSpheres(N, 0.65, seed)
        p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
        p.s, p.L = p0.s.copy(), p0.L; p.G3[:] = 0.0
        p.set_phi(phi_j + dphi); p.stabilize()
        z0, _, _ = p.coordination()
        i0, j0, *_ = p.contacts()
        b = BondedNet(N, 0.65, seed); b.rad = p.rad.copy(); b.vol_particles = p.vol_particles
        b.rc = p.rc; b.L = p.L; b.s = p.s.copy(); b.G3 = p.G3.copy(); b.bond(i0, j0)
        b.minimize(rel=1e-10)
        w, D, ell, info = allen_feldman(b, n_eta)
        d = 2 * b.rad.mean()
        rec = dict(N=N, seed=seed, dphi=dphi, dz0=z0 - Z_C, w=w, D=D, ell=ell, d=d, **info)
        # сводные числа
        wD = info["wD"]; c = info["c"]
        m_diff = (w > 0.3 * wD) & (w < 0.9 * wD)
        Dm = np.median(D[m_diff]) if m_diff.any() else np.nan
        # Иоффе-Регель: где ell падает до d (идём сверху по частоте)
        order = np.argsort(w); ws, ls = w[order], ell[order]
        # сглаженная ell по бинам
        nb = 24; edges = np.linspace(0, wD, nb + 1); wc, lc, Dc = [], [], []
        for k in range(nb):
            m = (ws >= edges[k]) & (ws < edges[k + 1])
            if m.sum() >= 3:
                wc.append(ws[m].mean()); lc.append(np.median(ls[m])); Dc.append(np.median(D[order][m]))
        wc, lc, Dc = map(np.array, (wc, lc, Dc))
        above = np.where(lc > d)[0]
        w_IR = wc[above.max()] if len(above) else np.nan
        rec.update(Dm=Dm, w_IR=w_IR, wc=wc, lc=lc, Dc=Dc)
        print(f"    dphi={dphi:g} dz={z0-Z_C:+.3f} mu={info['mu']:.4e} c={c:.4f} w_D={wD:.3f} d={d:.3f} | "
              f"D диффузонов (0.3-0.9 w_D): медиана {Dm:.4f} = {Dm/(d*c):.3f} d c = {Dm/(d*d*wD):.4f} d^2 w_D; "
              f"w_IR/w_D={w_IR/wD:.2f}  [{time.time()-t0:.0f} с]", flush=True)
        print("      w/w_D:   " + " ".join(f"{x/wD:5.2f}" for x in wc), flush=True)
        print("      D/(d c): " + " ".join(f"{x/(d*c):5.2f}" for x in Dc), flush=True)
        print("      ell/d:   " + " ".join(f"{x/d:5.2f}" for x in lc), flush=True)
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: перенос дрожания по спаянной сети — D(omega) Аллена-Фельдмана (T-4, звено 2)")
    dphis = sorted(set(r["dphi"] for r in recs))
    for d_ in dphis:
        rr = [r for r in recs if r["dphi"] == d_]
        dz = np.mean([r["dz0"] for r in rr])
        Dm = np.array([r["Dm"] / (r["d"] * r["c"]) for r in rr])
        wIR = np.array([r["w_IR"] / r["wD"] for r in rr])
        print(f"\n  == dphi={d_:g} (dz0={dz:.3f}), {len(rr)} образцов ==")
        print(f"  D диффузонов / (d c): {Dm.mean():.3f} ± {Dm.std():.3f}  (по образцам: {', '.join(f'{x:.3f}' for x in Dm)})")
        print(f"  w_IR/w_D (ell = d):   {np.nanmean(wIR):.2f}  (по образцам: {', '.join(f'{x:.2f}' for x in wIR)})")
        # общий профиль
        wc = np.linspace(0.05, 0.95, 19)
        Dc = np.array([np.interp(wc, r["wc"] / r["wD"], r["Dc"] / (r["d"] * r["c"])) for r in rr]).mean(0)
        lc = np.array([np.interp(wc, r["wc"] / r["wD"], r["lc"] / r["d"]) for r in rr]).mean(0)
        print("      w/w_D:   " + " ".join(f"{x:5.2f}" for x in wc))
        print("      D/(d c): " + " ".join(f"{x:5.2f}" for x in Dc))
        print("      ell/d:   " + " ".join(f"{x:5.2f}" for x in lc))
    print("""
  Чтение. D ~ d c (l ~ d) в диффузонной части спектра — перенос диффузионный с планковским пробегом;
  на низких частотах l растёт (пропагоны/рэлеевское рассеяние). Для механизма B важно: любое D >= d c
  означает поток энергии P = 8 pi (D/d^2)(<u^2>_L/d^2) E, см. 7.8 — бюджет энергии.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2")
    ap.add_argument("--neta", type=float, default=4.0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--analyse", type=str, default="")
    args = ap.parse_args()
    if args.analyse:
        recs = []
        for f in args.analyse.split(","):
            recs += list(np.load(f, allow_pickle=True)["recs"])
        analyse(recs); return
    seeds = [int(s) for s in args.seeds.split(",")]
    dphis = [float(s) for s in args.dphi.split(",")]
    head(f"ПЕРЕНОС ДРОЖАНИЯ (Аллен-Фельдман) на спаянной сети: N={args.N}, затравки {seeds}, dphi={dphis}, n_eta={args.neta}")
    out = args.out or os.path.join(HERE, f"jitter_transport_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.neta)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
