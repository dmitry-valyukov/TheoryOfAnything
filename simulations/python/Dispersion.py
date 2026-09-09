"""Дисперсия поперечной ветви, затухание и пробег на спаянной заклиненной сети (T-2, T-1 остаток, T-14).

Считается по собственным модам гармонического гессиана спаянной сети (BondedNet из StateDrift.py):
  * динамический структурный фактор смещений S_T(k, w) = sum_m |sum_j (e_m,j . e_T) exp(i k r_j)|^2 delta(w - w_m)
    (две поперечные поляризации) и S_L(k, w) (продольная) для k = 2 pi n / L вдоль трёх осей ячейки, n = 1 … n_max;
    аппроксимация затухающим гармоническим осциллятором (DHO) -> Omega_T(k), Gamma_T(k); то же для L;
  * скорость звука c_T = lim Omega/k, сравнение с sqrt(mu/rho) по гессиановым модулям;
    отклонение от линейности: Omega = c k [1 - xi_2 (k d)^2] -> xi_2;
  * закон затухания Gamma(k) ∝ k^p (рэлеевское p = 4 против p = 2), пробег l = c / Gamma;
  * плотность состояний D(w), D(w)/w^2 и частота w* (максимум D/w^2 — бозонный пик / начало плато по Вьярту);
  * статический структурный фактор S(k) упаковки по сферическим оболочкам и показатель гипероднородности
    S(k) ∝ k^alpha при малых k.

Запуск: python Dispersion.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2] [--nmax 6]
        python Dispersion.py --analyse dispersion_N512.npz
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from StateDrift import BondedNet, assemble_full, moduli_full  # noqa: E402


def dho(w, A, Om, Ga):
    return A * Ga * Om ** 2 / ((w ** 2 - Om ** 2) ** 2 + (w * Ga) ** 2)


def fit_dho(wgrid, S):
    """Подгонка DHO по сетке; старт — положение максимума и ширина на полувысоте."""
    k = int(np.argmax(S)); Om0 = wgrid[k]
    half = S[k] / 2
    lo = k
    while lo > 0 and S[lo] > half: lo -= 1
    hi = k
    while hi < len(S) - 1 and S[hi] > half: hi += 1
    Ga0 = max(wgrid[hi] - wgrid[lo], wgrid[1] - wgrid[0])
    try:
        popt, _ = curve_fit(dho, wgrid, S, p0=(S[k] * Ga0, Om0, Ga0), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        return popt[1], popt[2], True
    except Exception:
        return Om0, Ga0, False


def structure_factor_modes(p, w, E, kvecs, pol):
    """Спектральные веса |sum_j (e_m,j . pol) exp(i k r_j)|^2 для каждого k и моды m. pol: (nk, 3) единичный."""
    x = np.mod(p.s, p.L) @ p.F().T      # реальные координаты
    N = p.N
    out = np.zeros((len(kvecs), E.shape[1]))
    for a, kv in enumerate(kvecs):
        ph = np.exp(1j * (x @ kv))        # (N,)
        Em = E.reshape(N, 3, -1)           # (N,3,M)
        proj = np.einsum('njm,j->nm', Em, pol[a])   # (N, M)
        amp = (ph[:, None] * proj).sum(axis=0)
        out[a] = np.abs(amp) ** 2 / N
    return out


def static_sk(p, nmax=12):
    """S(k) по всем целочисленным векторам |n| <= nmax, усреднение по оболочкам |k|."""
    x = np.mod(p.s, p.L) @ p.F().T
    L = p.L
    rng = range(-nmax, nmax + 1)
    ks, Sv = [], []
    for nx in rng:
        for ny in rng:
            for nz in rng:
                if nx == ny == nz == 0: continue
                if nx < 0 or (nx == 0 and ny < 0) or (nx == 0 and ny == 0 and nz < 0): continue
                n2 = nx * nx + ny * ny + nz * nz
                if n2 > nmax * nmax: continue
                kv = 2 * math.pi / L * np.array([nx, ny, nz])
                rho_k = np.exp(1j * (x @ kv)).sum()
                ks.append(np.linalg.norm(kv)); Sv.append(abs(rho_k) ** 2 / p.N)
    ks, Sv = np.array(ks), np.array(Sv)
    # оболочки по |n|^2
    order = np.argsort(ks); ks, Sv = ks[order], Sv[order]
    uk, inv = np.unique(np.round(ks, 6), return_inverse=True)
    Sk = np.array([Sv[inv == i].mean() for i in range(len(uk))])
    return uk, Sk


def run_sample(N, seed, dphis, nmax, verbose=True):
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
        H, Xis, aff, V, pr = assemble_full(b)
        w2, E = np.linalg.eigh(H)
        keep = w2 > 1e-8 * w2.max()
        w = np.sqrt(w2[keep]); E = E[:, keep]
        mus, B, _, _ = moduli_full(b)
        mu = mus.mean(); rho = N / V; cT_mod = math.sqrt(mu / rho); cL_mod = math.sqrt((B + 4 * mu / 3) / rho)
        d = 2 * b.rad.mean(); L = b.L; wD = w.max()
        # k-векторы вдоль осей
        kvecs, polT1, polT2, polL, kidx = [], [], [], [], []
        axes = np.eye(3)
        for n in range(1, nmax + 1):
            for ax in range(3):
                kv = 2 * math.pi * n / L * axes[ax]
                kvecs.append(kv); kidx.append(n)
                t1 = axes[(ax + 1) % 3]; t2 = axes[(ax + 2) % 3]
                polT1.append(t1); polT2.append(t2); polL.append(axes[ax])
        kvecs = np.array(kvecs); kidx = np.array(kidx)
        AT = structure_factor_modes(b, w, E, kvecs, np.array(polT1)) + structure_factor_modes(b, w, E, kvecs, np.array(polT2))
        AL = structure_factor_modes(b, w, E, kvecs, np.array(polL))
        # спектры на сетке с гауссовым сглаживанием
        eta = 0.01 * wD
        wgrid = np.linspace(0, 1.05 * wD, 420)
        Gm = np.exp(-0.5 * ((wgrid[:, None] - w[None, :]) / eta) ** 2) / (math.sqrt(2 * math.pi) * eta)
        OmT, GaT, OmL, GaL, okT = [], [], [], [], []
        for n in range(1, nmax + 1):
            m = kidx == n
            ST = (AT[m][:, :, None] * Gm.T[None, :, :]).sum(axis=(0, 1)) / m.sum()
            SL = (AL[m][:, :, None] * Gm.T[None, :, :]).sum(axis=(0, 1)) / m.sum()
            # ограничим подгонку областью до первого минимума после пика (чтобы не цеплять хвост)
            o, g, ok = fit_dho(wgrid, ST); OmT.append(o); GaT.append(g); okT.append(ok)
            o, g, ok = fit_dho(wgrid, SL); OmL.append(o); GaL.append(g)
        kk = 2 * math.pi * np.arange(1, nmax + 1) / L
        OmT, GaT, OmL, GaL = map(np.array, (OmT, GaT, OmL, GaL))
        # DOS и w*
        hist, edges = np.histogram(w, bins=60, range=(0, wD)); wc = 0.5 * (edges[1:] + edges[:-1])
        dos = hist / (len(w) * (edges[1] - edges[0]))
        red = dos / np.maximum(wc, 1e-12) ** 2
        red_s = np.convolve(red, np.ones(5) / 5, mode='same')
        w_star = wc[np.argmax(red_s[2:-2]) + 2]
        # статический S(k)
        uk, Sk = static_sk(b, nmax=10)
        rec = dict(N=N, seed=seed, dphi=dphi, dz0=z0 - Z_C, d=d, L=L, wD=wD, mu=mu, B=B, rho=rho, cT_mod=cT_mod, cL_mod=cL_mod,
                   k=kk, OmT=OmT, GaT=GaT, OmL=OmL, GaL=GaL, wc=wc, dos=dos, w_star=w_star, uk=uk, Sk=Sk, z=z0)
        # оценки
        cT_k = OmT[0] / kk[0]; cT_k2 = OmT[1] / kk[1]
        print(f"    dphi={dphi:g} dz={z0-Z_C:+.3f} L={L:.2f} d={d:.2f} w_D={wD:.3f} | c_T(модули)={cT_mod:.4f}, c_T(k1)={cT_k:.4f}, c_T(k2)={cT_k2:.4f}; "
              f"c_L(модули)={cL_mod:.4f}, c_L(k1)={OmL[0]/kk[0]:.4f}; w*={w_star:.3f} ({w_star/wD:.2f} w_D)  [{time.time()-t0:.0f} с]", flush=True)
        print("      k d:      " + " ".join(f"{x*d:6.3f}" for x in kk), flush=True)
        print("      Om_T/k c: " + " ".join(f"{o/(x*cT_mod):6.3f}" for o, x in zip(OmT, kk)), flush=True)
        print("      Ga_T/Om_T:" + " ".join(f"{g/o:6.3f}" for g, o in zip(GaT, OmT)), flush=True)
        print("      l_T/d:    " + " ".join(f"{cT_mod/g/d:6.2f}" for g in GaT), flush=True)
        print("      Om_L/k c: " + " ".join(f"{o/(x*cL_mod):6.3f}" for o, x in zip(OmL, kk)), flush=True)
        print("      S(k) при k d=" + ", ".join(f"{x*d:.2f}:{s:.3f}" for x, s in zip(uk[:6], Sk[:6])), flush=True)
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: дисперсия, затухание, пробег, S(k) на спаянной сети (T-2, T-14)")
    dphis = sorted(set(r["dphi"] for r in recs))
    Ns = sorted(set(r["N"] for r in recs))
    print(f"  N={Ns}, dphi={dphis}, образцов {len(recs)}")
    for d_ in dphis:
        rr = [r for r in recs if r["dphi"] == d_]
        dz = np.mean([r["dz0"] for r in rr])
        print(f"\n  == dphi={d_:g} (dz0={dz:.3f}), {len(rr)} образцов ==")
        # дисперсия: объединяем все (k d, Om/(k c_mod)) точки
        kd = np.concatenate([r["k"] * r["d"] for r in rr]); ratio = np.concatenate([r["OmT"] / (r["k"] * r["cT_mod"]) for r in rr])
        gam = np.concatenate([r["GaT"] / r["wD"] for r in rr]); ell = np.concatenate([r["cT_mod"] / r["GaT"] / r["d"] for r in rr])
        om = np.concatenate([r["OmT"] / r["wD"] for r in rr])
        order = np.argsort(kd)
        # бины по kd
        ub = np.unique(np.round(kd, 2))
        print(f"  {'k d':>6} {'Om_T/(k c_mod)':>15} {'Om_T/w_D':>9} {'Ga_T/w_D':>9} {'l_T/d':>7}")
        for u in ub:
            m = np.abs(kd - u) < 0.005
            print(f"  {u:>6.2f} {ratio[m].mean():>15.3f} {om[m].mean():>9.3f} {gam[m].mean():>9.4f} {ell[m].mean():>7.2f}")
        # xi_2: Om/(k c0) = 1 - xi2 (k d)^2, c0 по первым двум точкам
        m = kd < 1.2
        if m.sum() >= 3:
            A = np.vstack([np.ones(m.sum()), -(kd[m]) ** 2]).T
            c0r, xi2 = np.linalg.lstsq(A, ratio[m], rcond=None)[0]
            print(f"  линейная область (k d<1.2): Om_T/(k c_mod) -> {c0r:.3f} при k->0 (c_T(k->0)/c_T(модули)), xi_2 = {xi2/c0r:+.3f} (Om = c k [1 - xi_2 (k d)^2])")
        # затухание: Ga ∝ k^p
        m = (kd < 1.5) & (gam > 0)
        if m.sum() >= 3:
            pfit = np.polyfit(np.log(kd[m]), np.log(gam[m]), 1)[0]
            print(f"  затухание Ga_T ∝ k^p, p = {pfit:.2f} (k d < 1.5); пробег l_T/d при наименьшем k: {ell[order][:3].mean():.1f}")
        ws = np.mean([r["w_star"] / r["wD"] for r in rr])
        print(f"  w*/w_D (макс D(w)/w^2) = {ws:.3f};  c_T(k1)/c_T(модули) = {np.mean([r['OmT'][0]/r['k'][0]/r['cT_mod'] for r in rr]):.3f}")
        # S(k)
        ukd = np.concatenate([r["uk"] * r["d"] for r in rr]); Sk = np.concatenate([r["Sk"] for r in rr])
        o = np.argsort(ukd); ukd, Sk = ukd[o], Sk[o]
        ub = np.unique(np.round(ukd, 2))
        vals = [(u, Sk[np.abs(ukd - u) < 0.005].mean()) for u in ub[:10]]
        print("  S(k): " + "  ".join(f"kd={u:.2f}:{s:.3f}" for u, s in vals))
        m = ukd < 2.0
        if m.sum() >= 4:
            al = np.polyfit(np.log(ukd[m]), np.log(np.maximum(Sk[m], 1e-6)), 1)[0]
            print(f"  гипероднородность: S(k) ∝ k^alpha, alpha = {al:.2f} (k d < 2)")
    print("""
  Чтение. Om_T/(k c) -> 1 при k->0 — одна поперечная ветвь с c = sqrt(mu/rho); xi_2 — знак и величина
  поправки второго порядка (T-2); Ga ∝ k^4 — рэлеевское рассеяние на беспорядке, l ∝ k^-4 (T-14);
  w* — начало аномальных мод (Вьярт); alpha > 0 — гипероднородность упаковки.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2")
    ap.add_argument("--nmax", type=int, default=6)
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
    head(f"ДИСПЕРСИЯ спаянной сети: N={args.N}, затравки {seeds}, dphi={dphis}, nmax={args.nmax}")
    out = args.out or os.path.join(HERE, f"dispersion_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.nmax)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
