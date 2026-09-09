"""Дисперсия поперечной ветви на БОЛЬШОЙ спаянной сети (N = 4096 … 32768) методом Ланцоша (T-2, T-14).

Большой образец нужен, чтобы выйти в область k < k* (линейный поперечный звук маргинальной упаковки:
k* d ≈ 0.7 при N=1024 — Dispersion.py, там это наименьшее k). Полная диагонализация невозможна, поэтому:
  * упаковка без бисекции: phi = phi_J(∞) + dphi, phi_J(∞) ≈ 0.648 (бидисперсная 1:1.4, по N=512…2048),
    релаксация L-BFGS/FIRE при фиксированной кубической ячейке (сдвиговая стабилизация не нужна: при больших N
    остаточный сдвиг мал); спаянная сеть = все контакты двусторонние (гессиан тот же);
  * разреженный гессиан (scipy.sparse), спектральные функции S_T(k, w), S_L(k, w) для k = 2 pi n / L вдоль
    трёх осей — процедурой Ланцоша (M шагов) по стартовому вектору v = pol ⊗ exp(i k r) (действительная и
    мнимая части отдельно), спектр через собственные значения трёхдиагональной матрицы с гауссовым сглаживанием;
  * аппроксимация DHO -> Omega(k), Gamma(k); c_T = sqrt(mu/rho) по гессиану — mu из разреженного решателя
    (неаффинная поправка Xi^T H^-1 Xi через CG);
  * статический S(k).

Запуск: python DispersionLarge.py [--N 8192] [--seeds 1] [--dphi 3e-2,1e-1] [--nmax 8] [--M 300]
        python DispersionLarge.py --analyse dispersion_large_N8192.npz
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import SoftSpheres, head, Z_C  # noqa: E402
from GravitySoftening import SHEARS  # noqa: E402

PHI_J_INF = 0.648


def dho(w, A, Om, Ga):
    return A * Ga * Om ** 2 / ((w ** 2 - Om ** 2) ** 2 + (w * Ga) ** 2)


def fit_dho(wgrid, S):
    k = int(np.argmax(S)); Om0 = wgrid[k]
    half = S[k] / 2
    lo = k
    while lo > 0 and S[lo] > half: lo -= 1
    hi = k
    while hi < len(S) - 1 and S[hi] > half: hi += 1
    Ga0 = max(wgrid[hi] - wgrid[lo], wgrid[1] - wgrid[0])
    m = wgrid <= 2.5 * max(Om0, wgrid[3])          # окно подгонки вокруг пика, хвост DOS не цепляем
    try:
        popt, _ = curve_fit(dho, wgrid[m], S[m], p0=(S[k] * Ga0, Om0, Ga0), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        return popt[1], popt[2]
    except Exception:
        return Om0, Ga0


def sparse_hessian(p):
    """Разреженный гессиан по контактам (двусторонние гармонические пружины), плюс данные пар."""
    i, j, dx, r, ov = p.contacts()
    sig = p.rad[i] + p.rad[j]
    dVdr = -ov / sig; d2V = 1.0 / sig ** 2
    n = dx / r[:, None]
    K = (d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :] + (dVdr / r)[:, None, None] * np.eye(3)[None]
    N = p.N
    rows, cols, vals = [], [], []
    for a_ in range(3):
        for b_ in range(3):
            kab = K[:, a_, b_]
            rows += [3 * i + a_, 3 * j + a_, 3 * i + a_, 3 * j + a_]
            cols += [3 * i + b_, 3 * j + b_, 3 * j + b_, 3 * i + b_]
            vals += [kab, kab, -kab, -kab]
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * N, 3 * N))
    return H, (i, j, dx, r, ov, K, dVdr, n)


def shear_moduli_sparse(p, H, pairs):
    i, j, dx, r, ov, K, dVdr, n = pairs
    N = p.N; V = p.L ** 3
    g = dVdr[:, None] * n
    Hr = H + sp.identity(3 * N) * (1e-9 * H.diagonal().mean())
    mus = []
    for (a_, b_) in SHEARS:
        e = np.zeros_like(dx); e[:, a_] = dx[:, b_]
        xi_pair = np.einsum('kab,kb->ka', K, e)
        Xi = np.zeros((N, 3)); np.add.at(Xi, j, xi_pair); np.add.at(Xi, i, -xi_pair); Xi = Xi.ravel()
        aff = ((xi_pair[:, a_] * dx[:, b_]) + g[:, a_] * e[:, b_]).sum() / V
        x, info = cg(Hr, Xi, rtol=1e-8, maxiter=20000)
        mus.append(aff - Xi @ x / V)
    return np.array(mus)


def lanczos_spectrum(H, v0, M, wgrid, eta):
    """Спектральная функция <v0| delta(w^2 - H) |v0> через M шагов Ланцоша; возвращает S(w) по wgrid
    (в переменной w = sqrt(lambda), с якобианом 2w и гауссовым сглаживанием по w)."""
    v0 = v0 / np.linalg.norm(v0)
    V_prev = np.zeros_like(v0); v = v0.copy(); beta = 0.0
    alphas, betas = [], []
    for m in range(M):
        w_ = H @ v - beta * V_prev
        alpha = v @ w_
        w_ -= alpha * v
        # полная реортогонализация не делаем (дорого); частичная — против двух последних
        beta_new = np.linalg.norm(w_)
        alphas.append(alpha);
        if beta_new < 1e-12 or m == M - 1:
            break
        betas.append(beta_new)
        V_prev, v, beta = v, w_ / beta_new, beta_new
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    lam, U = np.linalg.eigh(T)
    weights = U[0, :] ** 2
    lam = np.maximum(lam, 0.0); wl = np.sqrt(lam)
    S = np.zeros_like(wgrid)
    for x, wt in zip(wl, weights):
        S += wt * np.exp(-0.5 * ((wgrid - x) / eta) ** 2) / (math.sqrt(2 * math.pi) * eta)
    return S


def static_sk(p, nmax=10):
    x = np.mod(p.s, p.L) @ p.F().T; L = p.L
    rng = range(-nmax, nmax + 1); ks, Sv = [], []
    for nx in rng:
        for ny in rng:
            for nz in rng:
                if nx == ny == nz == 0: continue
                if nx < 0 or (nx == 0 and ny < 0) or (nx == 0 and ny == 0 and nz < 0): continue
                if nx * nx + ny * ny + nz * nz > nmax * nmax: continue
                kv = 2 * math.pi / L * np.array([nx, ny, nz])
                ks.append(np.linalg.norm(kv)); Sv.append(abs(np.exp(1j * (x @ kv)).sum()) ** 2 / p.N)
    ks, Sv = np.array(ks), np.array(Sv)
    uk, inv = np.unique(np.round(ks, 6), return_inverse=True)
    return uk, np.array([Sv[inv == q].mean() for q in range(len(uk))])


def run_sample(N, seed, dphis, nmax, M, eta_frac=0.01, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed} ---", flush=True)
    out = []
    for dphi in dphis:
        p = SoftSpheres(N, PHI_J_INF + dphi, seed)
        cache = os.path.join(HERE, f"packing_N{N}_s{seed}_dphi{dphi:g}.npz")
        if os.path.exists(cache):
            z = np.load(cache); p.s = z["s"]; p.L = float(z["L"]); p.rad = z["rad"]; p._pairs = None
            print(f"    упаковка из кэша {os.path.basename(cache)}", flush=True)
        else:
            p.minimize(rel=1e-9)
            p.minimize(rel=1e-10)      # дожать (большие системы)
            np.savez(cache, s=p.s, L=p.L, rad=p.rad)
        z0, nrat, nc = p.coordination(); _, pr = p.stress_pressure()
        H, pairs = sparse_hessian(p)
        mus = shear_moduli_sparse(p, H, pairs)
        mu = mus.mean(); V = p.L ** 3; rho = N / V; cT = math.sqrt(max(mu, 1e-30) / rho)
        d = 2 * p.rad.mean(); L = p.L
        # оценка w_D: несколько шагов степенного метода
        v = np.random.default_rng(seed).standard_normal(3 * N)
        for _ in range(60):
            v = H @ v; v /= np.linalg.norm(v)
        wD = math.sqrt(v @ (H @ v))
        print(f"    dphi={dphi:g} phi={p.phi:.4f} dz={z0-Z_C:+.3f} p={pr:.3e} L={L:.2f} mu6={np.array2string(mus, precision=4)} "
              f"mu={mu:.4e} c_T={cT:.4f} w_D~{wD:.3f}  [{time.time()-t0:.0f} с]", flush=True)
        x = np.mod(p.s, p.L) @ p.F().T
        eta = eta_frac * wD; wgrid = np.linspace(0, 1.05 * wD, 1200)
        axes = np.eye(3)
        kk = 2 * math.pi * np.arange(1, nmax + 1) / L
        OmT, GaT, OmL, GaL = [], [], [], []
        for n in range(1, nmax + 1):
            ST = np.zeros_like(wgrid); SL = np.zeros_like(wgrid)
            for ax in range(3):
                kv = 2 * math.pi * n / L * axes[ax]
                ph = x @ kv
                for pol in (axes[(ax + 1) % 3], axes[(ax + 2) % 3]):
                    for f in (np.cos, np.sin):
                        v0 = (f(ph)[:, None] * pol[None, :]).ravel()
                        ST += lanczos_spectrum(H, v0, M, wgrid, eta)
                for f in (np.cos, np.sin):
                    v0 = (f(ph)[:, None] * axes[ax][None, :]).ravel()
                    SL += lanczos_spectrum(H, v0, M, wgrid, eta)
            o, g = fit_dho(wgrid, ST); OmT.append(o); GaT.append(g)
            o, g = fit_dho(wgrid, SL); OmL.append(o); GaL.append(g)
            print(f"      n={n} k d={kk[n-1]*d:.3f}: Om_T={OmT[-1]:.4f} (Om_T/k c_T={OmT[-1]/(kk[n-1]*cT):.3f}), Ga_T/Om_T={GaT[-1]/OmT[-1]:.3f}, "
                  f"l_T/d={cT/GaT[-1]/d:.2f}; Om_L/k={OmL[-1]/kk[n-1]:.4f}  [{time.time()-t0:.0f} с]", flush=True)
        OmT, GaT, OmL, GaL = map(np.array, (OmT, GaT, OmL, GaL))
        uk, Sk = static_sk(p, nmax=8)
        rec = dict(N=N, seed=seed, dphi=dphi, phi=p.phi, dz0=z0 - Z_C, p=pr, d=d, L=L, wD=wD, mu=mu, mus=mus, rho=rho, cT_mod=cT,
                   k=kk, OmT=OmT, GaT=GaT, OmL=OmL, GaL=GaL, uk=uk, Sk=Sk)
        print("      S(k) при k d=" + ", ".join(f"{u*d:.2f}:{s:.3f}" for u, s in zip(uk[:6], Sk[:6])), flush=True)
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: дисперсия на большой спаянной сети (Ланцош) — T-2, T-14")
    dphis = sorted(set(r["dphi"] for r in recs))
    for d_ in dphis:
        rr = [r for r in recs if r["dphi"] == d_]
        dz = np.mean([r["dz0"] for r in rr]); Ns = sorted(set(r["N"] for r in rr))
        print(f"\n  == dphi={d_:g} (dz0={dz:.3f}), N={Ns}, {len(rr)} образцов ==")
        kd = np.concatenate([r["k"] * r["d"] for r in rr]); ratio = np.concatenate([r["OmT"] / (r["k"] * r["cT_mod"]) for r in rr])
        gam = np.concatenate([r["GaT"] / r["OmT"] for r in rr]); ell = np.concatenate([r["cT_mod"] / r["GaT"] / r["d"] for r in rr])
        gabs = np.concatenate([r["GaT"] / r["wD"] for r in rr])
        print(f"  {'k d':>6} {'Om_T/(k c_mod)':>15} {'Ga_T/Om_T':>10} {'Ga_T/w_D':>9} {'l_T/d':>7}")
        for u in np.unique(np.round(kd, 2)):
            m = np.abs(kd - u) < 0.005
            print(f"  {u:>6.2f} {ratio[m].mean():>15.3f} {gam[m].mean():>10.3f} {gabs[m].mean():>9.4f} {ell[m].mean():>7.2f}")
        m = kd < 0.9
        if m.sum() >= 3:
            A = np.vstack([np.ones(m.sum()), -(kd[m]) ** 2]).T
            c0r, b = np.linalg.lstsq(A, ratio[m], rcond=None)[0]
            print(f"  k d < 0.9: Om_T/(k c_mod) -> {c0r:.3f} при k->0; xi_2 = {b/c0r:+.3f} (Om = c k [1 - xi_2 (k d)^2])")
            pfit = np.polyfit(np.log(kd[m]), np.log(gabs[m]), 1)[0]
            print(f"  затухание Ga_T ∝ k^p, p = {pfit:.2f} (k d < 0.9)")
        ukd = np.concatenate([r["uk"] * r["d"] for r in rr]); Sk = np.concatenate([r["Sk"] for r in rr])
        o = np.argsort(ukd); ukd, Sk = ukd[o], Sk[o]
        vals = [(u, Sk[np.abs(ukd - u) < 0.005].mean()) for u in np.unique(np.round(ukd, 2))[:8]]
        print("  S(k): " + "  ".join(f"kd={u:.2f}:{s:.3f}" for u, s in vals))
        m = ukd < 1.5
        if m.sum() >= 4:
            al = np.polyfit(np.log(ukd[m]), np.log(np.maximum(Sk[m], 1e-6)), 1)[0]
            print(f"  гипероднородность: S(k) ∝ k^alpha, alpha = {al:.2f} (k d < 1.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--seeds", type=str, default="1")
    ap.add_argument("--dphi", type=str, default="3e-2,1e-1")
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--M", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.01, help="гауссово сглаживание в долях w_D")
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
    head(f"ДИСПЕРСИЯ (Ланцош) большой спаянной сети: N={args.N}, затравки {seeds}, dphi={dphis}, nmax={args.nmax}, M={args.M}, eta={args.eta} w_D")
    out = args.out or os.path.join(HERE, f"dispersion_large_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.nmax, args.M, args.eta)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
