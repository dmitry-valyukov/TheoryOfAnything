"""Профиль кандидата (а′) гравитации на большом образце (T-4, продолжение 7.9).

7.9 (EntanglementArea.py, N=1024, плотная диагонализация) нашёл возмущение энтропии от
включения dS(r) со спадом между r^-2 и r^-3 на доступных 2–4d — различить 1/r (нужен для
ньютоновского потенциала) и 1/r^2 нельзя. Плотный метод (eigh 3N x 3N) до N~10^4 не масштабируется.

Здесь — разреженный ТОЧНЫЙ линейный отклик на N>=8192, дающий показатель спада чисто:

  (1) Функция Грина упругой среды G(r): отклик смещения на точечную силу (с уравновешивающей
      фоновой силой; трансляции дефлированы). В континуальной 3D-упругости G ~ 1/r. Это
      «потенциалоподобное» поле.
  (2) Возмущение энергии дрожания dU2(r) = delta<u^2>(r) от жёсткого включения (контакты в шаре
      1.5d жёстче в 4 раза — та же модель узла, что 7.9), классический тепловой отклик
      <u_i^2> = T [H^{-1}]_ii, точный линейный отклик
          delta<u^2>_i = -T [G dH G]_ii = -T (Y_i) dH_CC (Y_i)^T,  Y_a = H^+ e_a, a in C(включение).
      Монополь (среднее по сфере) и показатель спада — то, что нужно для (а′).

Ожидание из линейного отклика: delta<u^2> ~ G^2 ~ 1/r^2 (произведение двух функций Грина),
монополь НЕ нуль (в отличие от статики 7.6, где девиатор даёт r^-3 со средним нуль). Проверяем:
даёт ли (а′) нужный 1/r или тот же 1/r^2, что закрывает энтропийный фланг наравне со статикой.

Метод решения: разреженный H (спаянная сеть), дефляция 3 трансляций, CG с якобиевым
предобуславливателем. Число решений = 1 (G) + |C| (включение) ~ несколько десятков.

Запуск: python EntropyProfileLarge.py --N 8192 --seed 1 [--dphi 3e-2] [--out entropy_profile_N8192.npz]
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from StateDrift import BondedNet  # noqa: E402


def build_sparse_H(b, center=None, r0=None, lam=1.0):
    """Разреженный гессиан спаянной сети. Если задан шар (center,r0), контакты с серединой
    в нём жёстче в lam раз (включение)."""
    i, j, dx, r, ov = b.contacts()
    x = np.mod(b.s, b.L) @ b.F().T
    sig = b.rad[i] + b.rad[j]
    dVdr = -ov / sig
    d2V = np.ones_like(r) / sig ** 2
    if center is not None:
        mid = x[i] + 0.5 * dx
        dm = mid - center; dm -= b.L * np.round(dm / b.L)
        hot = (dm ** 2).sum(axis=1) < r0 * r0
        d2V[hot] *= lam; dVdr[hot] *= lam
    n = dx / r[:, None]
    K = ((d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :]
         + (dVdr / r)[:, None, None] * np.eye(3)[None])
    N = b.N
    rows = []; cols = []; vals = []
    for a_ in range(3):
        for c_ in range(3):
            for (I, J, s) in ((3 * i + a_, 3 * i + c_, +1), (3 * j + a_, 3 * j + c_, +1),
                              (3 * i + a_, 3 * j + c_, -1), (3 * j + a_, 3 * i + c_, -1)):
                rows.append(I); cols.append(J); vals.append(s * K[:, a_, c_])
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N))
    return H, (hot.sum() if center is not None else 0)


def null_translations(N):
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1.0
    T /= np.linalg.norm(T, axis=0)
    return T


def make_solver(H, Tnull):
    """Прямой разреженный решатель H^+ b. Трансляции сняты крошечным риджем eps*I
    (длина экранирования ~ sqrt(meandiag/eps) >> ящика, дальнее поле не искажается),
    RHS и решение проецируются ⟂ трансляций, чтобы убить вклад ~1/eps."""
    meandiag = float(H.diagonal().mean())
    eps = 1e-9 * meandiag
    Hreg = (H + eps * sp.eye(H.shape[0], format="csc")).tocsc()
    lu = splu(Hreg)

    def project(v):
        return v - Tnull @ (Tnull.T @ v)

    def solve(b):
        bb = project(b)
        y = lu.solve(bb)
        return project(y)
    return solve


def shell_fit(rvals, field, d, L, rmin=2.0, rmax=None, nbin=16, mincount=15):
    """Среднее по сферическим оболочкам (монополь). rmax по умолчанию 0.35 L (вдали от образов)."""
    if rmax is None:
        rmax = 0.35 * L / d
    edges = np.linspace(rmin, rmax, nbin + 1)
    rc, fc, nc = [], [], []
    for a, bnd in zip(edges[:-1], edges[1:]):
        m = (rvals / d >= a) & (rvals / d < bnd)
        if m.sum() < mincount:
            continue
        rc.append(rvals[m].mean() / d); fc.append(field[m].mean()); nc.append(int(m.sum()))
    return np.array(rc), np.array(fc), np.array(nc)


def loglog_slope(rc, fc):
    m = np.isfinite(fc) & (np.abs(fc) > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(rc[m]), np.log(np.abs(fc[m])), 1)[0])


def fit_power_offset(rc, fc):
    """Модель |f| = A r^-p + c (c — фон от мягкой моды/образов). Возвращает (p, c, p_чист).
    p_чист — наклон после вычитания c."""
    from scipy.optimize import curve_fit
    y = np.abs(fc)
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 4:
        return np.nan, np.nan, loglog_slope(rc, fc)
    r, y = rc[m], y[m]
    try:
        popt, _ = curve_fit(lambda r, A, p, c: A * r ** (-p) + c, r, y,
                            p0=[y[0] * r[0], 2.0, y[-1]], maxfev=20000,
                            bounds=([0, 0, -abs(y).max()], [np.inf, 6, abs(y).max()]))
        A, p, c = popt
        # чистый наклон: убрать оценённый c и пересчитать
        ycl = y - c
        mm = ycl > 0
        pcl = -np.polyfit(np.log(r[mm]), np.log(ycl[mm]), 1)[0] if mm.sum() >= 3 else np.nan
        return float(p), float(c), float(pcl)
    except Exception:
        return np.nan, np.nan, loglog_slope(rc, fc)


def run(N, seed, dphi, out):
    t0 = time.time()
    cache = os.path.join(HERE, f"_packing_bonded_N{N}_s{seed}_dphi{dphi:g}.npz")
    head(f"ПРОФИЛЬ (а′) НА БОЛЬШОМ ОБРАЗЦЕ: N={N}, seed={seed}, dphi={dphi:g}")
    if os.path.exists(cache):
        z = np.load(cache)
        b = BondedNet(N, 0.65, seed); b.rad = z["rad"]; b.vol_particles = float(z["volp"])
        b.rc = float(z["rc"]); b.L = float(z["L"]); b.s = z["s"]; b.G3 = z["G3"]
        b.bond(z["bi"], z["bj"]); b.minimize(rel=1e-10)
        print(f"  упаковка из кэша {cache}  [{time.time()-t0:.0f} с]", flush=True)
    else:
        print("  генерация заклиненной упаковки (быстрый путь: прямое сжатие)...", flush=True)
        # прямое сжатие до phi выше phi_J(~0.65) и одна минимизация вместо бисекции
        phi_target = 0.65 + dphi + 0.033      # ~0.68 при dphi=0.03 -> Δz~1.3
        p = PinnedSpheres(N, phi_target, seed)
        p.minimize(rel=1e-6)
        assert p.energy() / N > 1e-16, "не заклинено"
        p.stabilize()
        i0, j0, *_ = p.contacts()
        b = BondedNet(N, 0.65, seed); b.rad = p.rad.copy(); b.vol_particles = p.vol_particles
        b.rc = p.rc; b.L = p.L; b.s = p.s.copy(); b.G3 = p.G3.copy(); b.bond(i0, j0)
        b.minimize(rel=1e-10)
        np.savez(cache, rad=b.rad, volp=b.vol_particles, rc=b.rc, L=b.L, s=b.s, G3=b.G3,
                 bi=i0, bj=j0)
        print(f"  упаковка построена и сохранена  [{time.time()-t0:.0f} с]", flush=True)

    z0, _, _ = b.coordination()
    d = 2 * b.rad.mean(); L = b.L
    x = np.mod(b.s, b.L) @ b.F().T
    print(f"  N={N}, L={L:.1f} ({L/d:.1f} d), dz={z0-Z_C:+.2f}, полуящик {0.45*L/d:.1f} d", flush=True)

    H, _ = build_sparse_H(b)
    Tnull = null_translations(N)
    solve = make_solver(H, Tnull)

    rng = np.random.default_rng(7000 + seed)
    npos = 24             # число положений источника/включения для усреднения
    # случайные центры-частицы, не у края
    cand = np.arange(N)
    centers_i = rng.choice(cand, npos, replace=False)

    # ---- (1) функция Грина: усреднение по источникам и направлениям силы ----
    print(f"\n  (1) функция Грина G(r): {npos} источников, случайные направления...", flush=True)
    shellsG = {}   # r-бин -> список |u|
    for ci in centers_i:
        cen = x[ci].copy()
        rvec = x - cen; rvec -= L * np.round(rvec / L)
        rr = np.linalg.norm(rvec, axis=1)
        fdir = rng.normal(size=3); fdir /= np.linalg.norm(fdir)
        f = np.zeros(3 * N); f[3 * ci:3 * ci + 3] = fdir
        u = solve(f).reshape(N, 3)
        umag = np.linalg.norm(u, axis=1)
        rc, fc, nc = shell_fit(rr, umag, d, L, rmin=1.5, rmax=0.30 * L / d)
        for rr_, ff_ in zip(rc, fc):
            shellsG.setdefault(round(rr_, 1), []).append(ff_)
    rcG = np.array(sorted(shellsG)); fcG = np.array([np.mean(shellsG[k]) for k in rcG])
    slopeG = loglog_slope(rcG, fcG)
    print("   r/d:   " + " ".join(f"{v:6.2f}" for v in rcG), flush=True)
    print("   |u|:   " + " ".join(f"{v:6.1e}" for v in fcG), flush=True)
    print(f"   => G(r) ~ r^({slopeG:.2f})   (континуум упругости: -1)", flush=True)

    # ---- (2) возмущение энергии дрожания от жёсткого включения, усреднение по положениям ----
    print(f"\n  (2) возмущение дрожания delta<u^2>(r) от включения (x4, шар 1.5d), {npos} положений...", flush=True)
    shells = {}; shellsAbs = {}; nhot_tot = 0; nC_tot = 0
    for ipos, ci in enumerate(centers_i):
        cen = x[ci].copy()
        rvec = x - cen; rvec -= L * np.round(rvec / L)
        rr = np.linalg.norm(rvec, axis=1)
        H1, nhot = build_sparse_H(b, center=cen, r0=1.5 * d, lam=4.0)
        dHm = (H1 - H)
        dH = dHm.tocoo()
        Cset = np.unique(np.concatenate([dH.row, dH.col]))
        dHcc = np.asarray(dHm[np.ix_(Cset, Cset)].todense())
        Y = np.zeros((3 * N, len(Cset)))
        for k, a in enumerate(Cset):
            e = np.zeros(3 * N); e[a] = 1.0
            Y[:, k] = solve(e)
        q = ((Y @ dHcc) * Y).sum(axis=1)
        du2 = -(q[0::3] + q[1::3] + q[2::3])
        rc2, fc2, _ = shell_fit(rr, du2, d, L, rmin=2.0, rmax=0.30 * L / d)
        rc2a, fc2a, _ = shell_fit(rr, np.abs(du2), d, L, rmin=2.0, rmax=0.30 * L / d)
        for rr_, ff_ in zip(rc2, fc2):
            shells.setdefault(round(rr_, 1), []).append(ff_)
        for rr_, ff_ in zip(rc2a, fc2a):
            shellsAbs.setdefault(round(rr_, 1), []).append(ff_)
        nhot_tot += nhot; nC_tot = len(Cset)
        print(f"     положение {ipos+1}/{npos} (ci={ci}, {nhot} гор. контактов)  [{time.time()-t0:.0f} с]", flush=True)
    rc2 = np.array(sorted(shells)); fc2 = np.array([np.mean(shells[k]) for k in rc2])
    fc2a = np.array([np.mean(shellsAbs[k]) for k in rc2])
    slope2 = loglog_slope(rc2, fc2)
    p_off, c_off, p_clean = fit_power_offset(rc2, fc2)
    slope2a = loglog_slope(rc2, fc2a)
    print("   r/d:        " + " ".join(f"{v:7.2f}" for v in rc2), flush=True)
    print("   <du2> (моно):" + " ".join(f"{v:+7.1e}" for v in fc2), flush=True)
    print("   <|du2|>:     " + " ".join(f"{v:7.1e}" for v in fc2a), flush=True)
    frac = np.abs(fc2) / np.maximum(fc2a, 1e-300)
    print("   монопольная доля |<.>|/<|.|>: " + " ".join(f"{v:.2f}" for v in frac), flush=True)
    print(f"   => монополь delta<u^2>: сырой ~r^({slope2:.2f}); с вычетом фона c={c_off:+.1e} -> ~r^(-{p_clean:.2f}); |.|~r^({slope2a:.2f})", flush=True)

    np.savez(out, N=N, seed=seed, dphi=dphi, d=d, L=L, dz0=z0 - Z_C, npos=npos,
             rcG=rcG, fcG=fcG, slopeG=slopeG,
             rc2=rc2, fc2=fc2, fc2a=fc2a, slope2=slope2, slope2a=slope2a,
             p_off=p_off, c_off=c_off, p_clean=p_clean)
    print(f"\n  СВОДКА N={N}: G(r)~r^{slopeG:.2f}; монополь delta<u^2>~r^{slope2:.2f} (фон-вычет ~r^-{p_clean:.2f}; |.|~r^{slope2a:.2f}).", flush=True)
    print(f"  Для 1/r-потенциала (а′) нужен монополь ~r^-1; r^-2 = G^2 = обструкция наравне со статикой.  [{time.time()-t0:.0f} с]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dphi", type=float, default=3e-2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, f"entropy_profile_N{args.N}_s{args.seed}.npz")
    run(args.N, args.seed, args.dphi, out)


if __name__ == "__main__":
    main()
