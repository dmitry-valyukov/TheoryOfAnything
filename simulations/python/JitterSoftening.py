"""T-4, звено 2б: отклик гессианова модуля сдвига заклиненного образца на ДРОЖАНИЕ.

Постановка. Механизм B гравитации (7.4, 7.6): узел накачивает дрожание среды, оно
переносится диффузионно, и избыток <u^2> размягчает модуль сдвига по закону
    mu_eff = mu0 (1 - <u^2>/<u^2>_L)              (линдемановский порог <u^2>_L).
ОТО требует линейности delta mu по энергии, т.е. по <u^2> (аналитичность). Здесь
проверяется первое звено: как гессианов модуль заклиненного образца зависит от
дисперсии наложенного дрожания — линейно ли по <u^2>, с каким наклоном, где порог.

Образец. Генератор JammedSoftening.py (мягкие сферы 1:1.4, гармонический или
герцевский контакт), сдвиговая стабилизация (GravitySoftening.PinnedSpheres).

Дрожание. К равновесным координатам x0 добавляются случайные смещения xi с
заданной среднеквадратичной амплитудой u = sqrt(<|xi_i|^2>):
  «белое»    — независимые гауссовы смещения каждой частицы;
  «тепловое» — суперпозиция собственных мод гессиана с амплитудами ~ 1/omega
               (равнораспределение), затем нормировка на тот же u.
Три модуля на каждой конфигурации x0 +- xi (пары +xi/-xi, чтобы первый порядок по xi сократился
точно; среднее по M реализациям и по 6 сдвигам ab):
  mu_Q  — «замороженный» (quenched): гессианов модуль мгновенной конфигурации с
          ТЕКУЩИМ набором контактов (контакты размыкаются/замыкаются) — так видит среду
          волна, быстрая по сравнению с дрожанием;
  mu_F  — тот же модуль, но с набором контактов ИСХОДНОГО состояния (контакт остаётся
          двусторонней пружиной при отрицательном перекрытии) — гладкая, аналитическая
          часть отклика (только геометрия и преднапряжение);
  mu_A  — «отожжённый» (annealed): модуль по усреднённым H, Xi, mu_aff — так видит
          среду волна, медленная по сравнению с дрожанием.
Наблюдаемое: 1 - mu(u)/mu0 как функция u; показатель nu в 1 - mu/mu0 ~ u^nu при малых u
(nu = 2 — аналитичность, закон Линдемана; nu = 1 — размыкание контактов, |u|-линейно);
масштаб u_x, на котором теряется 10 % и 50 %; порог <u^2>_L = u^2/(1 - mu/mu0) на участке nu = 2,
если он есть; зависимость от dz.

Запуск:  python JitterSoftening.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2,1e-1]
                                   [--M 6] [--hertz] [--out jitter_softening_N512.npz]
         python JitterSoftening.py --analyse jitter_softening_N512.npz
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres, SHEARS  # noqa: E402

U_GRID = np.geomspace(1e-5, 1e-1, 9)


# ---------------------------------------------------------------------------
def assemble(p, contacts=None):
    """Матрица Гессе (все N частиц, регуляризованная), Xi для 6 сдвигов, аффинные модули.
    contacts=None — текущие контакты (перекрытие > 0); иначе (i, j) заданного набора,
    контакт считается двусторонней пружиной (перекрытие может быть < 0; только гармонический)."""
    if contacts is None:
        i, j, dx, r, ov = p.contacts()
    else:
        i, j = contacts
        s = np.mod(p.s, p.L)
        ds = s[j] - s[i]
        ds -= p.L * np.round(ds / p.L)
        dx = ds @ p.F().T
        r = np.linalg.norm(dx, axis=1)
        ov = 1.0 - r / (p.rad[i] + p.rad[j])
    sig = p.rad[i] + p.rad[j]
    a = p.alpha
    if contacts is None or a == 2.0:
        dVdr = -(np.abs(ov) ** (a - 1)) * np.sign(ov) / sig if a != 2.0 else -ov / sig
        d2V = (a - 1) * np.abs(ov) ** (a - 2) / sig ** 2 if a != 2.0 else 1.0 / sig ** 2
    else:
        raise ValueError("замороженный набор контактов — только для гармонического потенциала")
    V = p.L ** 3
    n = dx / r[:, None]
    N = p.N
    K = (d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :] \
        + (dVdr / r)[:, None, None] * np.eye(3)[None]
    H = np.zeros((3 * N, 3 * N))
    for a_ in range(3):
        for b_ in range(3):
            np.add.at(H, (3 * i + a_, 3 * i + b_), K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * j + b_), K[:, a_, b_])
            np.add.at(H, (3 * i + a_, 3 * j + b_), -K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * i + b_), -K[:, a_, b_])
    Xis, muaff = [], []
    g = dVdr[:, None] * n
    for (a_, b_) in SHEARS:
        e = np.zeros_like(dx); e[:, a_] = dx[:, b_]
        xi_pair = np.einsum('kab,kb->ka', K, e)
        Xi = np.zeros((N, 3))
        np.add.at(Xi, j, xi_pair); np.add.at(Xi, i, -xi_pair)
        Xis.append(Xi.ravel())
        Ke = xi_pair
        muaff.append(((Ke[:, a_] * dx[:, b_]) + g[:, a_] * e[:, b_]).sum() / V)
    return H, np.array(Xis), np.array(muaff), V


def solve_moduli(H, Xis, muaff, V):
    """mu_ab = mu_aff - Xi^T H^{-1} Xi / V для 6 сдвигов; трансляции снимаются регуляризацией."""
    Hr = H.copy()
    lam = 1e-9 * np.trace(Hr) / Hr.shape[0]
    Hr[np.diag_indices_from(Hr)] += lam
    try:
        cf = cho_factor(Hr, check_finite=False)
        sol = lambda b: cho_solve(cf, b, check_finite=False)
    except np.linalg.LinAlgError:
        lu = lu_factor(Hr, check_finite=False)
        sol = lambda b: lu_solve(lu, b, check_finite=False)
    mus = np.array([muaff[k] - Xis[k] @ sol(Xis[k]) / V for k in range(len(Xis))])
    return mus


def moduli(p, contacts=None):
    return solve_moduli(*assemble(p, contacts))


# ---------------------------------------------------------------------------
def thermal_basis(p):
    """Собственные моды гессиана равновесия (для «теплового» дрожания): возвращает
    матрицу базиса B (3N x K) со столбцами e_k/omega_k (без трансляций и мод ~0)."""
    H, *_ = assemble(p)
    w, v = np.linalg.eigh(H)
    keep = w > 1e-8 * w.max()
    return v[:, keep] / np.sqrt(w[keep])[None, :]


def jitter(rng, p, u, kind, B=None):
    N = p.N
    if kind == "white":
        xi = rng.standard_normal((N, 3))
    else:
        eta = rng.standard_normal(B.shape[1])
        xi = (B @ eta).reshape(N, 3)
    xi *= u / math.sqrt((xi ** 2).sum(axis=1).mean())
    return xi


def run_sample(N, seed, dphis, M, hertz, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed}, {'Герц' if hertz else 'гармонический'} ---", flush=True)
    p0, phi_j = find_phi_j(N, seed, hertz, verbose=verbose)
    p = PinnedSpheres(N, 0.65, seed, hertz=hertz)
    p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
    p.s, p.L = p0.s.copy(), p0.L
    s_j, L_j = p.s.copy(), p.L
    rng = np.random.default_rng(2000 + seed)
    out = []
    for dphi in dphis:
        p.s, p.L = s_j.copy(), L_j
        p.G3[:] = 0.0
        p.set_phi(phi_j + dphi)
        p.stabilize()
        s0 = p.s.copy()
        z, nrat, nc = p.coordination()
        _, pr = p.stress_pressure()
        i0, j0, dx0, r0, ov0 = p.contacts()
        cs0 = (i0.copy(), j0.copy())
        mu0_dir = moduli(p)
        mu0 = mu0_dir.mean()
        ov_med = float(np.median(ov0)); ov_mean = float(ov0.mean())
        Bth = thermal_basis(p)
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, p=pr, z=z, dz=z - Z_C, nrat=nrat,
                   mu0=mu0, mu0_dir=mu0_dir, L=p.L, a=(p.L ** 3 / N) ** (1 / 3), ov_med=ov_med,
                   ov_mean=ov_mean, hertz=hertz, u=U_GRID, kinds=["white", "thermal"], data={})
        print(f"    dphi={dphi:<6g} p={pr:.3e} z={z:.3f} dz={z-Z_C:+.3f} mu0={mu0:.4e} "
              f"перекрытие медиана {ov_med:.2e}  [{time.time()-t0:.0f} с]", flush=True)
        for kind in rec["kinds"]:
            muQ = np.zeros((len(U_GRID), M)); muF = np.full((len(U_GRID), M), np.nan)
            muA = np.zeros(len(U_GRID)); dnc = np.zeros((len(U_GRID), M))
            for ku, u in enumerate(U_GRID):
                Hs = 0; Xs = 0; As = 0
                for m in range(M):
                    xi = jitter(rng, p, u, kind, Bth)
                    q = 0.0; f = 0.0; dn = 0.0
                    for sgn in (+1, -1):                     # пара +xi/-xi: первый порядок сокращается точно
                        p.s = s0 + sgn * xi; p._pairs = None
                        H, Xis, muaff, V = assemble(p)
                        q += 0.5 * solve_moduli(H, Xis, muaff, V).mean()
                        Hs = Hs + 0.5 * H; Xs = Xs + 0.5 * Xis; As = As + 0.5 * muaff
                        if not hertz:
                            f += 0.5 * moduli(p, cs0).mean()
                        i1, j1, *_ = p.contacts()
                        dn += 0.5 * len(set(zip(i0.tolist(), j0.tolist())) ^ set(zip(i1.tolist(), j1.tolist())))
                    muQ[ku, m] = q; dnc[ku, m] = dn
                    if not hertz:
                        muF[ku, m] = f
                muA[ku] = solve_moduli(Hs / M, Xs / M, As / M, V).mean()
                p.s = s0.copy(); p._pairs = None
            rec["data"][kind] = dict(muQ=muQ, muF=muF, muA=muA, dnc=dnc)
            print(f"      {kind:8s}: 1-muQ/mu0 при u=1e-4,1e-3,1e-2: " +
                  ", ".join(f"{1-muQ[k].mean()/mu0:+.3f}" for k in (2, 4, 6)) +
                  f";  1-muF/mu0: " + ", ".join(f"{1-np.nanmean(muF[k])/mu0:+.3f}" for k in (2, 4, 6)) +
                  f";  1-muA/mu0: " + ", ".join(f"{1-muA[k]/mu0:+.3f}" for k in (2, 4, 6)) +
                  f"  [{time.time()-t0:.0f} с]", flush=True)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def slope(x, y):
    m = (x > 0) & (y > 0)
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0] if m.sum() >= 3 else np.nan


def crossover(u, d, level):
    idx = np.where(d >= level)[0]
    if len(idx) == 0:
        return np.nan
    k = idx[0]
    if k == 0:
        return u[0]
    x0, x1 = math.log(u[k - 1]), math.log(u[k]); y0, y1 = d[k - 1], d[k]
    return math.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def analyse(recs):
    head("СВОДКА T-4/2б: отклик модуля на дрожание")
    dphis = sorted(set(r["dphi"] for r in recs)); Ns = sorted(set(r["N"] for r in recs))
    hz = sorted(set(bool(r["hertz"]) for r in recs))
    print(f"  образцов: {len(recs)}; N={Ns}; dphi={dphis}; контакт: {['Герц' if h else 'гарм' for h in hz]}")
    print("  d = 1 - mu/mu0 (среднее по образцам и реализациям); nu — показатель d ~ u^nu на участке 0.01 < d < 0.3;")
    print("  u_10, u_50 — амплитуда потери 10 и 50 %; <u^2>_L = u^2/d при малых u (имеет смысл только при nu = 2).\n")
    u = recs[0]["u"]
    for h in hz:
        for N in Ns:
            for d_ in dphis:
                rr = [r for r in recs if r["N"] == N and r["dphi"] == d_ and bool(r["hertz"]) == h]
                if not rr:
                    continue
                dz = np.mean([r["dz"] for r in rr]); mu0 = np.mean([r["mu0"] for r in rr])
                ovm = np.mean([r["ov_med"] for r in rr]); a = np.mean([r["a"] for r in rr])
                print(f"  ==== {'Герц' if h else 'гарм'} N={N} dphi={d_:g}: dz={dz:.3f} mu0={mu0:.3e} медианное перекрытие {ovm:.2e}, образцов {len(rr)} ====")
                print("       u:        " + " ".join(f"{x:8.1e}" for x in u))
                for kind in ("white", "thermal"):
                    for name, stat in (("muQ", "mean"), ("muQ", "median"), ("muF", "mean"), ("muF", "median"), ("muA", "mean"), ("muA", "median")):
                        vals = []
                        for r in rr:
                            arr = r["data"][kind][name]
                            v = arr if arr.ndim == 2 else arr[:, None]
                            vals.append(1 - v / r["mu0"])
                        allv = np.concatenate(vals, axis=1)          # (n_u, реализации × образцы)
                        if np.all(np.isnan(allv)):
                            continue
                        dd = np.nanmean(allv, axis=1) if stat == "mean" else np.nanmedian(allv, axis=1)
                        name = f"{name}/{stat[:3]}"
                        msk = (dd > 0.01) & (dd < 0.3)
                        nu = slope(u[msk], dd[msk]) if msk.sum() >= 3 else np.nan
                        # локальный показатель по первым точкам
                        nu_lo = slope(u[:4], np.abs(dd[:4]))
                        u10 = crossover(u, dd, 0.1); u50 = crossover(u, dd, 0.5)
                        uL2 = u ** 2 / np.where(dd > 0, dd, np.nan)
                        print(f"    {kind:7s} {name:8s}: d = " + " ".join(f"{x:+8.4f}" for x in dd)
                              + f"   nu={nu:5.2f} (первые точки {nu_lo:5.2f})  u_10={u10:.1e} u_50={u50:.1e}"
                              + f"  <u^2>_L/a^2 (u<=1e-4)={np.nanmean(uL2[:3])/a**2:.2e}")
                    dn = np.mean([np.mean(r["data"][kind]["dnc"], axis=1) for r in rr], axis=0)
                    print(f"    {kind:7s} изменённых контактов: " + " ".join(f"{x:8.1f}" for x in dn))
    print("""
  Чтение.
  * mu_Q (замороженный) — модуль мгновенной конфигурации с текущими контактами: включает
    размыкание контактов; nu = 1 при малых u означает |u|-линейное размягчение (доля
    разомкнутых контактов ~ u), nu = 2 — аналитический линдемановский закон.
  * mu_F (замороженный набор контактов) — чисто гладкая часть: геометрия и преднапряжение;
    здесь nu = 2 обязан выполняться, и его коэффициент — «аналитический» порог <u^2>_L.
  * mu_A (отожжённый) — среднее гессиана: так среду видит медленная волна.
  * ОТО (линейность по E) требует nu = 2 у ТОГО модуля, который видит свет; если nu = 1
    из-за размыкания контактов, механизм B наследует неаналитичность 1.10 и 7.6.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--hertz", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--analyse", type=str, default="")
    args = ap.parse_args()
    if args.analyse:
        recs = []
        for f in args.analyse.split(","):
            recs += list(np.load(f, allow_pickle=True)["recs"])
        analyse(recs)
        return
    seeds = [int(s) for s in args.seeds.split(",")]
    dphis = [float(s) for s in args.dphi.split(",")]
    head(f"T-4/2б: дрожание на заклиненном образце, N={args.N}, затравки {seeds}, dphi={dphis}, M={args.M}, "
         f"{'Герц' if args.hertz else 'гармонический контакт'}")
    out = args.out or os.path.join(HERE, f"jitter_softening_N{args.N}{'_hertz' if args.hertz else ''}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.M, args.hertz)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
