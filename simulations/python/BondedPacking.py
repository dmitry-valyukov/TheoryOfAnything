"""Вариант C структуры среды: заклиненная упаковка с НЕРАЗМЫКАЮЩИМИСЯ контактами («спаянная»).

Постановка (развилка T-44 после 1.10, 7.6, 7.7). Три отклика заклиненной упаковки
мягких сфер — на сдвиг, на локальный источник, на дрожание — неаналитичны по одной
причине: односторонний контакт размыкается при сколь угодно малом возмущении. Гипотеза C:
среда — та же заклиненная упаковка (та же геометрия, то же сжимающее преднапряжение), но
контакты после заклинивания не размыкаются — связь двусторонняя (класс FrozenNet из
Birefringence.py: V = ½(1 - r/σ)² при любом знаке перекрытия). Линейный гессиан у C и у
односторонней упаковки один и тот же, поэтому все линейные свойства (mu0, спектр,
маргинальность) совпадают по построению; проверить надо то, что отличается — нелинейный
отклик:

  1. Несжимаемость: объёмный модуль B (по давлению при сжатии ±0.1 % с релаксацией)
     и отношение mu0/B на спаянной сети против односторонней; сохраняется ли mu0/B ∝ dz → 0.
     Плюс нелинейное сжатие/растяжение на ±1 %: остаётся ли B конечным.
  2. Закон под сдвигом: квазистатический сдвиг ячейки gamma = 1e-4 … 0.1 с полной релаксацией,
     секущий модуль mu_s(gamma)/mu0 = sigma/(mu0 gamma) и гессианов mu_H(gamma)/mu0;
     ожидание (SofteningLaw.py: сжимающее преднапряжение у изостатики даёт размягчение) —
     аналитический закон 1 - mu/mu0 = a gamma² с a > 0; измеряются a (= квартичный g'
     в единицах mu0), показатель nu на малых gamma, масштаб gamma_x потери 10 % и 50 %.
  Оба знака gamma усредняются.

Запуск:  python BondedPacking.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2,1e-1]
                                 [--out bonded_packing_N512.npz]
         python BondedPacking.py --analyse bonded_packing_N512.npz
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
from JammedSoftening import find_phi_j, head, Z_C, bulk_modulus  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from JitterSoftening import moduli  # noqa: E402
from Birefringence import FrozenNet  # noqa: E402

G_GRID = np.geomspace(1e-4, 1e-1, 13)


def frozen_stress_pressure(pf):
    ds, dx, r = pf._geom(pf.s)
    ov = 1.0 - r / pf._sig
    dVdr = -ov / pf._sig
    V = pf.L ** 3
    sxy = (dVdr / r * dx[:, 0] * dx[:, 1]).sum() / V
    p = -(dVdr * r).sum() / (3 * V)
    return sxy, p


def frozen_bulk(pf, rel=1e-3):
    s0, L0 = pf.s.copy(), pf.L
    phi0 = pf.phi
    ps = []
    for f in (1 - rel, 1 + rel):
        pf.s, pf.L = s0.copy(), L0
        pf.set_phi(phi0 * f); pf._pairs = None; pf.build_pairs()
        pf.minimize(rel=1e-9)
        ps.append(frozen_stress_pressure(pf)[1])
    pf.s, pf.L = s0.copy(), L0; pf._pairs = None; pf.build_pairs()
    return phi0 * (ps[1] - ps[0]) / (2 * rel * phi0), ps


def make_frozen(p, s0, G0, i0, j0):
    pf = FrozenNet(p.N, 0.65, 0); pf.rad = p.rad.copy(); pf.vol_particles = p.vol_particles
    pf.rc = p.rc; pf.L = p.L; pf.s = s0.copy(); pf.G3 = G0.copy(); pf.freeze(i0, j0)
    return pf


def run_sample(N, seed, dphis, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed} ---", flush=True)
    p0, phi_j = find_phi_j(N, seed, False, verbose=verbose)
    p = PinnedSpheres(N, 0.65, seed)
    p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
    p.s, p.L = p0.s.copy(), p0.L
    s_j, L_j = p.s.copy(), p.L
    out = []
    for dphi in dphis:
        p.s, p.L = s_j.copy(), L_j
        p.G3[:] = 0.0
        p.set_phi(phi_j + dphi)
        p.stabilize()
        s0 = p.s.copy(); G0 = p.G3.copy()
        z, nrat, nc = p.coordination(); _, pr = p.stress_pressure()
        i0, j0, *_ = p.contacts()
        mu0_dir = moduli(p); mu0 = mu0_dir.mean()
        B1 = bulk_modulus(p)                       # односторонняя, ±0.1 %
        p.s, p.L = s0.copy(), p.L; p._pairs = None
        # спаянная сеть
        pf = make_frozen(p, s0, G0, i0, j0)
        pf.minimize(rel=1e-10); sF0 = pf.s.copy()
        muF_dir = moduli(pf, (i0, j0)); muF = muF_dir.mean()
        muF_xy = muF_dir[0]                                # модуль именно того сдвига, что задаёт G3[0] (SHEARS[0]: dx_x += g dx_y)
        sxy0 = frozen_stress_pressure(pf)[0]
        BF, _ = frozen_bulk(pf)
        # нелинейное сжатие/растяжение спаянной сети: p(phi) на ±1 %, ±3 %
        pnl = {}
        for f in (0.97, 0.99, 1.01, 1.03):
            pf.s, pf.L = sF0.copy(), p.L; pf.set_phi(p.phi * f); pf._pairs = None; pf.build_pairs()
            pf.minimize(rel=1e-9); pnl[f] = frozen_stress_pressure(pf)[1]
        pf.s, pf.L = sF0.copy(), p.L; pf._pairs = None; pf.build_pairs()
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, p=pr, z=z, dz=z - Z_C, mu0=mu0, muF=muF, muF_xy=muF_xy,
                   B1=B1, BF=BF, pnl=pnl, g=G_GRID, sig=np.zeros((len(G_GRID), 2)),
                   muH=np.zeros((len(G_GRID), 2)), E=np.zeros((len(G_GRID), 2)))
        print(f"    dphi={dphi:<6g} dz={z-Z_C:+.3f} p={pr:.3e} mu0={mu0:.4e} (спаянная {muF:.4e}) "
              f"B: односторонняя {B1:.4e}, спаянная {BF:.4e}; mu/B = {mu0/B1:.4f} / {muF/BF:.4f}  [{time.time()-t0:.0f} с]", flush=True)
        print("      p(phi) спаянной при phi×0.97,0.99,1.01,1.03: " + " ".join(f"{pnl[f]:+.3e}" for f in (0.97, 0.99, 1.01, 1.03)) + f"  (p0={pr:.3e})", flush=True)
        # закон под сдвигом (спаянная сеть), оба знака
        E0 = pf.energy_grad(pf.s.ravel())[0]
        for kg, g in enumerate(G_GRID):
            for ks, sgn in enumerate((+1, -1)):
                pf.s = sF0.copy(); pf.G3 = G0.copy(); pf.G3[0] += sgn * g
                pf.minimize(rel=1e-10)
                sxy, _ = frozen_stress_pressure(pf)
                rec["sig"][kg, ks] = sgn * (sxy - sxy0)
                mm = moduli(pf, (i0, j0)); rec["muH"][kg, ks] = mm[0]
                rec["E"][kg, ks] = pf.energy_grad(pf.s.ravel())[0] - E0
            ms = rec["sig"][kg].mean() / g / muF_xy
            print(f"      gamma={g:.1e}: mu_s/mu0={ms:.5f}  mu_H/mu0={rec['muH'][kg].mean()/muF_xy:.5f}  "
                  f"1-mu_s/mu0={1-ms:+.2e}  (1-mu_s/mu0)/gamma^2={(1-ms)/g**2:+.3f}", flush=True)
        pf.s = sF0.copy(); pf.G3 = G0.copy()
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: вариант C — заклиненная упаковка с неразмыкающимися контактами")
    dphis = sorted(set(r["dphi"] for r in recs))
    print(f"  образцов: {len(recs)}; N={sorted(set(r['N'] for r in recs))}; dphi={dphis}\n")
    print("  1. Несжимаемость: mu0/B (односторонняя | спаянная), среднее по образцам")
    print(f"  {'dphi':>7} {'dz':>6} {'mu0/B одност.':>15} {'mu/B спаян.':>13} {'B одност.':>10} {'B спаян.':>10}  p(0.97) p(0.99) p0 p(1.01) p(1.03)")
    for d in dphis:
        rr = [r for r in recs if r["dphi"] == d]
        dz = np.mean([r["dz"] for r in rr])
        gb1 = np.mean([r["mu0"] / r["B1"] for r in rr]); gbF = np.mean([r["muF"] / r["BF"] for r in rr])
        B1 = np.mean([r["B1"] for r in rr]); BF = np.mean([r["BF"] for r in rr])
        pn = {f: np.mean([r["pnl"][f] for r in rr]) for f in (0.97, 0.99, 1.01, 1.03)}
        p0 = np.mean([r["p"] for r in rr])
        print(f"  {d:>7g} {dz:>6.3f} {gb1:>15.4f} {gbF:>13.4f} {B1:>10.4f} {BF:>10.4f}  "
              f"{pn[0.97]:+.2e} {pn[0.99]:+.2e} {p0:+.2e} {pn[1.01]:+.2e} {pn[1.03]:+.2e}")
    print("\n  2. Закон под сдвигом на спаянной сети (среднее по знакам и образцам): 1 - mu_s/mu0 и a = (1-mu_s/mu0)/gamma^2")
    g = recs[0]["g"]
    print("       gamma:      " + " ".join(f"{x:8.1e}" for x in g))
    for d in dphis:
        rr = [r for r in recs if r["dphi"] == d]
        dz = np.mean([r["dz"] for r in rr])
        MS = np.array([r["sig"].mean(axis=1) / g / r["muF_xy"] for r in rr])
        MH = np.array([r["muH"].mean(axis=1) / r["muF_xy"] for r in rr])
        d_s = 1 - MS.mean(axis=0); d_h = 1 - MH.mean(axis=0)
        a_s = d_s / g ** 2
        m = (d_s > 0.005) & (d_s < 0.3)
        nu = np.polyfit(np.log(g[m]), np.log(d_s[m]), 1)[0] if m.sum() >= 3 else np.nan
        def cross(dd, lv):
            idx = np.where(dd >= lv)[0]
            if len(idx) == 0: return np.nan
            k = idx[0]
            if k == 0: return g[0]
            x0, x1 = math.log(g[k-1]), math.log(g[k]); y0, y1 = dd[k-1], dd[k]
            return math.exp(x0 + (lv - y0) * (x1 - x0) / (y1 - y0))
        print(f"  == dphi={d:g} dz={dz:.3f} ==")
        print("       1-mu_s/mu0: " + " ".join(f"{x:+8.1e}" for x in d_s))
        print("       1-mu_H/mu0: " + " ".join(f"{x:+8.1e}" for x in d_h))
        print("       a=d/gamma^2:" + " ".join(f"{x:+8.2f}" for x in a_s))
        print(f"       nu (0.005<d<0.3) = {nu:.2f}; a на gamma<=1e-3: {a_s[:5].mean():+.2f} (медиана по образцам {np.median([np.mean(((1-r['sig'].mean(axis=1)/g/r['muF_xy'])/g**2)[:5]) for r in rr]):+.2f}); "
              f"gamma_10% = {cross(d_s,0.1):.2e}, gamma_50% = {cross(d_s,0.5):.2e}")
    print("""
  Чтение.
  * mu/B спаянной сети должно совпадать с односторонней (тот же линейный гессиан) и уходить в
    нуль линейно по dz — несжимаемость сохраняется; p(phi) на ±3 % показывает, что B остаётся конечным
    и при растяжении (связи держат).
  * Под сдвигом: a > 0 — размягчение (нужный знак для узла, T-34: g' = a mu0), nu = 2 — аналитичность
    (линейное окно), gamma_x — его ширина.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2,1e-1")
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
    head(f"ВАРИАНТ C: спаянная заклиненная упаковка, N={args.N}, затравки {seeds}, dphi={dphis}")
    out = args.out or os.path.join(HERE, f"bonded_packing_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
