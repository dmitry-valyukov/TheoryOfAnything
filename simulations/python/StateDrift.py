"""Дрейф состояния спаянной упаковки (вариант C, 1.11) как кандидат на космологический дрейф (T-45).

Постановка. В картине главы 11 красное смещение — дрейф безразмерного состояния среды; сама по себе
самоподобная перемасштабировка (все a, mu, c вместе) изнутри невидима. Здесь считается, ЧТО именно
и НАСКОЛЬКО меняется в безразмерном состоянии спаянной заклиненной упаковки под двумя из трёх
двигателей главы 11:

  * двигатель 3 — медленное сжатие внешним давлением: phi -> phi (1+eps), eps = -3 % … +6 %,
    контакты не размыкаются (двусторонние пружины с длиной покоя sigma_ij), новые контакты
    замыкаются и тоже становятся связями (класс BondedNet). На каждом шаге: p, z, mu0 (гессиан,
    6 сдвигов), B (гессиан, дилатация, с поправкой +2p/3 = термодинамическое B = phi dp/dphi;
    в eps = 0 сверяется с конечной разностью), mu/B, и производные величины
    ln(mu a^3), ln(c/a) при rho ∝ a^{-3}: mu a^3 ∝ mu/phi, c/a ∝ sqrt(mu/phi) phi^{1/3};
  * двигатель 1 — уход дрожания: гауссово дрожание по тепловым модам амплитуды u (СКО смещения
    в единицах диаметра), закалённые модули mu_Q(u), B_Q(u) (гессиан в смещённой конфигурации,
    пары ±xi), порог Линдемана u_L из 1 - mu/mu0 = (u/u_L)^2; считается в eps = 0 и в двух
    точках сжатия, чтобы видеть дрейф самого порога.

Что НЕ считается: энергия узла (нужна модель солитона), поэтому знак и величина дрейфа f = a/lambda_C
отсюда не следуют; выдаётся дрейф безразмерных параметров состояния на единицу деформации/дрожания.

Запуск: python StateDrift.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2] [--out ...]
        python StateDrift.py --analyse state_drift_N512.npz
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

EPS_GRID = np.round(np.arange(-0.03, 0.0601, 0.005), 4)
U_GRID = np.geomspace(1e-3, 1e-1, 9)
EPS_JITTER = (0.0, 0.02, 0.05)
M_JIT = 3


class BondedNet(PinnedSpheres):
    """Спаянная сеть: связи (i,j) — двусторонние гармонические пружины с длиной покоя sigma_ij,
    остальные пары — обычный односторонний контакт. Список соседей — обычный (KD-дерево) плюс
    все связи (на случай растяжения за радиус списка)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.bond_keys = np.zeros(0, np.int64)
        self.allow_new = True      # False: активны только связи (новые контакты не считаются)

    def bond(self, i, j):
        i, j = np.minimum(i, j), np.maximum(i, j)
        self.bond_keys = np.union1d(self.bond_keys, i.astype(np.int64) * self.N + j)
        self._pairs = None

    def build_pairs(self):
        super().build_pairs()
        keys = self._i.astype(np.int64) * self.N + self._j
        miss = np.setdiff1d(self.bond_keys, keys)
        if len(miss):
            keys = np.concatenate([keys, miss])
            order = np.argsort(keys); keys = keys[order]
        self._i, self._j = (keys // self.N).astype(int), (keys % self.N).astype(int)
        self._pairs = np.stack([self._i, self._j], 1)
        self._sig = self.rad[self._i] + self.rad[self._j]
        self._bonded = np.isin(keys, self.bond_keys)

    def _active(self, s):
        ds, dx, r = self._geom(s)
        ov = 1.0 - r / self._sig
        m = ((ov > 0) | self._bonded) if self.allow_new else self._bonded.copy()
        return m, dx, r, ov

    def energy_grad(self, sflat):
        s = sflat.reshape(self.N, 3)
        m, dx, r, ov = self._active(s)
        E = 0.5 * (ov[m] ** 2).sum()
        fac = np.zeros_like(r)
        fac[m] = -ov[m] / self._sig[m] / r[m]
        f = fac[:, None] * dx
        gx = np.stack([np.bincount(self._j, f[:, c], self.N) - np.bincount(self._i, f[:, c], self.N)
                       for c in range(3)], axis=1)
        gs = gx @ self.F()
        return E, gs.ravel()

    def force_scale(self):
        if self.need_rebuild():
            self.build_pairs()
        m, dx, r, ov = self._active(self.s)
        return float((np.abs(ov[m]) / self._sig[m]).mean()) + 1e-300

    def contacts(self):
        """Активные пары (связи + замкнутые контакты) — так base.stress_pressure/coordination
        видят и растянутые связи."""
        if self.need_rebuild():
            self.build_pairs()
        m, dx, r, ov = self._active(self.s)
        return self._i[m], self._j[m], dx[m], r[m], ov[m]

    def bond_all_active(self):
        i, j, *_ = self.contacts()
        n0 = len(self.bond_keys)
        self.bond(i, j)
        return len(self.bond_keys) - n0


# ---------------------------------------------------------------------------
def assemble_full(p):
    """Гессиан по активным парам (гармонический, двусторонний), Xi и аффинные части для
    6 сдвигов и для дилатации. Возвращает H, Xis(7), aff(7), V, p."""
    i, j, dx, r, ov = p.contacts()
    sig = p.rad[i] + p.rad[j]
    dVdr = -ov / sig
    d2V = 1.0 / sig ** 2
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
    Xis, aff = [], []
    g = dVdr[:, None] * n
    strains = [np.eye(3)[[a_]].T @ np.eye(3)[[b_]] for (a_, b_) in SHEARS] + [np.eye(3)]
    for eps in strains:
        e = dx @ eps.T                     # e = eps dx
        xi_pair = np.einsum('kab,kb->ka', K, e)
        Xi = np.zeros((N, 3))
        np.add.at(Xi, j, xi_pair); np.add.at(Xi, i, -xi_pair)
        Xis.append(Xi.ravel())
        aff.append(((xi_pair * e).sum() + (g * (e @ eps.T)).sum()) / V)
    pr = -(dVdr * r).sum() / (3 * V)
    return H, np.array(Xis), np.array(aff), V, pr


def moduli_full(p):
    """(mu_6 [сдвиги], B) по гессиану. B термодинамический: (aff_vol - Xi H^-1 Xi/V)/9 + 2p/3."""
    H, Xis, aff, V, pr = assemble_full(p)
    Hr = H.copy()
    lam = 1e-9 * np.trace(Hr) / Hr.shape[0]
    Hr[np.diag_indices_from(Hr)] += lam
    try:
        cf = cho_factor(Hr, check_finite=False)
        sol = lambda b: cho_solve(cf, b, check_finite=False)
    except np.linalg.LinAlgError:
        lu = lu_factor(Hr, check_finite=False)
        sol = lambda b: lu_solve(lu, b, check_finite=False)
    vals = np.array([aff[k] - Xis[k] @ sol(Xis[k]) / V for k in range(7)])
    mus = vals[:6]
    # aff_vol содержит член g·e = sum V' r = -3 p V -> vals[6] = (sum V'' r^2 - Xi H^-1 Xi - 3pV)/V ;
    # термодинамическое B = (sum V'' r^2 - Xi H^-1 Xi)/(9V) + 2p/3 = (vals[6] + 3p)/9 + 2p/3
    B = (vals[6] + 3 * pr) / 9.0 + 2 * pr / 3.0
    return mus, B, pr, H


def bulk_fd(p, rel=1e-3):
    s0, L0, phi0 = p.s.copy(), p.L, p.phi
    ps = []
    for f in (1 - rel, 1 + rel):
        p.s, p.L = s0.copy(), L0
        p.set_phi(phi0 * f); p._pairs = None; p.build_pairs()
        p.minimize(rel=1e-9)
        ps.append(p.stress_pressure()[1])
    p.s, p.L = s0.copy(), L0; p._pairs = None; p.build_pairs()
    return (ps[1] - ps[0]) / (2 * rel)


def thermal_basis_from_H(H):
    w, v = np.linalg.eigh(H)
    keep = w > 1e-8 * w.max()
    return v[:, keep] / np.sqrt(w[keep])[None, :]


def jitter_sweep(p, rng, M=M_JIT):
    """Закалённые mu_Q(u), B_Q(u) в смещённых конфигурациях s0 ± xi (тепловые моды).
    Два варианта: только связи (новые контакты при дрожании не считаются — как FrozenNet в 7.7)
    и связи + замкнувшиеся при смещении новые контакты (канал «старения»). out[..., 0:2] / [..., 2:4]."""
    s0 = p.s.copy()
    mus0, B0, pr0, H = moduli_full(p)
    Bas = thermal_basis_from_H(H)
    mu0 = mus0.mean()
    out = np.zeros((len(U_GRID), 4))
    for ku, u in enumerate(U_GRID):
        acc = []
        for m in range(M):
            eta = rng.standard_normal(Bas.shape[1])
            xi = (Bas @ eta).reshape(p.N, 3)
            xi *= u / math.sqrt((xi ** 2).sum(axis=1).mean())
            for sgn in (+1, -1):
                p.s = s0 + sgn * xi; p._pairs = None; p.build_pairs()
                p.allow_new = False
                musF, BF, _, _ = moduli_full(p)
                p.allow_new = True
                musN, BN, _, _ = moduli_full(p)
                acc.append((musF.mean(), BF, musN.mean(), BN))
        acc = np.array(acc)
        out[ku] = acc.mean(axis=0)
    p.s = s0.copy(); p._pairs = None; p.build_pairs()
    return mu0, B0, out


def lindemann(mu0, muQ):
    d = 1 - muQ / mu0
    m = (d > 0.005) & (d < 0.3)
    if m.sum() < 2:
        return np.nan, np.nan
    sl, ic = np.polyfit(np.log(U_GRID[m]), np.log(d[m]), 1)
    uL = math.exp(-ic / 2)  # если d = (u/uL)^2
    return sl, uL


def run_sample(N, seed, dphis, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed} ---", flush=True)
    p0, phi_j = find_phi_j(N, seed, False, verbose=verbose)
    rng = np.random.default_rng(3000 + seed)
    out = []
    for dphi in dphis:
        p = PinnedSpheres(N, 0.65, seed)
        p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
        p.s, p.L = p0.s.copy(), p0.L
        p.G3[:] = 0.0
        p.set_phi(phi_j + dphi)
        p.stabilize()
        z0, _, _ = p.coordination(); _, pr0 = p.stress_pressure()
        i0, j0, *_ = p.contacts()
        # спаянная сеть от этого состояния
        b = BondedNet(N, 0.65, seed); b.rad = p.rad.copy(); b.vol_particles = p.vol_particles
        b.rc = p.rc; b.L = p.L; b.s = p.s.copy(); b.G3 = p.G3.copy(); b.bond(i0, j0)
        b.minimize(rel=1e-10)
        s_ref, L_ref, phi_ref = b.s.copy(), b.L, b.phi
        keys_ref = b.bond_keys.copy()
        # сверка B: гессиан против конечной разности в eps=0
        mus, B_H, pr, _ = moduli_full(b)
        B_fd = bulk_fd(b)
        print(f"    dphi={dphi:g} dz={z0-Z_C:+.3f} p={pr:.3e} mu0={mus.mean():.4e} "
              f"B_H={B_H:.4e} B_fd={B_fd:.4e} (отн. {B_H/B_fd-1:+.1e}) mu/B={mus.mean()/B_H:.4f}  [{time.time()-t0:.0f} с]", flush=True)
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, dz0=z0 - Z_C, p0=pr, B_fd0=B_fd,
                   eps=EPS_GRID, phi=np.zeros(len(EPS_GRID)), p=np.zeros(len(EPS_GRID)),
                   z=np.zeros(len(EPS_GRID)), mu=np.zeros(len(EPS_GRID)), mu6=np.zeros((len(EPS_GRID), 6)),
                   B=np.zeros(len(EPS_GRID)), nb=np.zeros(len(EPS_GRID), int), fneg=np.zeros(len(EPS_GRID)),
                   jit={})
        # ход по сжатию: от eps=0 вверх, затем от eps=0 вниз (связи набираются по пути)
        for branch in (EPS_GRID[EPS_GRID >= 0], EPS_GRID[EPS_GRID < 0][::-1]):
            b.s, b.L = s_ref.copy(), L_ref; b.bond_keys = keys_ref.copy(); b._pairs = None
            for e in branch:
                b.set_phi(phi_ref * (1 + e)); b._pairs = None; b.build_pairs()
                b.minimize(rel=1e-10)
                b.bond_all_active()
                k = int(np.where(EPS_GRID == e)[0][0])
                mus, B, pr, _ = moduli_full(b)
                z, _, _ = b.coordination()
                _, _, _, _, ov = b.contacts()
                rec["phi"][k] = b.phi; rec["p"][k] = pr; rec["z"][k] = z; rec["mu"][k] = mus.mean()
                rec["mu6"][k] = mus; rec["B"][k] = B; rec["nb"][k] = len(b.bond_keys); rec["fneg"][k] = (ov < 0).mean()
                print(f"      eps={e:+.3f} phi={b.phi:.4f} p={pr:+.3e} z={z:.3f} mu={mus.mean():.4e} B={B:.4e} "
                      f"mu/B={mus.mean()/B:.4f} связей={len(b.bond_keys)} растянутых={(ov<0).mean():.2f}  [{time.time()-t0:.0f} с]", flush=True)
                if e in EPS_JITTER:
                    mu0j, B0j, jq = jitter_sweep(b, rng)
                    sl, uL = lindemann(mu0j, jq[:, 0])
                    rec["jit"][float(e)] = dict(mu0=mu0j, B0=B0j, muQ=jq[:, 0], BQ=jq[:, 1], muN=jq[:, 2], BN=jq[:, 3], nu=sl, uL=uL)
                    print(f"        дрожание при eps={e:+.3f}: nu={sl:.2f} u_L={uL:.3g}; mu_Q/mu0 (связи): " +
                          " ".join(f"{x:.3f}" for x in jq[:, 0] / mu0j) + " | B_Q/B0: " +
                          " ".join(f"{x:.3f}" for x in jq[:, 1] / B0j) + " | mu_N/mu0 (с новыми): " +
                          " ".join(f"{x:.3f}" for x in jq[:, 2] / mu0j) + f"  [{time.time()-t0:.0f} с]", flush=True)
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: дрейф состояния спаянной упаковки (T-45)")
    dphis = sorted(set(r["dphi"] for r in recs))
    print(f"  образцов: {len(recs)}; N={sorted(set(r['N'] for r in recs))}; dphi={dphis}")
    print(f"  сверка B (гессиан/конечная разность - 1) в eps=0: " +
          ", ".join(f"{np.interp(0, r['eps'], r['B'])/r['B_fd0']-1:+.1e}" for r in recs))
    eps = recs[0]["eps"]
    for d in dphis:
        rr = [r for r in recs if r["dphi"] == d]
        dz = np.mean([r["dz0"] for r in rr])
        k0 = int(np.where(eps == 0)[0][0])
        MU = np.array([r["mu"] / r["mu"][k0] for r in rr]); B = np.array([r["B"] / r["B"][k0] for r in rr])
        PHI = np.array([r["phi"] / r["phi"][k0] for r in rr]); Z = np.array([r["z"] for r in rr])
        P = np.array([r["p"] for r in rr]); MB = np.array([r["mu"] / r["B"] for r in rr])
        lmu, lB, lphi = np.log(MU).mean(0), np.log(B).mean(0), np.log(PHI).mean(0)
        l_mua3 = lmu - lphi                       # mu a^3 ∝ mu/phi
        l_ca = 0.5 * (lmu - lphi) + lphi / 3       # c/a ∝ sqrt(mu/phi) phi^{1/3}
        l_mB = lmu - lB
        print(f"\n  == dphi={d:g} (dz0={dz:.3f}) — двигатель 3, сжатие; средние по {len(rr)} образцам ==")
        print(f"  {'eps':>7} {'p':>10} {'z':>6} {'mu/mu0':>8} {'B/B0':>8} {'mu/B':>8} {'ln(mu a^3)':>11} {'ln(c/a)':>8} {'ln(mu/B)':>9}")
        for k in range(len(eps)):
            print(f"  {eps[k]:>+7.3f} {P.mean(0)[k]:>10.3e} {Z.mean(0)[k]:>6.3f} {MU.mean(0)[k]:>8.4f} {B.mean(0)[k]:>8.4f} "
                  f"{MB.mean(0)[k]:>8.4f} {l_mua3[k]:>+11.4f} {l_ca[k]:>+8.4f} {l_mB[k]:>+9.4f}")
        # односторонние производные по eps у нуля: сжатие (0…+0.02) и растяжение (-0.02…0)
        for lab, m in (("сжатие 0…+2 %", (eps >= -1e-9) & (eps <= 0.0201)), ("растяжение -2 %…0", (eps >= -0.0201) & (eps <= 1e-9))):
            def slope(y):
                return np.polyfit(eps[m], y[m], 1)[0]
            print(f"  производные у eps=0 ({lab}), на единицу eps=dphi/phi:  d ln mu = {slope(lmu):+.2f}, d ln B = {slope(lB):+.2f}, "
                  f"d ln(mu a^3) = {slope(l_mua3):+.2f}, d ln(c/a) = {slope(l_ca):+.2f}, d ln(mu/B) = {slope(l_mB):+.2f}, dz/deps = {slope(Z.mean(0)):+.1f}")
        for e in EPS_JITTER:
            js = [r["jit"][e] for r in rr if e in r["jit"]]
            if not js:
                continue
            nu = np.nanmean([j["nu"] for j in js]); uL = np.nanmean([j["uL"] for j in js])
            muQ = np.mean([j["muQ"] / j["mu0"] for j in js], axis=0); BQ = np.mean([j["BQ"] / j["B0"] for j in js], axis=0)
            muN = np.mean([j["muN"] / j["mu0"] for j in js], axis=0); BN = np.mean([j["BN"] / j["B0"] for j in js], axis=0)
            print(f"  двигатель 1 (дрожание) при eps={e:+.3f}: только связи: nu={nu:.2f}, u_L={uL:.3g}")
            print("      u:                 " + " ".join(f"{x:7.1e}" for x in U_GRID))
            print("      mu_Q/mu0 (связи):  " + " ".join(f"{x:7.3f}" for x in muQ))
            print("      B_Q/B0   (связи):  " + " ".join(f"{x:7.3f}" for x in BQ))
            print("      (mu/B)_Q/(mu/B)_0: " + " ".join(f"{x:7.3f}" for x in muQ / BQ))
            print("      mu_N/mu0 (с новыми контактами): " + " ".join(f"{x:7.3f}" for x in muN))
            print("      B_N/B0   (с новыми контактами): " + " ".join(f"{x:7.3f}" for x in BN))
    print("""
  Чтение. Самоподобная часть (все a, mu, c вместе) изнутри невидима; наблюдаемо только изменение
  безразмерного состояния (mu/B, z, u_L, ...). Таблица говорит, насколько состояние движется на единицу
  сжатия и на единицу дрожания; перевод в дрейф f = a/lambda_C требует модели узла и здесь не делается.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2")
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
    head(f"ДРЕЙФ СОСТОЯНИЯ спаянной упаковки: N={args.N}, затравки {seeds}, dphi={dphis}")
    out = args.out or os.path.join(HERE, f"state_drift_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
