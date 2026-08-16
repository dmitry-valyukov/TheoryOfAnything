"""T-4: коэффициент размягчения против ОТО на заклиненном образце (локальный статический источник).

Постановка. ОТО в единицах среды (G = c^3 a^2/hbar, hbar = rho a^4 c, mu = rho c^2) требует
для слабого поля источника с энергией E на расстоянии r
    delta c / c  = -2 (E / mu a^3) (a / r),      delta mu / mu = -4 (E / mu a^3) (a / r),
причём линейно по E (аналитично по амплитуде). Проверяется, даёт ли механизм
«размягчение заклиненной среды упругим полем» такой закон: на заклиненном образце
при конечном dz создаётся ЛОКАЛИЗОВАННОЕ СТАТИЧЕСКОЕ поле деформации с упругой
энергией E, и по разности гессиановых (линейных) модулей сдвига одного и того же
образца с полем и без измеряется delta mu(r) в сферических оболочках вокруг источника.

Образец. Тот же генератор, что в JammedSoftening.py (мягкие сферы, бидисперсная
смесь 1:1.4, гармонический контакт, бисекция phi_J, сжатие до phi_J + dphi).

Источники (оба знака, несколько амплитуд):
  D — центр дилатации: радиус одной частицы умножается на (1 + eps), образец
      релаксирует;  E = E1 - E0 (в сжатом образце E нечётна по eps: работа
      против давления, плюс чётная часть ~ eps^2);
  P — закреплённая пара (силовой диполь): две контактирующие частицы раздвигаются
      (сдвигаются) вдоль связи на +-delta/2 и держатся, остальные релаксируют;
      E = E1 - E0 ~ delta^2 (чётна по знаку). Опорный расчёт для P — та же
      пара закреплена при delta = 0.
  Контроль «eps = 0» (повторная релаксация без источника) даёт шумовой пол.

Локальный модуль. Глобальный простой сдвиг ab с полной неаффинной релаксацией
(u = -H^{-1} Xi, H — матрица Гессе по живым частицам, закреплённые узлы исключены);
вклад контакта c в d sigma_ab / d gamma:
    mu_c = [ (K_c d_c)_a dx_b + g_a d_c,b ] / V,   d_c = e_c + Delta u_c,  e_c = dx_b e_a,
где K_c — жёсткость пары, g = V'(r) n. Сумма по всем контактам = mu0 (гессианов
модуль образца, тождество проверяется). Среднее по шести упорядоченным парам ab.
Оболочка r (по серединам контактов от центра источника): mu(r) = sum_c mu_c / V_shell.
Разность delta mu(r) = mu(r; с полем) - mu(r; без поля).

Что сравнивается. Коэффициент
    C(r) = -[delta mu(r)/mu0] / [(E/mu0 a^3)(a/r)] = -delta mu(r) a^2 r / E,
a = (V/N)^{1/3} — шаг упаковки; ОТО: C = 4, не зависит от r и от E.
Отдельно чётная (по знаку источника, ~E) и нечётная части delta mu; показатель
убывания m в delta mu ~ r^{-m} (ОТО: m = 1; линейная упругость: m = 3 в первом
порядке по амплитуде, m = 6 во втором); зависимость от dz.

Запуск:  python GravitySoftening.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2,1e-1]
                                    [--nsrc 8] [--eps 1e-3,1e-2] [--out gravity_softening_N512.npz]
         python GravitySoftening.py --analyse gravity_softening_N512.npz
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
from JammedSoftening import SoftSpheres, find_phi_j, head, Z_C  # noqa: E402

SHEARS = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]


class PinnedSpheres(SoftSpheres):
    """SoftSpheres с набором закреплённых частиц (их градиент обнуляется) и с тремя
    сдвиговыми степенями свободы ячейки G3 = (g_xy, g_xz, g_yz): F = I + верхний
    треугольник. Образец «сдвигово стабилизируется» (Dagois-Bohy и др., 2012):
    энергия минимизируется и по форме ячейки, так что остаточные сдвиговые
    напряжения нулевые и все сдвиговые модули неотрицательны (иначе в малом
    образце при фиксированной кубической ячейке часть направлений даёт mu < 0)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fixed = np.zeros(self.N, bool)
        self.G3 = np.zeros(3)
        self._G3_at_build = None

    def F(self, g=None):
        F = np.eye(3)
        F[0, 1], F[0, 2], F[1, 2] = self.G3
        return F

    def need_rebuild(self):
        if self._pairs is None or self._G3_at_build is None:
            return True
        if np.abs(self._G3_at_build - self.G3).max() > 0.02:
            return True
        d = np.abs(self._s_at_build - self.s).max()
        return d > 0.5 * self.skin / 1.3

    def build_pairs(self):
        super().build_pairs()
        self._G3_at_build = self.G3.copy()

    def energy_grad(self, sflat):
        E, g = super().energy_grad(sflat)
        if self.fixed.any():
            g = g.reshape(self.N, 3).copy()
            g[self.fixed] = 0.0
            g = g.ravel()
        return E, g

    def _energy_grad_ext(self, x):
        """Энергия и градиент по (s, G3)."""
        self.G3 = x[-3:].copy()
        E, g = self.energy_grad(x[:-3])
        s = x[:-3].reshape(self.N, 3)
        ds, dx, r = self._geom(s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        dVdr = np.zeros_like(r); dVdr[m] = -(ov[m] ** (self.alpha - 1)) / self._sig[m]
        gg = (dVdr / r)[:, None] * dx                # g = V'(r) n
        gG = np.array([(gg[:, 0] * ds[:, 1]).sum(), (gg[:, 0] * ds[:, 2]).sum(), (gg[:, 1] * ds[:, 2]).sum()])
        return E, np.concatenate([g, gG])

    def stabilize(self, rel=1e-9, rounds=6):
        """Совместная минимизация по частицам и форме ячейки, затем дожим по частицам."""
        from scipy.optimize import minimize as _min
        for _ in range(rounds):
            self._pairs = None; self.build_pairs()
            gtol = max(1e-14, rel * self.force_scale())
            x0 = np.concatenate([self.s.ravel(), self.G3])
            thr = 0.5 * self.skin / 1.3
            st = {}

            def cb(xk):
                if np.abs(xk[:-3].reshape(self.N, 3) - self._s_at_build).max() > thr:
                    st["x"] = xk.copy(); raise StopIteration
            try:
                res = _min(self._energy_grad_ext, x0, jac=True, method="L-BFGS-B", callback=cb,
                           options=dict(maxiter=50000, gtol=gtol, ftol=0.0, maxcor=30))
                x = res.x
            except StopIteration:
                x = st["x"]
            self.s = x[:-3].reshape(self.N, 3); self.G3 = x[-3:].copy()
            self.s = np.mod(self.s, self.L); self._pairs = None
            E = self.minimize(rel=rel)
            sxy = self.residual_shear()
            if np.abs(sxy).max() < 1e-3 * self.stress_pressure()[1]:
                break
        return E

    def residual_shear(self):
        i, j, dx, r, ov = self.contacts()
        sig = self.rad[i] + self.rad[j]
        dVdr = -(ov ** (self.alpha - 1)) / sig
        gg = (dVdr / r)[:, None] * dx
        V = self.L ** 3
        return np.array([(gg[:, 0] * dx[:, 1]).sum(), (gg[:, 0] * dx[:, 2]).sum(), (gg[:, 1] * dx[:, 2]).sum()]) / V

    def contact_moduli(self):
        """Вклады контактов в гессианов модуль сдвига (среднее по 6 сдвигам ab)
        и середины контактов. Возвращает mid (n_c, 3), mu_c (n_c,), mu0."""
        i, j, dx, r, ov = self.contacts()
        sig = self.rad[i] + self.rad[j]
        a = self.alpha
        dVdr = -(ov ** (a - 1)) / sig
        d2V = (a - 1) * ov ** (a - 2) / sig ** 2
        V = self.L ** 3
        n = dx / r[:, None]
        # живые частицы (без дребезжащих)
        alive = np.ones(self.N, bool)
        while True:
            m = alive[i] & alive[j]
            deg = np.bincount(np.concatenate([i[m], j[m]]), minlength=self.N)
            rat = alive & (deg < 4)
            if not rat.any():
                break
            alive[rat] = False
        idx = -np.ones(self.N, int); idx[alive] = np.arange(alive.sum())
        m = alive[i] & alive[j]
        i2, j2, n2, dVdr2, d2V2, r2, dx2 = idx[i[m]], idx[j[m]], n[m], dVdr[m], d2V[m], r[m], dx[m]
        M = alive.sum()
        K = (d2V2 - dVdr2 / r2)[:, None, None] * n2[:, :, None] * n2[:, None, :] \
            + (dVdr2 / r2)[:, None, None] * np.eye(3)[None]
        H = np.zeros((3 * M, 3 * M))
        for a_ in range(3):
            for b_ in range(3):
                np.add.at(H, (3 * i2 + a_, 3 * i2 + b_), K[:, a_, b_])
                np.add.at(H, (3 * j2 + a_, 3 * j2 + b_), K[:, a_, b_])
                np.add.at(H, (3 * i2 + a_, 3 * j2 + b_), -K[:, a_, b_])
                np.add.at(H, (3 * j2 + a_, 3 * i2 + b_), -K[:, a_, b_])
        lam = 1e-9 * np.trace(H) / (3 * M)
        H[np.diag_indices_from(H)] += lam
        free = np.ones(3 * M, bool)
        fx = self.fixed & alive
        if fx.any():
            ff = idx[fx]
            for c in range(3):
                free[3 * ff + c] = False
        Hf = H[np.ix_(free, free)]
        try:
            cf = cho_factor(Hf, check_finite=False); solve = lambda b: cho_solve(cf, b, check_finite=False)
        except np.linalg.LinAlgError:          # почти нулевая мода (незавершённая перестройка) — LU
            lu = lu_factor(Hf, check_finite=False); solve = lambda b: lu_solve(lu, b, check_finite=False)
        g = dVdr2[:, None] * n2                     # g = V'(r) n = dV/d(dx)
        mu_c = np.zeros(len(i2)); mu0 = 0.0
        for (a_, b_) in SHEARS:
            e = np.zeros_like(dx2); e[:, a_] = dx2[:, b_]
            xi_pair = np.einsum('kab,kb->ka', K, e)
            Xi = np.zeros((M, 3))
            np.add.at(Xi, j2, xi_pair); np.add.at(Xi, i2, -xi_pair)
            Xi = Xi.ravel()
            u = np.zeros(3 * M)
            u[free] = -solve(Xi[free])
            u = u.reshape(M, 3)
            d = e + u[j2] - u[i2]
            Kd = np.einsum('kab,kb->ka', K, d)
            mc = (Kd[:, a_] * dx2[:, b_] + g[:, a_] * d[:, b_]) / V
            mu_c += mc / len(SHEARS)
            mu0 += mc.sum() / len(SHEARS)
        # середины контактов в аффинных координатах (расстояния берутся через F)
        s = np.mod(self.s, self.L)
        ds2 = dx2 @ np.linalg.inv(self.F()).T
        mid = s[i[m]] + 0.5 * ds2                  # ds = s_j - s_i (мин. образ)
        return mid, mu_c, mu0, alive


def contact_set(p):
    i, j, *_ = p.contacts()
    return set(zip(i.tolist(), j.tolist()))


def shell_profile(mid, mu_c, center, L, edges, F=None):
    d = mid - center
    d -= L * np.round(d / L)
    if F is not None:
        d = d @ F.T
    r = np.linalg.norm(d, axis=1)
    sums = np.zeros(len(edges) - 1)
    k = np.searchsorted(edges, r, side="right") - 1
    ok = (k >= 0) & (k < len(sums))
    np.add.at(sums, k[ok], mu_c[ok])
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return sums / vol


def relax(p, rel=1e-9):
    p._pairs = None
    return p.minimize(rel=rel)


def run_sample(N, seed, dphis, nsrc, epss, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed} ---", flush=True)
    p0, phi_j = find_phi_j(N, seed, False, verbose=verbose)
    p = PinnedSpheres(N, 0.65, seed)
    p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
    p.s, p.L = p0.s.copy(), p0.L
    s_j, L_j = p.s.copy(), p.L
    rng = np.random.default_rng(1000 + seed)
    out = []
    for dphi in dphis:
        p.s, p.L = s_j.copy(), L_j
        p.fixed[:] = False
        p.set_phi(phi_j + dphi)
        p.G3[:] = 0.0
        E0 = p.stabilize()
        s0 = p.s.copy(); rad0 = p.rad.copy()
        print(f"    сдвиговая стабилизация: G3={p.G3}, остаточный сдвиг/p = {np.abs(p.residual_shear()).max()/p.stress_pressure()[1]:.1e}", flush=True)
        z, nrat, nc = p.coordination()
        _, pr = p.stress_pressure()
        mid0, muc0, mu0, alive = p.contact_moduli()
        cs0 = contact_set(p)
        a_cell = (p.L ** 3 / N) ** (1.0 / 3.0)
        edges = np.arange(0.0, p.L / 2 + 1e-9, 0.25)
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, p=pr, z=z, dz=z - Z_C, nrat=nrat,
                   mu0=mu0, L=p.L, a=a_cell, edges=edges, E0=E0, src=[])
        print(f"    dphi={dphi:<6g} p={pr:.3e} z={z:.3f} dz={z-Z_C:+.3f} mu0={mu0:.4e} a={a_cell:.3f} "
              f"L={p.L:.2f}  [{time.time()-t0:.0f} с]", flush=True)
        # выбор источников: живые частицы; для P — живой сосед
        i_c, j_c, *_ = p.contacts()
        cand = np.where(alive)[0]
        srcs = rng.choice(cand, size=min(nsrc, len(cand)), replace=False)

        # контроль: повторная релаксация без источника (шумовой пол)
        p.s = s0.copy(); p.rad = rad0.copy(); p.s += 1e-7 * rng.standard_normal(p.s.shape)
        E1 = relax(p)
        mid1, muc1, mu1, _ = p.contact_moduli()
        for isrc in srcs:
            c = np.mod(s0[isrc], p.L)
            prof0 = shell_profile(mid0, muc0, c, p.L, edges, p.F())
            prof1 = shell_profile(mid1, muc1, c, p.L, edges, p.F())
            rec["src"].append(dict(kind="0", i=int(isrc), eps=0.0, E=E1 - E0, dmu_tot=mu1 - mu0, dnc=0,
                                   dprof=prof1 - prof0, prof0=prof0))
        print(f"      контроль eps=0: dE={E1-E0:+.2e}  dmu0/mu0={(mu1-mu0)/mu0:+.2e}", flush=True)

        for isrc in srcs:
            c = np.mod(s0[isrc], p.L)
            prof0 = shell_profile(mid0, muc0, c, p.L, edges, p.F())
            # --- D: центр дилатации ---
            for eps in epss:
                for sgn in (+1, -1):
                    p.s = s0.copy(); p.rad = rad0.copy(); p.fixed[:] = False
                    p.rad[isrc] *= (1 + sgn * eps)
                    E1 = relax(p)
                    mid1, muc1, mu1, _ = p.contact_moduli()
                    prof1 = shell_profile(mid1, muc1, c, p.L, edges, p.F())
                    dnc = len(cs0 ^ contact_set(p))
                    rec["src"].append(dict(kind="D", i=int(isrc), eps=sgn * eps, E=E1 - E0, dnc=dnc,
                                           dmu_tot=mu1 - mu0, dprof=prof1 - prof0, prof0=prof0))
            # --- P: закреплённая пара ---
            nb = np.concatenate([j_c[i_c == isrc], i_c[j_c == isrc]])
            nb = nb[alive[nb]]
            if len(nb) == 0:
                continue
            jsrc = int(nb[rng.integers(len(nb))])
            d = s0[jsrc] - s0[isrc]; d -= p.L * np.round(d / p.L)
            nvec = d / np.linalg.norm(d)
            cP = np.mod(s0[isrc] + 0.5 * d, p.L)
            # опорный расчёт: та же пара закреплена, delta = 0
            p.s = s0.copy(); p.rad = rad0.copy(); p.fixed[:] = False
            p.fixed[[isrc, jsrc]] = True
            p._pairs = None
            midR, mucR, muR, _ = p.contact_moduli()
            profR = shell_profile(midR, mucR, cP, p.L, edges, p.F())
            for eps in epss:
                for sgn in (+1, -1):
                    p.s = s0.copy(); p.rad = rad0.copy()
                    p.fixed[:] = False; p.fixed[[isrc, jsrc]] = True
                    p.s[isrc] -= 0.5 * sgn * eps * nvec
                    p.s[jsrc] += 0.5 * sgn * eps * nvec
                    E1 = relax(p)
                    mid1, muc1, mu1, _ = p.contact_moduli()
                    prof1 = shell_profile(mid1, muc1, cP, p.L, edges, p.F())
                    dnc = len(cs0 ^ contact_set(p))
                    rec["src"].append(dict(kind="P", i=int(isrc), j=jsrc, eps=sgn * eps, E=E1 - E0, dnc=dnc,
                                           dmu_tot=mu1 - muR, dprof=prof1 - profR, prof0=profR, muR=muR))
            p.fixed[:] = False
            if verbose:
                nD = sum(1 for s_ in rec["src"] if s_["kind"] == "D" and s_["i"] == isrc)
                print(f"      источник {isrc}: D×{nD}, P — готово  [{time.time()-t0:.0f} с]", flush=True)
        p.s = s0.copy(); p.rad = rad0.copy(); p.fixed[:] = False; p._pairs = None
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def _fit_m(rc, y):
    ok = y > 0
    return -np.polyfit(np.log(rc[ok]), np.log(y[ok]), 1)[0] if ok.sum() >= 3 else np.nan


def analyse(recs):
    head("СВОДКА T-4: локальный статический источник против ОТО (C = 4, m = 1)")
    dphis = sorted(set(r["dphi"] for r in recs)); Ns = sorted(set(r["N"] for r in recs))
    epss = sorted(set(abs(s["eps"]) for r in recs for s in r["src"] if s["eps"] != 0))
    print(f"  образцов: {len(recs)}; N={Ns}; dphi={dphis}; амплитуды {epss}")
    print("  C(r) = -delta mu(r) a^2 r / E (ОТО: 4);  m — показатель |delta mu| ~ r^-m на оболочках")
    print("  1.5 <= r <= L/2 (ОТО: 1);  события делятся на «гладкие» (набор контактов не изменился, dnc = 0)")
    print("  и «перестройки» (dnc > 0). Для гладких — среднее по событиям и знакам; D: E ~ eps (нечётна),")
    print("  поэтому C сравнивается по знакам отдельно.\n")
    summary = {}
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["N"] == N and r["dphi"] == d]
            if not rr:
                continue
            nb = min(len(r["edges"]) for r in rr) - 1      # образцы с чуть разной L: общая часть сетки
            edges = rr[0]["edges"][:nb + 1]; rc = 0.5 * (edges[1:] + edges[:-1])
            for r in rr:
                for s_ in r["src"]:
                    s_["dprof"] = s_["dprof"][:nb]; s_["prof0"] = s_["prof0"][:nb]
            L = np.mean([r["L"] for r in rr]); a = np.mean([r["a"] for r in rr])
            mu0 = np.mean([r["mu0"] for r in rr]); dz = np.mean([r["dz"] for r in rr])
            sel = (rc >= 1.5) & (rc <= L / 2)
            print(f"  ==== N={N} dphi={d:g}: dz={dz:.3f} mu0={mu0:.4e} a={a:.3f} L={L:.2f}, образцов {len(rr)} ====")
            ctrl = [s for r in rr for s in r["src"] if s["kind"] == "0"]
            if ctrl:
                floor = np.sqrt(np.mean([s["dprof"] ** 2 for s in ctrl], axis=0)) / mu0
                print(f"    шумовой пол |delta mu(r)|/mu0 (контроль): {floor[sel].max():.1e}")
            print("       r:            " + " ".join(f"{x:8.2f}" for x in rc[sel]))
            for kind in ("D", "P"):
                for eps in epss:
                    ev = [s for r in rr for s in r["src"] if s["kind"] == kind and abs(abs(s["eps"]) - eps) < 1e-15]
                    if not ev:
                        continue
                    n = len(ev)
                    sm = [s for s in ev if s["dnc"] == 0]; rg = [s for s in ev if s["dnc"] > 0]
                    print(f"    -- {kind}, |eps|={eps:g}: событий {n}, гладких {len(sm)}, перестроек {len(rg)}"
                          + (f" (dnc медиана {np.median([s['dnc'] for s in rg]):.0f})" if rg else ""))
                    row = dict(dz=dz, n=n, n_smooth=len(sm), n_re=len(rg))
                    for name, grp in (("гладкие", sm), ("перестройки", rg)):
                        if not grp:
                            continue
                        E = np.array([s["E"] for s in grp]); P = np.array([s["dprof"] for s in grp])
                        T = np.array([s["dmu_tot"] for s in grp]); sg = np.sign([s["eps"] for s in grp])
                        C = -P * a ** 2 * rc[None, :] / E[:, None]
                        Pm = P.mean(axis=0)
                        m_abs = _fit_m(rc[sel], np.abs(Pm)[sel])
                        # для D — по знакам отдельно
                        parts = [("все", np.ones(len(grp), bool))]
                        if kind == "D":
                            parts = [("eps>0", sg > 0), ("eps<0", sg < 0)]
                        print(f"       [{name}] <E>/mu0a^3={E.mean()/mu0/a**3:+.2e}  <dmu_tot>/mu0={T.mean()/mu0:+.2e} "
                              f"(медиана {np.median(T)/mu0:+.2e}, доля отриц. {np.mean(T<0):.2f})  m(|<dmu>|)={m_abs:.2f}")
                        for pn, pm in parts:
                            if pm.sum() == 0:
                                continue
                            Cm = C[pm].mean(axis=0); Cse = C[pm].std(axis=0, ddof=1) / np.sqrt(pm.sum()) if pm.sum() > 1 else 0 * Cm
                            Pp = P[pm].mean(axis=0) / mu0
                            print(f"         {pn:6s} n={pm.sum():3d}  <E>/mu0a^3={E[pm].mean()/mu0/a**3:+.2e}")
                            print("           dmu/mu0:  " + " ".join(f"{x:8.1e}" for x in Pp[sel]))
                            print("           C:        " + " ".join(f"{x:8.3f}" for x in Cm[sel]))
                            print("           ±се:      " + " ".join(f"{x:8.3f}" for x in Cse[sel]))
                            row[(name, pn)] = dict(C=Cm, Cse=Cse, dmu=Pp, E=E[pm].mean() / mu0 / a ** 3,
                                                   T=T[pm].mean() / mu0, m=m_abs)
                    summary[(N, d, kind, eps)] = row
    print("""
  Чтение.
  * ОТО: C(r) = 4 на всех r, одинаково для любого источника с данной E, m = 1, никакой
    зависимости от знака источника (только от E).
  * Линейная упругость с ангармонизмом: гладкий отклик первого порядка ~амплитуде,
    m = 3, знак меняется со знаком источника; второго порядка ~E, m = 6.
  * Перестройки: пороговый, невзаимно-однозначный по E отклик; их доля ~ амплитуде.
  * Знак: C > 0 — размягчение вдали от источника (нужный знак), C < 0 — ужесточение.
""")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--nsrc", type=int, default=8)
    ap.add_argument("--eps", type=str, default="1e-4,1e-3,1e-2")
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
    epss = [float(s) for s in args.eps.split(",")]
    head(f"T-4: локальный источник на заклиненном образце, N={args.N}, затравки {seeds}, dphi={dphis}, "
         f"источников {args.nsrc}, амплитуды {epss}")
    recs = []
    out = args.out or os.path.join(HERE, f"gravity_softening_N{args.N}.npz")
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.nsrc, epss)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
