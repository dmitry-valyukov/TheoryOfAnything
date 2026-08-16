"""Закон размягчения по амплитуде на НАСТОЯЩЕМ заклиненном образце (T-44, T-34, T-4).

Постановка. Первый счёт (SofteningLaw.py) показал: гармоническая сеть
пружин под сдвигом упрочняется, знак меняет только сжимающее преднапряжение
у изостатики — размягчение есть свойство заклиненного состояния. Здесь
строится сам заклиненный образец и на нём измеряется закон mu(gamma).

Образец. Мягкие сферы (бидисперсная смесь 50:50, отношение диаметров 1.4,
малый диаметр = 1), периодический куб, контактный потенциал
    V(r) = (1/2)(1 - r/sigma_ij)^2   при r < sigma_ij   (гармонический)
    V(r) = (2/5)(1 - r/sigma_ij)^{5/2}                 (герцевский, ключ --hertz)
Точка заклинивания phi_J находится бисекцией по плотности упаковки
(релаксация L-BFGS до нулевой силы; «заклинено» = энергия на частицу
> 1e-16). Затем образец сжимается до phi_J + dphi; измеряются давление p,
координационное число z (без «дребезжащих» частиц, < 4 контактов),
dz = z - 6, линейный модуль сдвига mu0 и объёмный модуль B.

Сдвиг. Квазистатический атермический простой сдвиг: аффинный шаг gamma
(ячейка Лиза — Эдвардса, координаты хранятся в аффинном базисе s, x = F s),
затем полная неаффинная релаксация; контакты размыкаются и замыкаются
сами (потенциал односторонний). Напряжение sigma_xy = (1/V) dE/dgamma
(вириал). Секущий модуль mu_s(gamma) = sigma/gamma и касательный
mu_t = d sigma/d gamma сравниваются с mu0.

Что проверяется (программа T-44):
  (а) знак: размягчается ли заклиненный образец под сдвигом;
  (б) форма закона: 1 - mu(gamma)/mu0 ~ gamma^nu при малых gamma
      (nu = 2 — аналитический квартичный член, nu = 1 — размыкание
      контактов), масштаб перехода gamma_x и его зависимость от dz;
  (в) есть ли у глубины размягчения плато delta_max, универсальное по
      затравкам/N/dz, и как оно соотносится с 0.06–0.08 (4.14, путь 10).

Запуск:  python JammedSoftening.py [--N 1024] [--seeds 1,2,3] [--dphi 1e-3,3e-3,1e-2,3e-2,1e-1]
                                   [--gmax 0.1] [--hertz] [--out results_jammed.npz] [--quick]
Результаты (кривые) складываются в npz; сводка печатается в конце.
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # мелкие матрицы: потоки BLAS только мешают
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
Z_C = 6.0


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78, flush=True)


class SoftSpheres:
    """Мягкие сферы в периодической ячейке с простым сдвигом gamma.

    Координаты s лежат в кубе [0, L)^3 (аффинный базис), реальные x = F s,
    F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]."""

    def __init__(self, N, phi, seed, hertz=False, ratio=1.4, skin=0.4):
        rng = np.random.default_rng(seed)
        self.N = N
        self.hertz = hertz
        self.alpha = 2.5 if hertz else 2.0
        diam = np.where(np.arange(N) < N // 2, 1.0, ratio)
        rng.shuffle(diam)
        self.rad = 0.5 * diam
        self.vol_particles = (np.pi / 6.0 * diam ** 3).sum()
        self.L = (self.vol_particles / phi) ** (1.0 / 3.0)
        self.s = rng.uniform(0, self.L, (N, 3))
        self.gamma = 0.0
        self.skin = skin
        self.rc = 2 * self.rad.max()
        self._pairs = None
        self._s_at_build = None
        self._gamma_at_build = None

    # ---------- геометрия ----------
    @property
    def phi(self):
        return self.vol_particles / self.L ** 3

    def set_phi(self, phi):
        Lnew = (self.vol_particles / phi) ** (1.0 / 3.0)
        self.s *= Lnew / self.L
        self.L = Lnew
        self._pairs = None

    def F(self, g=None):
        g = self.gamma if g is None else g
        return np.array([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def build_pairs(self):
        F = self.F()
        Finv = np.linalg.inv(F)
        stretch = np.linalg.norm(Finv, 2)          # |ds| <= r * ||F^-1||
        rc_s = (self.rc + self.skin) * stretch
        s = np.mod(self.s, self.L)
        tree = cKDTree(s, boxsize=self.L)
        pr = tree.query_pairs(rc_s, output_type="ndarray")
        self._pairs = pr
        self._i, self._j = pr[:, 0], pr[:, 1]
        self._sig = self.rad[self._i] + self.rad[self._j]
        self._s_at_build = self.s.copy()
        self._gamma_at_build = self.gamma

    def need_rebuild(self):
        if self._pairs is None or self._gamma_at_build != self.gamma:
            return True
        d = np.abs(self._s_at_build - self.s).max()
        return d > 0.5 * self.skin / 1.3

    def _geom(self, s):
        ds = s[self._j] - s[self._i]
        ds -= self.L * np.round(ds / self.L)
        dx = ds @ self.F().T
        r = np.linalg.norm(dx, axis=1)
        return ds, dx, r

    # ---------- энергия ----------
    def energy_grad(self, sflat):
        s = sflat.reshape(self.N, 3)
        ds, dx, r = self._geom(s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        a = self.alpha
        E = (ov[m] ** a).sum() / a
        # dE/dr = -(ov^(a-1))/sig ;  dE/dx_j = dE/dr * dx/r
        fac = np.zeros_like(r)
        fac[m] = -(ov[m] ** (a - 1)) / self._sig[m] / r[m]
        f = fac[:, None] * dx                        # dE/d(dx)
        gx = np.stack([np.bincount(self._j, f[:, c], self.N) - np.bincount(self._i, f[:, c], self.N)
                       for c in range(3)], axis=1)
        gs = gx @ self.F()                           # dE/ds = F^T dE/dx
        return E, gs.ravel()

    def force_scale(self):
        """Средняя контактная сила — масштаб для относительного допуска."""
        if self.need_rebuild():
            self.build_pairs()
        ds, dx, r = self._geom(self.s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        if not m.any():
            return 0.0
        return float((ov[m] ** (self.alpha - 1) / self._sig[m]).mean())

    def fire(self, gtol, maxsteps=100000, dt0=0.02):
        """FIRE-дожим до max|grad| < gtol (не пользуется разностями энергии,
        поэтому доводит силы до предела точности самих сил)."""
        v = np.zeros((self.N, 3)); dt = dt0; a = 0.1; n_pos = 0
        for it in range(maxsteps):
            if self.need_rebuild():
                self.build_pairs()
            E, g = self.energy_grad(self.s.ravel())
            f = -g.reshape(self.N, 3)
            fmax = np.abs(f).max()
            if fmax < gtol:
                return True
            P = (f * v).sum()
            if P > 0:
                n_pos += 1
                vn, fn = np.linalg.norm(v), np.linalg.norm(f)
                v = (1 - a) * v + a * f * (vn / (fn + 1e-300))
                if n_pos > 5:
                    dt = min(dt * 1.1, 10 * dt0); a *= 0.99
            else:
                n_pos = 0; v[:] = 0; dt *= 0.5; a = 0.1
            v += dt * f
            self.s += dt * v
        return False

    def minimize(self, rel=1e-6, maxiter=50000):
        """Полная релаксация при фиксированных L и gamma: L-BFGS c относительным
        допуском по силе, при остановке или частых перестройках списка —
        FIRE-дожим (он перестраивает список сам). L-BFGS прерывается, как только
        смещения выходят за «шкурку» списка (иначе он бежит по неверной энергии)."""
        thr = 0.5 * self.skin / 1.3
        state = {}

        def cb(xk):
            if np.abs(xk.reshape(self.N, 3) - self._s_at_build).max() > thr:
                state["x"] = xk.copy()
                raise StopIteration

        n_int = 0
        for _ in range(100):
            if self.need_rebuild():
                self.build_pairs()
            gtol = max(1e-14, rel * self.force_scale())
            if n_int >= 8:                       # список рвётся слишком часто — FIRE
                self.fire(gtol)
                break
            state.clear()
            try:
                res = minimize(self.energy_grad, self.s.ravel(), jac=True, method="L-BFGS-B",
                               callback=cb,
                               options=dict(maxiter=maxiter, gtol=gtol, ftol=0.0, maxcor=30))
            except StopIteration:
                self.s = state["x"].reshape(self.N, 3)
                n_int += 1
                continue
            self.s = res.x.reshape(self.N, 3)
            if self.need_rebuild():
                n_int += 1
                continue
            # масштаб сил мог упасть по ходу релаксации (расклинивание) — допуск пересчитывается
            gtol2 = max(1e-14, rel * self.force_scale())
            if np.abs(res.jac).max() <= 10 * gtol2:
                break
            if res.status == 0 and gtol2 < gtol:      # сошёлся к старому допуску: ещё круг с новым
                continue
            self.fire(gtol2)                          # L-BFGS встал (поиск вдоль прямой)
            break
        self.s = np.mod(self.s, self.L)
        self._pairs = None
        self.build_pairs()
        return self.energy_grad(self.s.ravel())[0]

    # ---------- наблюдаемые ----------
    def contacts(self):
        if self.need_rebuild():
            self.build_pairs()
        ds, dx, r = self._geom(self.s)
        m = r < self._sig
        return self._i[m], self._j[m], dx[m], r[m], (1.0 - r[m] / self._sig[m])

    def stress_pressure(self):
        """sigma_xy = (1/V) dE/dgamma (аффинная производная = вириал),
        p = -(1/3V) sum V'(r) r."""
        i, j, dx, r, ov = self.contacts()
        sig = self.rad[i] + self.rad[j]
        dVdr = -(ov ** (self.alpha - 1)) / sig
        V = self.L ** 3
        # d(dx_x)/dgamma = ds_y = dx_y
        sxy = (dVdr / r * dx[:, 0] * dx[:, 1]).sum() / V
        p = -(dVdr * r).sum() / (3 * V)
        return sxy, p

    def coordination(self):
        """z без дребезжащих частиц (< 4 контактов, рекурсивно)."""
        i, j, *_ = self.contacts()
        alive = np.ones(self.N, bool)
        while True:
            m = alive[i] & alive[j]
            deg = np.bincount(np.concatenate([i[m], j[m]]), minlength=self.N)
            rat = alive & (deg < 4)
            if not rat.any():
                break
            alive[rat] = False
        m = alive[i] & alive[j]
        n_alive = alive.sum()
        z = 2.0 * m.sum() / max(n_alive, 1)
        return z, self.N - n_alive, int(m.sum())

    def energy(self):
        if self.need_rebuild():
            self.build_pairs()
        return self.energy_grad(self.s.ravel())[0]

    def linear_moduli(self):
        """Линейный отклик: аффинный модуль mu_aff и полный (с неаффинной
        релаксацией) mu0 = mu_aff - Xi^T H^{-1} Xi / V; плотная матрица
        Гессе по живым (не дребезжащим) частицам."""
        i, j, dx, r, ov = self.contacts()
        sig = self.rad[i] + self.rad[j]
        a = self.alpha
        dVdr = -(ov ** (a - 1)) / sig
        d2V = (a - 1) * ov ** (a - 2) / sig ** 2
        V = self.L ** 3
        n = dx / r[:, None]
        # аффинный модуль
        mu_aff = ((d2V - dVdr / r) * (dx[:, 0] * dx[:, 1] / r) ** 2 + dVdr / r * dx[:, 1] ** 2).sum() / V
        # живые частицы
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
        H = np.zeros((3 * M, 3 * M))
        # блок пары: K = d2V n n^T + (dVdr/r)(I - n n^T)
        K = (d2V2 - dVdr2 / r2)[:, None, None] * n2[:, :, None] * n2[:, None, :] \
            + (dVdr2 / r2)[:, None, None] * np.eye(3)[None]
        for a_ in range(3):
            for b_ in range(3):
                np.add.at(H, (3 * i2 + a_, 3 * i2 + b_), K[:, a_, b_])
                np.add.at(H, (3 * j2 + a_, 3 * j2 + b_), K[:, a_, b_])
                np.add.at(H, (3 * i2 + a_, 3 * j2 + b_), -K[:, a_, b_])
                np.add.at(H, (3 * j2 + a_, 3 * i2 + b_), -K[:, a_, b_])
        # Xi = d^2E/(dgamma dx): d(dx_x)/dgamma = dx_y  =>  dF/dgamma на пару
        # f_pair = dV/dr * n ; d f_pair/dgamma = K @ (dx_y, 0, 0)
        e = np.zeros_like(dx2); e[:, 0] = dx2[:, 1]
        xi_pair = np.einsum('kab,kb->ka', K, e)
        Xi = np.zeros((M, 3))
        np.add.at(Xi, j2, xi_pair); np.add.at(Xi, i2, -xi_pair)
        Xi = Xi.ravel()
        # снятие трансляций: сдвиг диагонали на малую величину
        lam = 1e-9 * np.trace(H) / (3 * M)
        H[np.diag_indices_from(H)] += lam
        u = np.linalg.solve(H, Xi)
        mu0 = mu_aff - (Xi @ u) / V
        return mu0, mu_aff


# ---------------------------------------------------------------------------
def find_phi_j(N, seed, hertz, phi_hi=0.70, phi_lo=0.58, tol=5e-6, verbose=True):
    """Бисекция по phi: заклинено <=> E/N > 1e-16 после релаксации
    (допуск по силе здесь грубее, 1e-4: нужен лишь ответ «да/нет»)."""
    p = SoftSpheres(N, phi_hi, seed, hertz=hertz)
    p.minimize(rel=1e-4)
    assert p.energy() / N > 1e-16, "стартовая плотность не заклинена"
    s_j, L_j = p.s.copy(), p.L
    it = 0
    while phi_hi - phi_lo > tol:
        it += 1
        phi_m = 0.5 * (phi_hi + phi_lo)
        p.s, p.L = s_j.copy(), L_j
        p.set_phi(phi_m)
        p.minimize(rel=1e-4)
        e = p.energy() / N
        if e > 1e-16:
            phi_hi = phi_m; s_j, L_j = p.s.copy(), p.L
        else:
            phi_lo = phi_m
    p.s, p.L = s_j.copy(), L_j
    p._pairs = None
    if verbose:
        print(f"    phi_J = {phi_hi:.6f}  ({it} шагов бисекции)", flush=True)
    return p, phi_hi


def bulk_modulus(p, rel=1e-3):
    """B = phi dp/dphi центральной разностью с релаксацией."""
    s0, L0 = p.s.copy(), p.L
    phi0 = p.phi
    ps = []
    for f in (1 - rel, 1 + rel):
        p.s, p.L = s0.copy(), L0
        p.set_phi(phi0 * f)
        p.minimize()
        ps.append(p.stress_pressure()[1])
    p.s, p.L = s0.copy(), L0
    p._pairs = None
    return phi0 * (ps[1] - ps[0]) / (2 * rel * phi0)


def shear_curve(p, gmax=0.1, per_decade=6, gmin=1e-5, dg_max=2e-3, rel=1e-4, verbose=False):
    """Квазистатический сдвиг: сетка по gamma логарифмическая, шаг <= dg_max."""
    pts = list(np.geomspace(gmin, gmax, int(round(per_decade * math.log10(gmax / gmin))) + 1))
    grid = [0.0]
    for g in pts:
        last = grid[-1]
        n_sub = max(1, int(math.ceil((g - last) / dg_max)))
        for k in range(1, n_sub + 1):
            grid.append(last + (g - last) * k / n_sub)
    grid = np.array(grid)
    sig = np.zeros_like(grid); Es = np.zeros_like(grid); zs = np.zeros_like(grid)
    ncs = np.zeros_like(grid, int)
    keep = np.isin(grid, np.concatenate([[0.0], pts]))
    muH = np.full(grid.shape, np.nan)      # линейный (гессианов) модуль текущего деформированного состояния
    p.gamma = 0.0
    p.minimize()
    for k, g in enumerate(grid):
        p.gamma = g
        Es[k] = p.minimize(rel=(1e-6 if g < 2e-4 else rel))   # у малых gamma сигнал мал — допуск строже
        sig[k], _ = p.stress_pressure()
        zs[k], _, ncs[k] = p.coordination()
        if keep[k]:
            muH[k] = p.linear_moduli()[0]
        if verbose and k % 50 == 0:
            print(f"      gamma={g:.4g} sigma={sig[k]:.4g} z={zs[k]:.3f}", flush=True)
    sig = sig - sig[0]                     # остаточное напряжение образца при gamma = 0
    p.gamma = 0.0
    return grid[keep], sig[keep], Es[keep], zs[keep], ncs[keep], grid, sig, muH[keep]


def run_sample(N, seed, dphis, hertz, gmax, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed}, {'Герц' if hertz else 'гармонический'} ---", flush=True)
    p, phi_j = find_phi_j(N, seed, hertz, verbose=verbose)
    print(f"    [{time.time()-t0:.0f} с на поиск phi_J]", flush=True)
    s_j, L_j = p.s.copy(), p.L
    out = []
    for dphi in dphis:
        p.s, p.L = s_j.copy(), L_j
        p.gamma = 0.0
        p.set_phi(phi_j + dphi)
        p.minimize()
        sxy0, pr = p.stress_pressure()
        z, nrat, nc = p.coordination()
        B = bulk_modulus(p)
        mu0, mu_aff = p.linear_moduli()
        g, sig, Es, zs, ncs, gfull, sfull, muH = shear_curve(p, gmax=gmax)
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, p=pr, z=z, dz=z - Z_C, nrat=nrat,
                   B=B, mu0=mu0, mu_aff=mu_aff, gamma=g, sigma=sig, E=Es, zg=zs, nc=ncs,
                   gamma_full=gfull, sigma_full=sfull, muH=muH)
        out.append(rec)
        print(f"    dphi={dphi:<7g} p={pr:.3e} z={z:.3f} dz={z-Z_C:+.3f} rattlers={nrat:3d} "
              f"mu0={mu0:.3e} B={B:.3e} G/B={mu0/B:.4f} mu0/mu_aff={mu0/mu_aff:.3f}   "
              f"[{time.time()-t0:.0f} с]", flush=True)
    return out


# ---------------------------------------------------------------------------
def secant(rec):
    g, s = rec["gamma"][1:], rec["sigma"][1:]
    return g, s / g / rec["mu0"]


def tangent(rec):
    """Касательный модуль по полной сетке (сглаженная производная по log-окнам)."""
    g, s = rec["gamma_full"], rec["sigma_full"]
    gc, mt = [], []
    for gk in rec["gamma"][1:]:
        lo, hi = gk / 1.5, gk * 1.5
        m = (g >= lo) & (g <= hi)
        if m.sum() >= 3:
            A = np.stack([g[m], np.ones(m.sum())], 1)
            c, *_ = np.linalg.lstsq(A, s[m], rcond=None)
            gc.append(gk); mt.append(c[0] / rec["mu0"])
        else:
            k = np.searchsorted(g, gk)
            k = min(max(k, 1), len(g) - 1)
            gc.append(gk); mt.append((s[k] - s[k - 1]) / (g[k] - g[k - 1]) / rec["mu0"])
    return np.array(gc), np.array(mt)


def crossover(g, mu_rel, level):
    """Первое gamma, при котором 1 - mu/mu0 достигает level (лог-интерполяция)."""
    d = 1 - mu_rel
    idx = np.where(d >= level)[0]
    if len(idx) == 0:
        return np.nan
    k = idx[0]
    if k == 0:
        return g[0]
    x0, x1 = math.log(g[k - 1]), math.log(g[k]); y0, y1 = d[k - 1], d[k]
    return math.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def yield_point(rec):
    g, s = rec["gamma_full"], rec["sigma_full"]
    k = int(np.argmax(s))
    return g[k], s[k]


def loglog_slope(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]


def ensemble(recs):
    """Средняя по образцам кривая mu_s(gamma)/mu0 (каждый образец нормирован
    на свой линейный модуль); та же сетка gamma у всех образцов."""
    g = recs[0]["gamma"][1:]
    M = np.array([secant(r)[1] for r in recs])
    return g, M.mean(axis=0), M.std(axis=0, ddof=1) / np.sqrt(len(recs)) if len(recs) > 1 else np.zeros_like(g)


def ensemble_muH(recs, stat="median"):
    g = recs[0]["gamma"][1:]
    M = np.array([r["muH"][1:] / r["muH"][0] for r in recs])
    c = np.median(M, axis=0) if stat == "median" else M.mean(axis=0)
    return g, c, M.std(axis=0, ddof=1) / np.sqrt(len(recs)) if len(recs) > 1 else np.zeros_like(g)


def ensemble_secant(recs, stat="median"):
    g = recs[0]["gamma"][1:]
    M = np.array([secant(r)[1] for r in recs])
    c = np.median(M, axis=0) if stat == "median" else M.mean(axis=0)
    return g, c


def ensemble_stress(recs):
    """<sigma>(gamma) / (<mu0> gamma): обе величины самоусредняющиеся, без деления на шумный mu0 образца."""
    g = recs[0]["gamma"][1:]
    S = np.mean([r["sigma"][1:] for r in recs], axis=0)
    mu0 = np.mean([r["mu0"] for r in recs])
    return g, S / g / mu0


def report_curves(recs, Ns, dphis, getter, title):
    print(f"\n  --- {title} ---")
    print("  Масштабы gamma_x, на которых кривая теряет 5, 10, 30, 50 % модуля; nu — показатель 1 - mu/mu0 ~ gamma^nu")
    print("  на участке 0.02 < 1 - mu/mu0 < 0.3; далее 1 - mu/mu0 при gamma = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2.\n")
    print(f"  {'N':>5} {'dphi':>7} {'n':>3} {'dz':>6} {'g_5%':>9} {'g_10%':>9} {'g_30%':>9} {'g_50%':>9} {'nu':>6}   1-mu/mu0")
    table = {}
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["dphi"] == d and r["N"] == N]
            if not rr:
                continue
            g, m = getter(rr)
            gx = [crossover(g, m, lv) for lv in (0.05, 0.1, 0.3, 0.5)]
            dd = 1 - m
            msk = (dd > 0.02) & (dd < 0.3) & (g > 0)
            nu = loglog_slope(g[msk], dd[msk]) if msk.sum() >= 3 else np.nan
            dz = np.mean([r["dz"] for r in rr])
            vals = [np.interp(x, g, dd) for x in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)]
            table[(N, d)] = dict(dz=dz, gx=gx, nu=nu, n=len(rr))
            print(f"  {N:>5} {d:>7g} {len(rr):>3} {dz:>6.3f} {gx[0]:>9.2e} {gx[1]:>9.2e} {gx[2]:>9.2e} {gx[3]:>9.2e} {nu:>6.2f}   "
                  + "  ".join(f"{v:+.3f}" for v in vals))
    print("  скейлинг gamma_x ~ dz^x (по N):")
    for N in Ns:
        keys = [k for k in table if k[0] == N]
        if len(keys) < 2:
            continue
        dz = np.array([table[k]["dz"] for k in keys])
        xs = []
        for lv in range(4):
            gx = np.array([table[k]["gx"][lv] for k in keys])
            xs.append(loglog_slope(dz, gx))
        print(f"    N={N}: x(5%)={xs[0]:.2f}  x(10%)={xs[1]:.2f}  x(30%)={xs[2]:.2f}  x(50%)={xs[3]:.2f}")
    return table


def analyse(recs, hertz=False):
    head("СВОДКА: закон размягчения на заклиненном образце")
    dphis0 = sorted(set(r["dphi"] for r in recs)); Ns0 = sorted(set(r["N"] for r in recs))
    if "muH" in recs[0]:
        report_curves(recs, Ns0, dphis0, lambda rr: ensemble_muH(rr, "median")[:2],
                      "Гессианов (касательный упругий) модуль деформированного состояния mu_H(gamma)/mu_H(0): МЕДИАНА по образцам")
        report_curves(recs, Ns0, dphis0, lambda rr: ensemble_muH(rr, "mean")[:2],
                      "mu_H(gamma)/mu_H(0): СРЕДНЕЕ по образцам")
        report_curves(recs, Ns0, dphis0, lambda rr: ensemble_secant(rr, "median"),
                      "Секущий модуль sigma/(mu0 gamma): МЕДИАНА по образцам")
        report_curves(recs, Ns0, dphis0, ensemble_stress,
                      "Средняя кривая напряжения <sigma>/(<mu0> gamma) — макроскопический секущий модуль")
        # потеря контактов
        print("\n  --- Потеря контактов: <z(gamma) - z(0)> при gamma = 1e-4, 1e-3, 1e-2, 3e-2 ---")
        for N in Ns0:
            for d in dphis0:
                rr = [r for r in recs if r["dphi"] == d and r["N"] == N]
                if not rr:
                    continue
                g = rr[0]["gamma"]
                dzg = np.mean([r["zg"] - r["zg"][0] for r in rr], axis=0)
                print(f"    N={N} dphi={d:<6g}: " + "  ".join(f"{np.interp(x, g, dzg):+.3f}" for x in (1e-4, 1e-3, 1e-2, 3e-2)))
    dphis = sorted(set(r["dphi"] for r in recs))
    Ns = sorted(set(r["N"] for r in recs))
    print(f"\n  Образцов: {len(recs)}; N = {Ns}; dphi = {dphis}")

    print("\n  --- Линейные свойства (среднее ± ст.откл. по образцам) ---")
    print(f"  {'N':>5} {'dphi':>7} {'n':>3} {'dz':>14} {'p':>10} {'G/B':>16} {'mu0/mu_aff':>16} {'phi_J':>8}")
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["dphi"] == d and r["N"] == N]
            if not rr:
                continue
            dz = np.array([r["dz"] for r in rr]); GB = np.array([r["mu0"] / r["B"] for r in rr])
            na = np.array([r["mu0"] / r["mu_aff"] for r in rr]); pr = np.array([r["p"] for r in rr])
            pj = np.array([r["phi_j"] for r in rr])
            print(f"  {N:>5} {d:>7g} {len(rr):>3} {dz.mean():>7.3f}±{dz.std():<6.3f} {pr.mean():>10.3e} "
                  f"{GB.mean():>8.4f}±{GB.std():<7.4f} {na.mean():>8.4f}±{na.std():<7.4f} {pj.mean():>8.4f}")

    print("\n  --- Ансамблевая кривая <sigma/(mu0 gamma)> (секущий модуль, нормированный на свой mu0) ---")
    print("  Масштабы gamma_x, на которых средняя кривая теряет 5, 10, 30, 50 % модуля; показатель nu при малых gamma")
    print("  (подгонка 1 - mu/mu0 ~ gamma^nu на участке 0.02 < 1 - mu/mu0 < 0.3); текучесть = максимум средней sigma.\n")
    print(f"  {'N':>5} {'dphi':>7} {'n':>3} {'dz':>6} {'g_5%':>9} {'g_10%':>9} {'g_30%':>9} {'g_50%':>9} {'nu':>6} "
          f"{'g_y':>8} {'mu_s(g_y)':>10}   1-mu/mu0 при gamma = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2")
    table = {}
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["dphi"] == d and r["N"] == N]
            if not rr:
                continue
            g, m, se = ensemble(rr)
            gx = [crossover(g, m, lv) for lv in (0.05, 0.1, 0.3, 0.5)]
            dd = 1 - m
            msk = (dd > 0.02) & (dd < 0.3)
            nu = loglog_slope(g[msk], dd[msk]) if msk.sum() >= 3 else np.nan
            # текучесть по средней sigma/mu0
            gf = rr[0]["gamma_full"]; sf = np.mean([r["sigma_full"] / r["mu0"] for r in rr], axis=0)
            ky = int(np.argmax(sf)); gy = gf[ky]; msy = sf[ky] / gy
            dz = np.mean([r["dz"] for r in rr])
            vals = [np.interp(x, g, dd) for x in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)]
            table[(N, d)] = dict(dz=dz, gx=gx, nu=nu, gy=gy, msy=msy, n=len(rr))
            print(f"  {N:>5} {d:>7g} {len(rr):>3} {dz:>6.3f} {gx[0]:>9.2e} {gx[1]:>9.2e} {gx[2]:>9.2e} {gx[3]:>9.2e} {nu:>6.2f} "
                  f"{gy:>8.4f} {msy:>10.3f}   " + "  ".join(f"{v:+.3f}" for v in vals))

    print("\n  --- Скейлинг масштабов по dz (по N отдельно; показатель x в gamma_x ~ dz^x) ---")
    for N in Ns:
        keys = [k for k in table if k[0] == N]
        dz = np.array([table[k]["dz"] for k in keys])
        for lv, name in ((0, "5%"), (1, "10%"), (2, "30%"), (3, "50%")):
            gx = np.array([table[k]["gx"][lv] for k in keys])
            print(f"    N={N}: уровень {name:>4}: x = {loglog_slope(dz, gx):.2f}")
        gy = np.array([table[k]["gy"] for k in keys])
        print(f"    N={N}: текучесть g_y ~ dz^{loglog_slope(dz, gy):.2f}")

    print("\n  --- Зависимость от размера при фиксированном dphi (масштаб 10 %-ной потери) ---")
    for d in dphis:
        row = [(N, table[(N, d)]["gx"][1], table[(N, d)]["n"]) for N in Ns if (N, d) in table]
        print(f"    dphi={d:<7g}: " + "; ".join(f"N={N}: {gx:.2e} (n={n})" for N, gx, n in row))

    print("\n  --- Ансамблевые кривые целиком: gamma, <mu_s/mu0> ± ст.ошибка ---")
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["dphi"] == d and r["N"] == N]
            if not rr:
                continue
            g, m, se = ensemble(rr)
            print(f"  N={N} dphi={d} (n={len(rr)}):")
            print("    " + " ".join(f"{x:.1e}:{y:.3f}±{z:.3f}" for x, y, z in zip(g, m, se)))

    print("""
  Чтение.
  * gamma_x — масштаб, на котором заклиненный образец теряет заданную долю
    модуля. Если он стремится к нулю при dz -> 0 (x > 0), у критической
    точки НЕТ конечного универсального масштаба амплитуды: размягчение
    начинается сколь угодно рано, а глубина размягчения при данной
    амплитуде — функция dz (и N), а не универсальное число.
  * nu — показатель при малых gamma: nu = 2 отвечал бы аналитическому
    квартичному члену (постановка T-44/T-34), nu ~ 1 — размыканию
    контактов (число разомкнутых контактов ~ gamma), nu < 1 — лавинам.
  * mu_s(g_y) — доля модуля, оставшаяся к пределу текучести; плато
    (delta_max) было бы горизонтальным участком 1 - mu/mu0 ниже текучести.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--dphi", type=str, default="1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--gmax", type=float, default=0.1)
    ap.add_argument("--hertz", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="N=256, одна затравка, два dphi")
    ap.add_argument("--analyse", type=str, default="", help="только сводка по готовым npz (через запятую)")
    args = ap.parse_args()

    if args.analyse:
        recs = []
        for f in args.analyse.split(","):
            recs += list(np.load(f, allow_pickle=True)["recs"])
        analyse(recs)
        return

    if args.quick:
        args.N, args.seeds, args.dphi = 256, "1", "3e-3,3e-2"
    seeds = [int(s) for s in args.seeds.split(",")]
    dphis = [float(s) for s in args.dphi.split(",")]
    head(f"ЗАКЛИНЕННЫЙ ОБРАЗЕЦ: N={args.N}, затравки {seeds}, dphi={dphis}, "
         f"{'Герц' if args.hertz else 'гармонический контакт'}, gmax={args.gmax}")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.hertz, args.gmax)
        out = args.out or os.path.join(HERE, f"jammed_softening_N{args.N}{'_hertz' if args.hertz else ''}.npz")
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs, hertz=args.hertz)


if __name__ == "__main__":
    main()
