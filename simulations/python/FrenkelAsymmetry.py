"""Асимметрия пары Френкеля: вакансия против межузельности на спаянной упаковке (T-27, к10).

Постановка (развилка о зеркале, к10). Если C-сопряжение = «ячейка <-> дырка» (ветвь з-а),
то тождественность частицы и античастицы (|m_e+ - m_e-|/m < 8e-9) требует, чтобы работа
на создание/удержание вакансии и межузельности была одинакова с большой точностью — либо
чтобы энергия дефектного слоя была пренебрежима против энергии узла. В обычных кристаллах
формовочные энергии вакансии и межузельности различаются в разы. Вопрос: как у спаянной
заклиненной упаковки (вариант C, 1.11), и симметричен ли ДАЛЬНИЙ отклик (амплитуда
радиального поля ~ «модуль заряда»), который в несжимаемом пределе мог бы быть защищён
сохранением объёма даже при асимметричных кернах.

Меряется на спаянной сети (двусторонние связи по замороженному списку контактов
+ односторонние прочие пары; сдвигово-стабилизированный образец, phi = phi_J + dphi):
  1. E_f^vac(sp)  = E(N-1, релакс) + eps_sp - E0   — формовочная энергия вакансии;
  2. E_f^int(sp)  = E(N+1, релакс) - eps_sp - E0   — формовочная энергия межузельности
     (вставка в наибольшую пустоту; частица без связей — контакты односторонние: по
     постулату 1.5.2(5) они замкнутся в связи, энергия релаксированного состояния та же);
  3. E_F = E_f^vac + E_f^int — энергия пары Френкеля (без химпотенциала: eps сокращается);
  4. дальнее поле: A(r) = <u_r> r^2 по оболочкам (для центра дилатации u_r = A/r^2 = const);
     поток Phi = 4 pi A против объёма частицы v0; асимметрия |A_int/A_vac|;
  5. сдвиг давления Delta p для обоих знаков дефекта.
eps_sp — доля энергии образца на частицу данного сорта (полусумма энергий её пар), из E0.

Запуск: python FrenkelAsymmetry.py [--N 512] [--seeds 1,2] [--dphi 3e-3,3e-2]
                                   [--nvac 8] [--nint 4] [--out frenkel_asym_N512.npz]
        python FrenkelAsymmetry.py --analyse frenkel_asym_N512.npz
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402


class DefectNet(PinnedSpheres):
    """Спаянная сеть: пары из замороженного списка — двусторонние пружины 1/2 (1-r/sig)^2
    при любом знаке перекрытия; все прочие пары — односторонний мягкий контакт (как у
    SoftSpheres). Число частиц произвольное (вакансия: N-1, межузельность: N+1)."""

    @classmethod
    def build(cls, rad, L, s, G3, bi, bj, rc):
        o = cls(8, 0.65, 0)
        o.N = len(rad)
        o.rad = rad.copy()
        o.vol_particles = (np.pi / 6.0 * (2 * rad) ** 3).sum()
        o.L = L
        o.s = s.copy()
        o.G3 = G3.copy()
        o.fixed = np.zeros(o.N, bool)
        o.rc = rc
        o.bi, o.bj = bi.copy(), bj.copy()          # bi < bj
        o.bsig = rad[o.bi] + rad[o.bj]
        o._bkey = np.sort(o.bi.astype(np.int64) * o.N + o.bj)
        o._pairs = None
        o._s_at_build = None
        o._G3_at_build = None
        return o

    def build_pairs(self):
        super().build_pairs()
        key = self._i.astype(np.int64) * self.N + self._j
        m = ~np.isin(key, self._bkey)
        self._pairs = self._pairs[m]
        self._i, self._j = self._pairs[:, 0], self._pairs[:, 1]
        self._sig = self.rad[self._i] + self.rad[self._j]

    def _geom_b(self, s):
        ds = s[self.bj] - s[self.bi]
        ds -= self.L * np.round(ds / self.L)
        dx = ds @ self.F().T
        r = np.linalg.norm(dx, axis=1)
        return ds, dx, r

    def energy_grad(self, sflat):
        s = sflat.reshape(self.N, 3)
        # односторонняя часть (не-связи)
        ds, dx, r = self._geom(s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        E = 0.5 * (ov[m] ** 2).sum()
        fac = np.zeros_like(r)
        fac[m] = -ov[m] / self._sig[m] / r[m]
        f = fac[:, None] * dx
        gx = np.stack([np.bincount(self._j, f[:, c], self.N) - np.bincount(self._i, f[:, c], self.N)
                       for c in range(3)], axis=1)
        # связи (двусторонние)
        dsb, dxb, rb = self._geom_b(s)
        ovb = 1.0 - rb / self.bsig
        E += 0.5 * (ovb ** 2).sum()
        facb = -ovb / self.bsig / rb
        fb = facb[:, None] * dxb
        gx += np.stack([np.bincount(self.bj, fb[:, c], self.N) - np.bincount(self.bi, fb[:, c], self.N)
                        for c in range(3)], axis=1)
        gs = gx @ self.F()
        return E, gs.ravel()

    def force_scale(self):
        if self.need_rebuild():
            self.build_pairs()
        ds, dx, r = self._geom(self.s)
        ov = 1.0 - r / self._sig
        parts = [np.abs(ov[ov > 0]) / self._sig[ov > 0]] if (ov > 0).any() else []
        dsb, dxb, rb = self._geom_b(self.s)
        ovb = 1.0 - rb / self.bsig
        parts.append(np.abs(ovb) / self.bsig)
        allf = np.concatenate(parts)
        return float(allf.mean()) + 1e-300

    def pressure(self):
        if self.need_rebuild():
            self.build_pairs()
        ds, dx, r = self._geom(self.s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        V = self.L ** 3
        p = ((ov[m] / self._sig[m]) * r[m]).sum() / (3 * V)
        dsb, dxb, rb = self._geom_b(self.s)
        ovb = 1.0 - rb / self.bsig
        p += ((ovb / self.bsig) * rb).sum() / (3 * V)
        return p

    def pair_energy_per_particle(self):
        """Доля энергии на частицу: полусумма энергий инцидентных пар."""
        e = np.zeros(self.N)
        ds, dx, r = self._geom(self.s)
        ov = 1.0 - r / self._sig
        m = ov > 0
        ee = 0.5 * ov[m] ** 2
        np.add.at(e, self._i[m], 0.5 * ee)
        np.add.at(e, self._j[m], 0.5 * ee)
        dsb, dxb, rb = self._geom_b(self.s)
        eb = 0.5 * (1.0 - rb / self.bsig) ** 2
        np.add.at(e, self.bi, 0.5 * eb)
        np.add.at(e, self.bj, 0.5 * eb)
        return e

    def new_contacts(self):
        """Замкнувшиеся НЕ-связевые контакты (стали бы новыми связями по 1.5.2(5))."""
        ds, dx, r = self._geom(self.s)
        return int((1.0 - r / self._sig > 0).sum())


def vacancy_net(host, k):
    keep = np.ones(host.N, bool)
    keep[k] = False
    idx = np.cumsum(keep) - 1                     # старый -> новый индекс
    mb = (host.bi != k) & (host.bj != k)
    bi = idx[host.bi[mb]]
    bj = idx[host.bj[mb]]
    return DefectNet.build(host.rad[keep], host.L, host.s[keep], host.G3, bi, bj, host.rc)


def interstitial_net(host, pos, rad_new):
    rad = np.concatenate([host.rad, [rad_new]])
    s = np.vstack([host.s, pos[None, :]])
    return DefectNet.build(rad, host.L, s, host.G3, host.bi, host.bj, host.rc)


def frenkel_net(host, k, pos):
    """Прямая пара Френкеля: частица k вырвана из своего места (её связи разорваны —
    это и есть акт рождения, к4) и поставлена в пустоту той же коробки. N и V те же,
    резервуар не нужен: E_F = E(релакс) - E0 — безусловная энергия пары."""
    mb = (host.bi != k) & (host.bj != k)
    s = host.s.copy()
    s[k] = pos
    return DefectNet.build(host.rad, host.L, s, host.G3, host.bi[mb], host.bj[mb], host.rc)


def find_voids(host, nsites, min_sep=2.0, ngrid=32, seed=0):
    """Крупная сетка + локальный дожим: максимумы зазора g(x)=min_i(|x-x_i|-rad_i)."""
    L = host.L
    s = np.mod(host.s, L)
    tree = cKDTree(s, boxsize=L)

    def gap(x):
        d, i = tree.query(np.mod(x, L), k=12)
        return (d - host.rad[i]).min(axis=-1)

    g1 = np.linspace(0, L, ngrid, endpoint=False)
    X, Y, Z = np.meshgrid(g1, g1, g1, indexing="ij")
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    gv = gap(pts)
    order = np.argsort(gv)[::-1]
    sites = []
    for o in order:
        x = pts[o]
        ok = True
        for xs, _ in sites:
            d = x - xs
            d -= L * np.round(d / L)
            if np.linalg.norm(d) < min_sep:
                ok = False
                break
        if not ok:
            continue
        # локальный дожим покоординатным поиском
        step = L / ngrid
        for _ in range(40):
            improved = False
            g0 = gap(x[None, :])[0]
            for c in range(3):
                for sgn in (+1, -1):
                    xt = x.copy()
                    xt[c] += sgn * step
                    if gap(xt[None, :])[0] > g0:
                        x = xt
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        sites.append((np.mod(x, L), gap(x[None, :])[0]))
        if len(sites) >= nsites:
            break
    return sites


def far_field(host, net, center_s, removed=None):
    """Оболочечный профиль A(r) = <u_r> r^2 (реальные координаты, минимальный образ)."""
    F = host.F()
    keep = np.ones(host.N, bool)
    if removed is not None:
        keep[removed] = False
    s0 = host.s[keep]
    s1 = net.s[: keep.sum()] if removed is None else net.s
    if removed is None:
        s1 = net.s[: host.N][keep]
    d = s1 - s0
    d -= host.L * np.round(d / host.L)
    ux = d @ F.T
    rel = s0 - center_s[None, :]
    rel -= host.L * np.round(rel / host.L)
    rx = rel @ F.T
    r = np.linalg.norm(rx, axis=1)
    ur = (ux * rx).sum(axis=1) / np.maximum(r, 1e-12)
    edges = np.arange(1.0, host.L / 2, 0.6)
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() < 4:
            prof.append((0.5 * (a + b), np.nan, int(m.sum())))
            continue
        rc = r[m].mean()
        prof.append((rc, float((ur[m] * r[m] ** 2).mean()), int(m.sum())))
    return prof


def run_sample(N, seed, dphis, nvac, nint, verbose=True):
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
        p.G3[:] = 0.0
        p.set_phi(phi_j + dphi)
        p.stabilize()
        i0, j0, *_ = p.contacts()
        host = DefectNet.build(p.rad, p.L, p.s, p.G3, i0, j0, p.rc)
        E0 = host.minimize(rel=1e-10)
        p0_pr = host.pressure()
        eperp = host.pair_energy_per_particle()
        small = host.rad < 0.6
        eps_sp = {0: eperp[small].mean(), 1: eperp[~small].mean()}
        z = 2 * (len(i0)) / N
        print(f"    dphi={dphi:g}: dz={z - Z_C:+.3f}  E0={E0:.4e}  E0/N={E0 / N:.3e}  p0={p0_pr:.3e}  "
              f"eps(мал)={eps_sp[0]:.3e} eps(бол)={eps_sp[1]:.3e}  [{time.time() - t0:.0f} с]", flush=True)
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, dz=z - Z_C, E0=E0, p0=p0_pr,
                   eps_small=eps_sp[0], eps_large=eps_sp[1], L=host.L,
                   vac=[], intr=[])
        # --- вакансии ---
        for sp, spname in ((0, "мал"), (1, "бол")):
            cand = np.where(small if sp == 0 else ~small)[0]
            for k in rng.choice(cand, nvac // 2, replace=False):
                net = vacancy_net(host, int(k))
                Ev = net.minimize(rel=1e-10)
                dE = Ev - E0
                Ef = dE + eps_sp[sp]
                dp = net.pressure() - p0_pr
                prof = far_field(host, net, host.s[int(k)], removed=int(k))
                Amid = np.nanmedian([a for _, a, n in prof if np.isfinite(a)])
                nc = net.new_contacts()
                rec["vac"].append(dict(k=int(k), sp=sp, dE=dE, Ef=Ef, dp=dp, A=Amid, prof=prof, nc=nc))
                print(f"      вакансия ({spname}) k={k}: dE={dE:+.4e}  E_f={Ef:+.4e}  "
                      f"dp={dp:+.2e}  A={Amid:+.3e}  новых контактов {nc}", flush=True)
        # --- межузельности ---
        sites = find_voids(host, nint)
        for sp, rnew, spname in ((0, 0.5, "мал"), (1, 0.7, "бол")):
            for pos, g in sites[: max(1, nint // 2)]:
                net = interstitial_net(host, pos, rnew)
                Ei = net.minimize(rel=1e-10)
                dE = Ei - E0
                Ef = dE - eps_sp[sp]
                dp = net.pressure() - p0_pr
                prof = far_field(host, net, np.mod(net.s[-1], host.L))
                Amid = np.nanmedian([a for _, a, n in prof if np.isfinite(a)])
                nc = net.new_contacts()
                rec["intr"].append(dict(sp=sp, gap=g, dE=dE, Ef=Ef, dp=dp, A=Amid, prof=prof, nc=nc))
                print(f"      межузел. ({spname}, r={rnew}) зазор {g:.2f}: dE={dE:+.4e}  E_f={Ef:+.4e}  "
                      f"dp={dp:+.2e}  A={Amid:+.3e}  контактов у вставки {nc}", flush=True)
        # --- прямая пара Френкеля (без резервуара): k -> пустота той же коробки ---
        rec["pair"] = []
        for sp, spname in ((0, "мал"), (1, "бол")):
            cand = np.where(small if sp == 0 else ~small)[0]
            for pos, g in sites[:2]:
                # частица того же сорта, дальше всего от места вставки
                rel = host.s[cand] - pos[None, :]
                rel -= host.L * np.round(rel / host.L)
                k = int(cand[np.argmax(np.linalg.norm(rel, axis=1))])
                net = frenkel_net(host, k, pos)
                EF = net.minimize(rel=1e-10) - E0
                rec["pair"].append(dict(sp=sp, k=k, gap=g, EF=EF))
                print(f"      пара Френкеля ({spname}) k={k} -> зазор {g:.2f}: E_F={EF:+.4e}", flush=True)
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА: асимметрия пары Френкеля на спаянной упаковке (к10)")
    dphis = sorted(set(r["dphi"] for r in recs))
    print(f"  образцов: {len(recs)}; N={sorted(set(r['N'] for r in recs))}; dphi={dphis}")
    print("  E_f — формовочные энергии (резервуар eps_sp); E_F = E_f^vac + E_f^int — пара Френкеля;")
    print("  A — амплитуда дальнего поля <u_r> r^2 (знак: вакансия <0 — втягивание, межузельность >0);")
    print("  v0/4pi — «голая» амплитуда: объём частицы/(4 pi).\n")
    for d in dphis:
        rr = [r for r in recs if r["dphi"] == d]
        dz = np.mean([r["dz"] for r in rr])
        eps0 = np.mean([r["eps_small"] for r in rr])
        print(f"  ==== dphi={d:g} (dz={dz:+.3f}; eps на частицу(мал) = {eps0:.3e}) ====")
        for sp, spname, rnew in ((0, "малая", 0.5), (1, "большая", 0.7)):
            Efv = np.array([v["Ef"] for r in rr for v in r["vac"] if v["sp"] == sp])
            Efi = np.array([v["Ef"] for r in rr for v in r["intr"] if v["sp"] == sp])
            dEv = np.array([v["dE"] for r in rr for v in r["vac"] if v["sp"] == sp])
            dEi = np.array([v["dE"] for r in rr for v in r["intr"] if v["sp"] == sp])
            Av = np.array([v["A"] for r in rr for v in r["vac"] if v["sp"] == sp])
            Ai = np.array([v["A"] for r in rr for v in r["intr"] if v["sp"] == sp])
            v0 = 4 * np.pi / 3 * rnew ** 3
            EF = np.median(Efv) + np.median(Efi)
            p0v = np.mean([r["p0"] for r in rr]) * v0
            EFd = np.array([v["EF"] for r in rr for v in r.get("pair", []) if v["sp"] == sp])
            print(f"    -- сорт: {spname} (r={rnew}, v0/4pi={v0 / (4 * np.pi):.3f}, p0*v0={p0v:.2e}) --")
            print(f"       E_f^vac = {np.median(Efv):+.4e} (медиана; {Efv.min():+.2e}..{Efv.max():+.2e}; dE {np.median(dEv):+.2e}; с pV: {np.median(Efv) + p0v:+.2e})")
            print(f"       E_f^int = {np.median(Efi):+.4e} (медиана; {Efi.min():+.2e}..{Efi.max():+.2e}; dE {np.median(dEi):+.2e}; с pV: {np.median(Efi) - p0v:+.2e})")
            print(f"       E_F пары (сумма формовочных) = {EF:+.4e};  ПРЯМАЯ E_F (перенос в той же коробке) = "
                  f"{np.median(EFd) if len(EFd) else np.nan:+.4e} ({len(EFd)} изм.)")
            print(f"       асимметрия E_f^int/E_f^vac = {np.median(Efi) / np.median(Efv) if np.median(Efv) != 0 else np.nan:+.2f}; "
                  f"доля межузельности в прямой паре ~ {np.median(Efi - p0v) / np.median(EFd) if len(EFd) and np.median(EFd) != 0 else np.nan:.2f}")
            print(f"       A_vac = {np.median(Av):+.3e}  A_int = {np.median(Ai):+.3e}  |A_int/A_vac| = {abs(np.median(Ai) / np.median(Av)) if np.median(Av) != 0 else np.nan:.3f}")
            print(f"       |A|/(v0/4pi): vac {abs(np.median(Av)) / (v0 / (4 * np.pi)):.3f}, int {abs(np.median(Ai)) / (v0 / (4 * np.pi)):.3f}")
    print("""
  Чтение.
  * E_f^int >> E_f^vac (в разы и больше) — генерический кристалльный итог: керны разноприродны;
    для ветви (з-а) это значит, что вырождение e+/e- может держаться ТОЛЬКО если энергия
    дефектного слоя не входит в массу (масса целиком у узла) — либо (з-а) неверна.
  * |A_int/A_vac| — асимметрия «модуля заряда»: если она заметно отличается от 1 при
    конечной сжимаемости, то равенство |e+|=|e-| требует топологической защиты потока
    (несжимаемый предел), а не энергетической.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="1,2")
    ap.add_argument("--dphi", type=str, default="3e-3,3e-2")
    ap.add_argument("--nvac", type=int, default=8)
    ap.add_argument("--nint", type=int, default=4)
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
    head(f"Асимметрия пары Френкеля: N={args.N}, затравки {seeds}, dphi={dphis}")
    out = args.out or os.path.join(HERE, f"frenkel_asym_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis, args.nvac, args.nint)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
