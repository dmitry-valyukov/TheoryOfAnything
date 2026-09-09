"""T-9: коэффициент двойного лучепреломления k_CM на заклиненном образце.

Постановка (10.2). В модели магнитное поле — вмороженное кручение ячеек, B = rot A,
A = sqrt(mu0_магн * G) u (3.5), т.е. однородное статическое B ~ вмороженный простой
сдвиг gamma0 = |rot u| среды. Под предсдвигом изотропная среда становится анизотропной:
модуль сдвига для плоскостей, содержащих ось B (mu_par: xz, yz при B || z), и для
плоскости, перпендикулярной B (mu_perp: xy), расходятся, а с ними скорости света двух
поляризаций: Delta n = n_par - n_perp ~ (mu_perp - mu_par)/(2 mu). По симметрии
(поворот на pi вокруг x меняет знак gamma0, но не диагональные модули) расщепление
чётно по gamma0, т.е. ~ gamma0^2 = ~ B^2 — форма Коттона — Мутона.

Здесь измеряется безразмерный коэффициент kappa в
    Delta n = kappa * gamma0^2,     kappa = (mu_perp - mu_par) / (2 mu0 gamma0^2),
на сдвигово-стабилизированном заклиненном образце (мягкие сферы, гармонический контакт):
ячейке придаётся сдвиг gamma0 (G3 = (gamma0, 0, 0): u_x = gamma0 y, rot u = -gamma0 z),
частицы полностью релаксируют (контакты могут размыкаться — считается число изменённых
контактов), затем точно, по матрице Гессе, вычисляются шесть модулей простого сдвига
mu_ab текущего состояния; mu_par = (mu_xz + mu_zx + mu_yz + mu_zy)/4, mu_perp = (mu_xy + mu_yx)/2.
Оба знака gamma0 усредняются (нечётная часть — артефакт образца).

Перевод в единицы Тл^-2. B = sqrt(mu0_магн G) gamma0 c G = mu = rho c^2 модулем сдвига
вакуума; при планковской ячейке (1.8, 4.14.9) mu = hbar c / a_Pl^4 ~ 4.6e113 Па, откуда
    k_CM = kappa / (mu0_магн mu)  [Тл^-2]
и сравнивается с КЭД (Эйлер — Гейзенберг): k_CM^QED = 4e-24 Тл^-2 (10.2). Печатается также
модуль mu, при котором механический k_CM сравнялся бы с КЭД.

Запуск:  python Birefringence.py [--N 512] [--seeds 1,2,3] [--dphi 3e-3,1e-2,3e-2,1e-1]
                                 [--out birefringence_N512.npz]
         python Birefringence.py --analyse birefringence_N512.npz
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres, SHEARS  # noqa: E402
from JitterSoftening import moduli  # noqa: E402

G_GRID = np.geomspace(1e-5, 3e-2, 8)
IDX = {ab: k for k, ab in enumerate(SHEARS)}
PAR = [IDX[(0, 2)], IDX[(2, 0)], IDX[(1, 2)], IDX[(2, 1)]]
PERP = [IDX[(0, 1)], IDX[(1, 0)]]

class FrozenNet(PinnedSpheres):
    """Тот же образец с ЗАМОРОЖЕННЫМ набором контактов: каждый контакт — двусторонняя
    гармоническая пружина (перекрытие может стать отрицательным). Даёт гладкую,
    аналитическую часть отклика на предсдвиг (третьего порядка упругость сети)."""

    def freeze(self, i, j):
        self._fi, self._fj = i.copy(), j.copy()
        self._pairs = np.stack([i, j], 1); self._i, self._j = i, j
        self._sig = self.rad[i] + self.rad[j]
        self._s_at_build = self.s.copy(); self._gamma_at_build = self.gamma
        self._G3_at_build = self.G3.copy()

    def need_rebuild(self):
        return False

    def build_pairs(self):
        self._pairs = np.stack([self._fi, self._fj], 1); self._i, self._j = self._fi, self._fj
        self._sig = self.rad[self._i] + self.rad[self._j]
        self._s_at_build = self.s.copy(); self._gamma_at_build = self.gamma
        self._G3_at_build = self.G3.copy()

    def energy_grad(self, sflat):
        s = sflat.reshape(self.N, 3)
        ds, dx, r = self._geom(s)
        ov = 1.0 - r / self._sig
        E = 0.5 * (ov ** 2).sum()
        fac = -ov / self._sig / r
        f = fac[:, None] * dx
        gx = np.stack([np.bincount(self._j, f[:, c], self.N) - np.bincount(self._i, f[:, c], self.N)
                       for c in range(3)], axis=1)
        gs = gx @ self.F()
        return E, gs.ravel()

    def force_scale(self):
        ds, dx, r = self._geom(self.s)
        ov = 1.0 - r / self._sig
        return float((np.abs(ov) / self._sig).mean()) + 1e-300


MU0_MAG = 4e-7 * np.pi
HBAR, C_LIGHT, L_PL = 1.054571817e-34, 2.99792458e8, 1.616255e-35
MU_VAC = HBAR * C_LIGHT / L_PL ** 4          # rho c^2 при планковской ячейке, Па
K_CM_QED = 4e-24


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
        i0, j0, *_ = p.contacts(); cs0 = set(zip(i0.tolist(), j0.tolist()))
        mu_dir0 = moduli(p); mu0 = mu_dir0.mean()
        rec = dict(N=N, seed=seed, dphi=dphi, phi_j=phi_j, p=pr, z=z, dz=z - Z_C, mu0=mu0,
                   mu_dir0=mu_dir0, g=G_GRID, mus=np.zeros((len(G_GRID), 2, 6)), dnc=np.zeros((len(G_GRID), 2)),
                   E=np.zeros((len(G_GRID), 2)), musF=np.zeros((len(G_GRID), 2, 6)))
        E0 = p.energy()
        # замороженная сеть: копия образца с фиксированными контактами
        pf = FrozenNet(N, 0.65, seed); pf.rad = p.rad.copy(); pf.vol_particles = p.vol_particles
        pf.rc = p.rc; pf.L = p.L; pf.s = s0.copy(); pf.G3 = G0.copy(); pf.freeze(i0, j0)
        pf.minimize(rel=1e-10)
        sF0 = pf.s.copy()
        muF0 = moduli(pf, (i0, j0)); rec["muF_dir0"] = muF0
        print(f"    dphi={dphi:<6g} p={pr:.3e} dz={z-Z_C:+.3f} mu0={mu0:.4e} "
              f"(xy {mu_dir0[PERP].mean():.4e}, xz/yz {mu_dir0[PAR].mean():.4e})  [{time.time()-t0:.0f} с]", flush=True)
        for kg, g in enumerate(G_GRID):
            for ks, sgn in enumerate((+1, -1)):
                p.s = s0.copy(); p.G3 = G0.copy(); p.G3[0] += sgn * g; p._pairs = None
                rec["E"][kg, ks] = p.minimize(rel=1e-9) - E0
                rec["mus"][kg, ks] = moduli(p)
                i1, j1, *_ = p.contacts()
                rec["dnc"][kg, ks] = len(cs0 ^ set(zip(i1.tolist(), j1.tolist())))
                pf.s = sF0.copy(); pf.G3 = G0.copy(); pf.G3[0] += sgn * g
                pf.minimize(rel=1e-10)
                rec["musF"][kg, ks] = moduli(pf, (i0, j0))
            m = rec["mus"][kg].mean(axis=0); mF = rec["musF"][kg].mean(axis=0)
            dn = ((m[PERP].mean() - m[PAR].mean()) - (mu_dir0[PERP].mean() - mu_dir0[PAR].mean())) / (2 * mu0)
            dnF = ((mF[PERP].mean() - mF[PAR].mean()) - (muF0[PERP].mean() - muF0[PAR].mean())) / (2 * mu0)
            print(f"      gamma0={g:.1e}: Delta n={dn:+.2e} kappa={dn/g**2:+.3e} dnc={rec['dnc'][kg].mean():.1f} | "
                  f"замороженная сеть: Delta n_F={dnF:+.2e} kappa_F={dnF/g**2:+.3e}", flush=True)
        p.s = s0.copy(); p.G3 = G0.copy(); p._pairs = None
        out.append(rec)
    return out


def analyse(recs):
    head("СВОДКА T-9: коэффициент Коттона — Мутона на заклиненном образце")
    dphis = sorted(set(r["dphi"] for r in recs)); Ns = sorted(set(r["N"] for r in recs))
    print(f"  образцов: {len(recs)}; N={Ns}; dphi={dphis}")
    print("  Delta n = (mu_perp - mu_par)/(2 mu0), kappa = Delta n / gamma0^2 (среднее по знакам gamma0 и образцам; медиана в скобках);")
    print("  dnc — изменённых контактов; kappa читается на гладком участке (dnc ~ 0).\n")
    g = recs[0]["g"]
    summary = {}
    for N in Ns:
        for d in dphis:
            rr = [r for r in recs if r["N"] == N and r["dphi"] == d]
            if not rr:
                continue
            dz = np.mean([r["dz"] for r in rr]); mu0 = np.mean([r["mu0"] for r in rr])
            print(f"  ==== N={N} dphi={d:g}: dz={dz:.3f} mu0={mu0:.3e}, образцов {len(rr)} ====")
            print("       gamma0:   " + " ".join(f"{x:9.1e}" for x in g))
            def dn_of(r, k, key, base):
                m = r[key][k].mean(axis=0); b = r[base]
                return ((m[PERP].mean() - m[PAR].mean()) - (b[PERP].mean() - b[PAR].mean())) / (2 * r["mu0"])
            DN = np.array([[dn_of(r, k, "mus", "mu_dir0") for k in range(len(g))] for r in rr])
            DNF = np.array([[dn_of(r, k, "musF", "muF_dir0") for k in range(len(g))] for r in rr])
            kapF = DNF / g[None, :] ** 2
            DNodd = np.array([[((r["mus"][k, 0][PERP].mean() - r["mus"][k, 0][PAR].mean()) - (r["mus"][k, 1][PERP].mean() - r["mus"][k, 1][PAR].mean())) / (4 * r["mu0"])
                               for k in range(len(g))] for r in rr])
            DNC = np.array([r["dnc"].mean(axis=1) for r in rr])
            kap = DN / g[None, :] ** 2
            print("       Delta n:  " + " ".join(f"{x:+9.2e}" for x in DN.mean(axis=0)))
            print("       нечётн.:  " + " ".join(f"{x:+9.2e}" for x in DNodd.mean(axis=0)))
            print("       kappa:    " + " ".join(f"{x:+9.3f}" for x in kap.mean(axis=0)))
            print("       (медиана) " + " ".join(f"{x:+9.3f}" for x in np.median(kap, axis=0)))
            print("       dnc:      " + " ".join(f"{x:9.1f}" for x in DNC.mean(axis=0)))
            print("       kappa_F:  " + " ".join(f"{x:+9.3f}" for x in kapF.mean(axis=0)) + "   (замороженная сеть, гладкая часть)")
            print("       (медиана) " + " ".join(f"{x:+9.3f}" for x in np.median(kapF, axis=0)))
            # гладкий участок: dnc == 0 у всех знаков всех образцов
            sm = (DNC.max(axis=0) == 0)
            kap_s = kap[:, sm].mean() if sm.any() else np.nan
            print(f"       kappa на гладком участке (dnc=0, {sm.sum()} точек): {kap_s:+.3f}; "
                  f"на всём диапазоне медиана {np.median(kap):+.3f}")
            kF = np.median(kapF[:, (g >= 1e-4) & (g <= 3e-3)])
            print(f"       kappa_F (медиана, 1e-4 <= gamma0 <= 3e-3): {kF:+.3f}")
            summary[(N, d)] = dict(dz=dz, kappa=kap_s, kappa_all=np.median(kap), kappaF=kF)
    print(f"\n  Перевод: mu_вак = hbar c / a_Pl^4 = {MU_VAC:.2e} Па;  B = sqrt(mu0_магн mu_вак) gamma0 = {np.sqrt(MU0_MAG*MU_VAC):.2e} Тл * gamma0")
    for k, v in summary.items():
        for nm, kap in (("kappa_F (гладкая)", v["kappaF"]), ("kappa (с перестройками, медиана)", v["kappa_all"])):
            kcm = kap / (MU0_MAG * MU_VAC)
            print(f"    N={k[0]} dphi={k[1]:g} (dz={v['dz']:.2f}) {nm}={kap:+.3e} -> k_CM = {kcm:+.2e} Тл^-2; "
                  f"КЭД {K_CM_QED:.0e}; отношение {kcm/K_CM_QED:+.1e}; mu, при котором сравнялись бы: {abs(kap)/(MU0_MAG*K_CM_QED):.1e} Па")
    print("""
  Чтение.
  * kappa — безразмерный механический коэффициент Коттона — Мутона среды; знак: kappa > 0 —
    поляризация вдоль B (модуль mu_par) медленнее.
  * k_CM = kappa/(mu0_магн mu): при планковском модуле сдвига (~1e113 Па) любой kappa ~ 1
    даёт k_CM ~ 1e-108 Тл^-2 — на 80 порядков ниже КЭД. Механический канал прогноза 10.2
    ненаблюдаем; наблюдаемое двулучепреломление вакуума в модели обязано идти через
    поляризацию узлов (щелевой диэлектрик, 4.11), т.е. по тому же каналу, что у КЭД.
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
        analyse(recs)
        return
    seeds = [int(s) for s in args.seeds.split(",")]
    dphis = [float(s) for s in args.dphi.split(",")]
    head(f"T-9: двулучепреломление под предсдвигом, N={args.N}, затравки {seeds}, dphi={dphis}")
    out = args.out or os.path.join(HERE, f"birefringence_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, dphis)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
