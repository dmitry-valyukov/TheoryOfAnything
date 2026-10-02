"""Знак и закон спада взаимодействия двух включений через дрожание среды (тест 2, 100.12.8).

Что считается. В спаянной сети ([7.9](../../theory/07-9-entropiya-cherez-razrez.md)) ставятся ДВА
локальных включения (контакты в шаре r0 вокруг каждого центра умножены на lam) на расстоянии D.
Для каждого D берутся частоты мод w_k(D) гармонического гессиана и считаются термодинамические
потенциалы дрожания:

  E_zp = (1/2) sum w                      — энергия нулевых колебаний,
  F(T) = sum [ w/2 + T ln(1 - e^{-w/T}) ] — свободная энергия,
  S(T) = sum [ (w/T)/(e^{w/T}-1) - ln(1-e^{-w/T}) ] — энтропия (наты).

Энергия взаимодействия (казимировского типа) определяется вычитанием одиночных вкладов:

  U_int(D) = Q(пара на D) - 2 Q(одиночное) + Q(без включений),   Q = E_zp, F, S.

Знак силы: f = -dU_int/dD. U_int, растущая с D, означает притяжение.
Это и есть тест 2 из 100.12.8, поставленный на силу, а не на пробный шар: знак dS и знак силы
читаются прямо, а разложение F = U - TS сразу показывает, энергетическая часть тянет или
энтропийная ([100.12.2](../../theory/100-ontologiya-energii-massy.md)).

Источник берётся двух родов, потому что знак у них разный: lam>1 — ужесточение (как в 7.9),
lam<1 — размягчение (то, что нужно гравитации, 7.4/7.7).

Единицы как в EntanglementArea.py: hbar = 1, массы 1, k_контакта = 1/sigma^2, T в долях
sqrt(<diag H>). Шум оценки: печатается контрольная разность двух эквивалентных конфигураций.

Запуск: python EntropyPairSign.py [--N 1024] [--seeds 1,2] [--lam 4.0,0.25] [--T 0,0.1]
"""

import argparse, math, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from StateDrift import BondedNet, assemble_full  # noqa: E402


def build_net(N, seed, dphi, verbose=False):
    """Та же сборка спаянной сети, что в EntanglementArea.run_sample."""
    p0, phi_j = find_phi_j(N, seed, False, verbose=verbose)
    p = PinnedSpheres(N, 0.65, seed)
    p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
    p.s, p.L = p0.s.copy(), p0.L; p.G3[:] = 0.0
    p.set_phi(phi_j + dphi); p.stabilize()
    z0, _, _ = p.coordination()
    i0, j0, *_ = p.contacts()
    b = BondedNet(N, 0.65, seed); b.rad = p.rad.copy(); b.vol_particles = p.vol_particles
    b.rc = p.rc; b.L = p.L; b.s = p.s.copy(); b.G3 = p.G3.copy(); b.bond(i0, j0)
    b.minimize(rel=1e-10)
    H0, *_ = assemble_full(b)
    return b, H0, z0 - Z_C


def hessian_with(b, centers, r0, lam):
    """Гессиан сети, где контакты с серединой в шаре r0 вокруг любого из centers умножены на lam.
    Возвращает также число задетых контактов — по нему проверяется, что шары не пересеклись."""
    i, j, dx, r, ov = b.contacts()
    x = np.mod(b.s, b.L) @ b.F().T
    mid = x[i] + 0.5 * dx
    hot = np.zeros(len(r), bool)
    for c in centers:
        dm = mid - c; dm -= b.L * np.round(dm / b.L)
        hot |= (dm ** 2).sum(axis=1) < r0 * r0
    sig = b.rad[i] + b.rad[j]
    dVdr = -ov / sig
    d2V = np.ones_like(r) / sig ** 2
    d2V[hot] *= lam; dVdr[hot] *= lam
    n = dx / r[:, None]
    K = (d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :] \
        + (dVdr / r)[:, None, None] * np.eye(3)[None]
    N = b.N
    H = np.zeros((3 * N, 3 * N))
    for a_ in range(3):
        for b_ in range(3):
            np.add.at(H, (3 * i + a_, 3 * i + b_), K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * j + b_), K[:, a_, b_])
            np.add.at(H, (3 * i + a_, 3 * j + b_), -K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * i + b_), -K[:, a_, b_])
    return H, int(hot.sum())


def freqs(H):
    w2 = np.linalg.eigvalsh(H)
    keep = w2 > 1e-8 * w2.max()
    return np.sqrt(w2[keep])


def potentials(w, Ts):
    """E_zp, и для каждой T: F и S (наты)."""
    out = {"E": 0.5 * w.sum()}
    for T in Ts:
        if T <= 0:
            out[("F", T)] = out["E"]; out[("S", T)] = 0.0
            continue
        x = w / T
        # ln(1-e^-x) устойчиво
        l = np.log1p(-np.exp(-x))
        out[("F", T)] = float(0.5 * w.sum() + T * l.sum())
        out[("S", T)] = float((x / np.expm1(x) - l).sum())
    return out


def run(N, seeds, dphi, lams, Ts, nd, ndraw, r0f):
    """Парная выборка: для каждого розыгрыша (центр + ориентация) считается вся кривая по D,
    поэтому неупорядоченность частично сокращается в форме кривой, а не только в среднем.
    U_int(D) = Q(пара c1,c2) - Q(одиночное c1) - Q(одиночное c2) + Q(без включений)."""
    rows = []
    for seed in seeds:
        t0 = time.time()
        b, H0, dz = build_net(N, seed, dphi)
        d = 2 * b.rad.mean(); L = b.L
        r0 = r0f * d
        Tabs = [T * math.sqrt(H0.diagonal().mean()) for T in Ts]
        print(f"\n  --- N={N} seed={seed} dphi={dphi:g} dz={dz:.2f} d={d:.3f} L={L:.2f} r0={r0f:g}d "
              f"T_abs={['%.3f' % t for t in Tabs]} ---", flush=True)
        base = potentials(freqs(H0), Tabs)
        Ds = np.linspace(2.0, 3.8, nd) * d
        Ds = Ds[Ds < 0.42 * L]
        rng = np.random.default_rng(9000 + seed)
        for lam in lams:
            acc = {}
            nok = {}
            for k in range(ndraw):
                cen = rng.uniform(0, L, 3)
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                for D in Ds:
                    c1 = np.mod(cen - 0.5 * D * u, L); c2 = np.mod(cen + 0.5 * D * u, L)
                    Hp, npair = hessian_with(b, [c1, c2], r0, lam)
                    H1, n1 = hessian_with(b, [c1], r0, lam)
                    H2, n2 = hessian_with(b, [c2], r0, lam)
                    if n1 + n2 != npair or n1 == 0 or n2 == 0:
                        continue
                    qp = potentials(freqs(Hp), Tabs)
                    q1 = potentials(freqs(H1), Tabs)
                    q2 = potentials(freqs(H2), Tabs)
                    for key in qp:
                        acc.setdefault((round(D / d, 3), key), []).append(qp[key] - q1[key] - q2[key] + base[key])
                    nok[round(D / d, 3)] = nok.get(round(D / d, 3), 0) + 1
                print(f"      розыгрыш {k+1}/{ndraw} (lam={lam:g}) готов [{time.time()-t0:.0f} с]", flush=True)
            for Dd in sorted(nok):
                rec = dict(N=N, seed=seed, lam=lam, Dd=Dd, ndraw=nok[Dd])
                e = acc[(Dd, "E")]
                rec["Uint_E"] = float(np.mean(e)); rec["err_E"] = float(np.std(e, ddof=1) / math.sqrt(len(e))) if len(e) > 1 else float("nan")
                msg = f"    lam={lam:g} D/d={Dd:4.2f} (n={len(e)}): E_zp={rec['Uint_E']:+.3e}±{rec['err_E']:.1e}"
                for T, t in zip(Ts, Tabs):
                    for tag, key in (("F", ("F", t)), ("S", ("S", t))):
                        v = acc[(Dd, key)]
                        rec[f"Uint_{tag}_T{T:g}"] = float(np.mean(v))
                        rec[f"err_{tag}_T{T:g}"] = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
                    msg += (f"  F(T={T:g})={rec[f'Uint_F_T{T:g}']:+.3e}±{rec[f'err_F_T{T:g}']:.1e}"
                            f"  dS(T={T:g})={rec[f'Uint_S_T{T:g}']:+.3e}±{rec[f'err_S_T{T:g}']:.1e}")
                rows.append(rec)
                print(msg, flush=True)
    return rows


def summarise(rows, Ts):
    head("СВОДКА: знак и спад взаимодействия двух включений через дрожание (тест 2, 100.12.8)")
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["N"], r["seed"], r["lam"])].append(r)
    for key, rs in sorted(g.items()):
        rs.sort(key=lambda r: r["Dd"])
        N, seed, lam = key
        print(f"\n  N={N} seed={seed} lam={lam:g}")
        Dd = np.array([r["Dd"] for r in rs])
        for name in ["Uint_E"] + [f"Uint_F_T{T:g}" for T in Ts] + [f"Uint_S_T{T:g}" for T in Ts]:
            y = np.array([r[name] for r in rs])
            ekey = "err_E" if name == "Uint_E" else name.replace("Uint_", "err_")
            noise = np.array([r.get(ekey, np.nan) for r in rs])
            sign = "+" if y[0] > 0 else "-"
            # наклон по log|y| там, где |y| выше шума
            m = np.abs(y) > 3 * np.nan_to_num(noise, nan=0.0)
            al = np.polyfit(np.log(Dd[m]), np.log(np.abs(y[m])), 1)[0] if m.sum() >= 3 else np.nan
            print(f"    {name:14s}: знак {sign}, |значение| на D=2d {abs(y[0]):.2e}, "
                  f"показатель спада {al:.2f} (точек выше шума: {int(m.sum())}/{len(y)})")
        # сила из F: f = -dF/dD, притяжение если F растёт с D
        for T in Ts:
            y = np.array([r[f"Uint_F_T{T:g}"] for r in rs])
            if len(y) >= 2:
                trend = "растёт с D (притяжение)" if y[-1] > y[0] else "падает с D (отталкивание)"
                print(f"    F(T={T:g}): {trend};  F(2d)={y[0]:+.3e} -> F({Dd[-1]:.1f}d)={y[-1]:+.3e}")
    print("""
  Чтение. U_int — взаимодействие, наведённое дрожанием (казимировского типа): знак и показатель
  спада. Притяжение означает, что свободная энергия растёт с расстоянием. Разложение F = U - TS
  показывает, энергетическая часть тянет или энтропийная; знак dS отвечает на тест 2 напрямую.
  Ничего из этого не является гравитацией: ньютоновскому 1/r соответствовал бы показатель 1.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seeds", type=str, default="1,2")
    ap.add_argument("--dphi", type=float, default=3e-2)
    ap.add_argument("--lam", type=str, default="4.0,0.25")
    ap.add_argument("--T", type=str, default="0,0.1")
    ap.add_argument("--nd", type=int, default=5)
    ap.add_argument("--ndraw", type=int, default=4)
    ap.add_argument("--r0", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    lams = [float(s) for s in args.lam.split(",")]
    Ts = [float(s) for s in args.T.split(",")]
    head(f"ПАРА ВКЛЮЧЕНИЙ: N={args.N}, затравки {seeds}, lam={lams}, T={Ts}")
    rows = run(args.N, seeds, args.dphi, lams, Ts, args.nd, args.ndraw, args.r0)
    out = args.out or os.path.join(HERE, f"entropy_pair_N{args.N}.npz")
    np.savez(out, rows=np.array(rows, dtype=object))
    summarise(rows, Ts)


if __name__ == "__main__":
    main()
