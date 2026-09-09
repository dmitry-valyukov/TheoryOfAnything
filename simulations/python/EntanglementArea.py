"""Энтропия мод дрожания через разрез спаянной сети: площадной и объёмный законы (T-4 (а′), 11.3/10.5).

Маршрут Джейкобсона требует от среды двух входов; здесь считается первый — закон энтропии.
Гармоническая энтропия запутанности гауссова состояния мод дрожания для подсистемы A:
  X = <u u>|_A = [ (1/2) H^{-1/2} coth(sqrt(H)/(2T)) ]_AA,   P = <p p>|_A = [ (1/2) H^{1/2} coth(sqrt(H)/(2T)) ]_AA,
  симплектические числа nu_k = sqrt(eig(X_A P_A)) >= 1/2,
  S(A) = sum_k [(nu+1/2) ln(nu+1/2) - (nu-1/2) ln(nu-1/2)].
T = 0 — чистое основное состояние (нулевые колебания): ожидание — площадной закон S ∝ Area
(постулат I Верлинде); T > 0 — подщелевой тепловой остаток: ожидание — кроссовер к объёмному
закону S ∝ Volume (постулат II). Подсистемы — шары радиуса R вокруг случайных центров.
Дополнительно: локальное «включение» (контакты в шаре r0 вокруг центра жёстче в lam раз — модель
узла, занявшего окно частот) и профиль изменения энтропии маленьких шаров delta S(r) с расстоянием
от включения при T = 0 и T > 0.

Единицы: hbar = 1, массы 1, k_контакта = 1/sigma^2; температура в долях w_D.

Запуск: python EntanglementArea.py [--N 1024] [--seeds 1,2] [--dphi 3e-2] [--T 0,0.02,0.1]
        python EntanglementArea.py --analyse entanglement_N1024.npz
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import find_phi_j, head, Z_C  # noqa: E402
from GravitySoftening import PinnedSpheres  # noqa: E402
from StateDrift import BondedNet, assemble_full  # noqa: E402


def build_XP(H, T):
    """Полные корреляционные матрицы X = <uu>, P = <pp> (3N x 3N) при температуре T (hbar=1).
    Нулевые моды (трансляции) отбрасываются (проектор)."""
    w2, E = np.linalg.eigh(H)
    keep = w2 > 1e-8 * w2.max()
    w = np.sqrt(w2[keep]); E = E[:, keep]
    if T > 0:
        cth = 1.0 / np.tanh(w / (2 * T))
    else:
        cth = np.ones_like(w)
    X = (E * (cth / (2 * w))[None, :]) @ E.T
    P = (E * (cth * w / 2)[None, :]) @ E.T
    return X, P, w


def ent_entropy(X, P, idx):
    """S(A) по симплектическим числам подматриц (idx — индексы частиц)."""
    sel = np.concatenate([3 * idx + c for c in range(3)]); sel.sort()
    XA = X[np.ix_(sel, sel)]; PA = P[np.ix_(sel, sel)]
    ev = np.linalg.eigvals(XA @ PA).real
    nu = np.sqrt(np.maximum(ev, 0.25000001))
    return float(((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)).sum())


def ball_indices(p, center, R):
    x = np.mod(p.s, p.L) @ p.F().T
    dx = x - center
    dx -= p.L * np.round(dx / p.L)
    return np.where((dx ** 2).sum(axis=1) < R * R)[0]


def stiffen(b, center, r0, lam):
    """BondedNet с контактами, у которых середина в шаре r0 от center, жёстче в lam раз.
    Реализация: масштабируем sig эффективно через вес контакта — проще пересобрать гессиан вручную."""
    i, j, dx, r, ov = b.contacts()
    x = np.mod(b.s, b.L) @ b.F().T
    mid = x[i] + 0.5 * dx
    dm = mid - center; dm -= b.L * np.round(dm / b.L)
    hot = (dm ** 2).sum(axis=1) < r0 * r0
    sig = b.rad[i] + b.rad[j]
    dVdr = -ov / sig
    d2V = np.ones_like(r) / sig ** 2
    d2V[hot] *= lam
    dVdr[hot] *= lam
    n = dx / r[:, None]
    K = (d2V - dVdr / r)[:, None, None] * n[:, :, None] * n[:, None, :] + (dVdr / r)[:, None, None] * np.eye(3)[None]
    N = b.N
    H = np.zeros((3 * N, 3 * N))
    for a_ in range(3):
        for b_ in range(3):
            np.add.at(H, (3 * i + a_, 3 * i + b_), K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * j + b_), K[:, a_, b_])
            np.add.at(H, (3 * i + a_, 3 * j + b_), -K[:, a_, b_])
            np.add.at(H, (3 * j + a_, 3 * i + b_), -K[:, a_, b_])
    return H, int(hot.sum())


def run_sample(N, seed, dphi, Ts, verbose=True):
    t0 = time.time()
    print(f"\n  --- образец N={N}, seed={seed}, dphi={dphi:g} ---", flush=True)
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
    H, Xis, aff, V, pr = assemble_full(b)
    d = 2 * b.rad.mean(); L = b.L
    rng = np.random.default_rng(4000 + seed)
    centers = rng.uniform(0, L, (6, 3))
    Rs = np.array([1.0, 1.3, 1.7, 2.2, 2.8, 3.5]) * d
    Rs = Rs[Rs < 0.45 * L]
    rec = dict(N=N, seed=seed, dphi=dphi, dz0=z0 - Z_C, d=d, L=L, Rs=Rs, Ts=np.array(Ts), S={}, nA={})
    for T in Ts:
        X, P, w = build_XP(H, T * math.sqrt((H.diagonal()).mean()))  # T в долях sqrt(среднего диагонального) ~ w_D/2
        wD = w.max()
        Svals = np.zeros((len(Rs), len(centers))); nAvals = np.zeros((len(Rs), len(centers)), int)
        for ir, R in enumerate(Rs):
            for ic, cen in enumerate(centers):
                idx = ball_indices(b, cen, R)
                if len(idx) < 4 or len(idx) > N - 4:
                    Svals[ir, ic] = np.nan; continue
                Svals[ir, ic] = ent_entropy(X, P, idx)
                nAvals[ir, ic] = len(idx)
        rec["S"][T] = Svals; rec["nA"][T] = nAvals; rec["wD"] = wD
        Sm = np.nanmean(Svals, axis=1); nm = nAvals.mean(axis=1)
        # подгонка S = c_A (R/d)^2 + c_V (R/d)^3
        A = np.vstack([(Rs / d) ** 2, (Rs / d) ** 3]).T
        m = ~np.isnan(Sm)
        cA, cV = np.linalg.lstsq(A[m], Sm[m], rcond=None)[0]
        al = np.polyfit(np.log(Rs[m] / d), np.log(Sm[m]), 1)[0]
        print(f"    T={T:g} w_ref: S(R): " + " ".join(f"{x:.1f}" for x in Sm) +
              f" | n_A: " + " ".join(f"{x:.0f}" for x in nm) +
              f" | S ∝ R^{al:.2f}; S = {cA:.2f}(R/d)^2 + {cV:.2f}(R/d)^3  [{time.time()-t0:.0f} с]", flush=True)
    # включение: жёсткие контакты в шаре r0 вокруг центра ячейки, профиль delta S малых шаров
    cen0 = np.array([L / 2, L / 2, L / 2])
    H1, nhot = stiffen(b, cen0, 1.5 * d, 4.0)
    print(f"    включение: {nhot} контактов x4 жёстче в шаре 1.5d вокруг центра", flush=True)
    probe_R = 1.3 * d
    dists = np.array([2.0, 3.0, 4.0, 5.0, 6.5, 8.0]) * d
    dists = dists[dists < 0.45 * L]
    rec["dists"] = dists; rec["dS"] = {}
    for T in Ts:
        Tabs = T * math.sqrt((H.diagonal()).mean())
        X0, P0, _ = build_XP(H, Tabs)
        X1, P1, _ = build_XP(H1, Tabs)
        prof = np.zeros(len(dists)); cnt = np.zeros(len(dists))
        for idist, dist in enumerate(dists):
            for m in range(8):
                th = rng.uniform(0, math.pi); ph = rng.uniform(0, 2 * math.pi)
                cen = cen0 + dist * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                cen = np.mod(cen, L)
                idx = ball_indices(b, cen, probe_R)
                if len(idx) < 4: continue
                s0 = ent_entropy(X0, P0, idx); s1 = ent_entropy(X1, P1, idx)
                prof[idist] += s1 - s0; cnt[idist] += 1
        prof /= np.maximum(cnt, 1)
        rec["dS"][T] = prof
        print(f"    T={T:g}: dS(r/d=" + ",".join(f"{x/d:.1f}" for x in dists) + ") = " +
              " ".join(f"{x:+.4f}" for x in prof) + f"  [{time.time()-t0:.0f} с]", flush=True)
    return [rec]


def analyse(recs):
    head("СВОДКА: энтропия мод дрожания через разрез (T-4 (а′), маршрут Джейкобсона)")
    for r in recs:
        print(f"\n  N={r['N']} seed={r['seed']} dphi={r['dphi']:g} dz={r['dz0']:.2f}")
        d = r["d"]; Rs = r["Rs"]
        for T in r["Ts"]:
            S = np.nanmean(r["S"][float(T)], axis=1)
            m = ~np.isnan(S)
            al = np.polyfit(np.log(Rs[m] / d), np.log(S[m]), 1)[0]
            A = np.vstack([(Rs / d) ** 2, (Rs / d) ** 3]).T
            cA, cV = np.linalg.lstsq(A[m], S[m], rcond=None)[0]
            print(f"    T={T:g}: S ∝ R^{al:.2f};  c_area={cA:.2f}, c_vol={cV:.2f}  (S(R)= " + " ".join(f"{x:.1f}" for x in S) + ")")
        if "dS" in r:
            for T in r["Ts"]:
                prof = r["dS"][float(T)]
                print(f"    dS от включения, T={T:g}: " + " ".join(f"{x:+.4f}" for x in prof))
    print("""
  Чтение. T=0: показатель ~2 (площадной закон) — вход (i) Джейкобсона; T>0: показатель ->3 (объёмный
  закон) — «тёмная энергия» Верлинде как тепловой остаток. dS(r) — знак и закон спада возмущения
  энтропии вокруг включения; для энтропийной силы нужен медленный спад (~1/r), быстрый (r^-3) её закрывает.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seeds", type=str, default="1,2")
    ap.add_argument("--dphi", type=float, default=3e-2)
    ap.add_argument("--T", type=str, default="0,0.02,0.1")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--analyse", type=str, default="")
    args = ap.parse_args()
    if args.analyse:
        recs = []
        for f in args.analyse.split(","):
            recs += list(np.load(f, allow_pickle=True)["recs"])
        analyse(recs); return
    seeds = [int(s) for s in args.seeds.split(",")]
    Ts = [float(s) for s in args.T.split(",")]
    head(f"ЭНТРОПИЯ ЧЕРЕЗ РАЗРЕЗ: N={args.N}, затравки {seeds}, dphi={args.dphi:g}, T={Ts} (в долях опорной частоты)")
    out = args.out or os.path.join(HERE, f"entanglement_N{args.N}.npz")
    recs = []
    for sd in seeds:
        recs += run_sample(args.N, sd, args.dphi, Ts)
        np.savez(out, recs=np.array(recs, dtype=object))
    analyse(recs)


if __name__ == "__main__":
    main()
