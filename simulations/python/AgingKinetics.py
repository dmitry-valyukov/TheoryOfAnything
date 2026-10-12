"""Кинетика старения спаянной сети под дрожанием: темп дрейфа состояния Gamma(t) = d ln mu / dt
как кандидат на константу среды в линии Lambda (100.15.7 п. 5, 11.2.3).

Механизм 11.2.3: у маргинальной упаковки нет щели у нуля в распределении зазоров; дрожание
замыкает почти-контакты, замкнувшийся контакт остаётся связью (постулат 1.11). Здесь считается
не закалённый снимок при фиксированной амплитуде, а КИНЕТИКА: за каждый период дрожания
(независимая гауссова выборка по модам гессиана) пара (i,j) с зазором h замыкается с
вероятностью p = 1/2 erfc(h / (s_ij sqrt2)), где s_ij — СКО относительного смещения пары вдоль
её оси при заданном спектре дрожания (zp: <q_m^2> ∝ 1/omega_m — нулевое дрожание; th: ∝ 1/omega_m^2 —
классическое тепловое), нормированном на СКО смещения частицы u (в диаметрах). За n периодов
P_закр = 1 - (1-p)^n. Сеть «в момент n» = опорная + все пары с P_закр > 1/2 (связи с длиной покоя
sigma_ij), минимизация, каскад замыканий (bond_all_active), модули по гессиану.

Выход: z(n), mu(n)/mu0, C(n) = d ln mu / d ln n = Gamma * t, и показатель d ln C / d ln n.
Для Lambda: Gamma = C / t0; Omega_Lambda = rho_P (Gamma t_P)^2 / rho_c = 8 pi C^2 / (3 (H0 t0)^2) ≈ 9.3 C^2.
Нужно C ≈ 0.27 при log10(omega_P t0) ≈ 61.

Результат (N=512, 2 затравки, dz0=0.4/0.75/1.4, u=1e-3..3e-2, оба спектра): C(10^61) = (2.4±0.7)e-3,
C по окнам log10 n = 4-20 / 20-40 / 40-62: 0.012 / 0.0044 / 0.0026; Gamma ∝ t^(-1.011±0.006).
Кинетика логарифмическая: Gamma ∝ 1/t даёт rho_Lambda ∝ H^2 (перенормировка G, w ≈ 0, ускорения нет),
величина — 1e-4 нужного. См. 100.17.6.

Запуск: python AgingKinetics.py [--N 512] [--seeds 1,2] [--dphi 3e-3,1e-2,3e-2] [--u 1e-3,3e-3,1e-2,3e-2,1e-1]
"""
import argparse, math, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy.special import erfc
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from StateDrift import BondedNet, moduli_full, PinnedSpheres, find_phi_j, Z_C  # noqa

LOGN = np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 58, 60, 61, 62], float)


def build(N, seed, dphi, p0, phi_j):
    p = PinnedSpheres(N, 0.65, seed)
    p.rad = p0.rad.copy(); p.vol_particles = p0.vol_particles; p.rc = p0.rc
    p.s, p.L = p0.s.copy(), p0.L; p.G3[:] = 0.0
    p.set_phi(phi_j + dphi); p.stabilize()
    z0, _, _ = p.coordination()
    i0, j0, *_ = p.contacts()
    b = BondedNet(N, 0.65, seed); b.rad = p.rad.copy(); b.vol_particles = p.vol_particles
    b.rc = p.rc; b.L = p.L; b.s = p.s.copy(); b.G3 = p.G3.copy(); b.bond(i0, j0)
    b.minimize(rel=1e-10)
    return b, z0 - Z_C


def near_pairs(b, hmax):
    b._pairs = None; b.build_pairs()
    ds, dx, r = b._geom(b.s)
    gap = r - b._sig
    sel = (~b._bonded) & (gap > 0) & (gap < hmax)
    return b._i[sel], b._j[sel], gap[sel], dx[sel] / r[sel, None]


def pair_sigma(H, i, j, n, u, N, weighting):
    w, v = np.linalg.eigh(H)
    keep = w > 1e-8 * w.max()
    w, v = w[keep], v[:, keep]
    var = 1.0 / np.sqrt(w) if weighting == "zp" else 1.0 / w      # <q_m^2> до нормировки
    scale = u ** 2 / (var.sum() / N)                                 # mean_i <|xi_i|^2> = u^2
    var = var * scale
    V = v.reshape(N, 3, -1)
    d = np.einsum('kc,kcm->km', n, V[j] - V[i])                      # (P, M)
    s2 = (d ** 2 * var[None, :]).sum(axis=1)
    return np.sqrt(s2), w


def kinetics(b, dz0, u, weighting, d, log):
    s_ref, keys_ref = b.s.copy(), b.bond_keys.copy()
    mus0, B0, pr0, H = moduli_full(b)
    mu0 = mus0.mean()
    i, j, gap, n = near_pairs(b, 0.6 * d)
    s, w = pair_sigma(H, i, j, n, u * d, b.N, weighting)
    p = 0.5 * erfc(gap / (s * math.sqrt(2)))
    p = np.clip(p, 1e-300, 1.0)
    lognstar = np.log10(math.log(2) / p)                              # n p = ln 2  <=> P_закр = 1/2
    out = []
    for ln in LOGN:
        closed = lognstar <= ln
        b.s = s_ref.copy(); b.bond_keys = keys_ref.copy(); b._pairs = None
        if closed.any():
            b.bond(i[closed], j[closed])
        b.build_pairs(); b.minimize(rel=1e-10)
        ncas = b.bond_all_active()
        if ncas:
            b.minimize(rel=1e-10)
        mus, B, pr, _ = moduli_full(b)
        z, _, _ = b.coordination()
        out.append((ln, int(closed.sum()), ncas, z, mus.mean() / mu0, B / B0))
        log(f"      u={u:g} {weighting} log10 n={ln:4.0f}: замкнуто {closed.sum():4d} (+каскад {ncas:3d})  z={z:.3f}  mu/mu0={mus.mean()/mu0:.4f}  B/B0={B/B0:.4f}")
    b.s = s_ref.copy(); b.bond_keys = keys_ref.copy(); b._pairs = None; b.build_pairs()
    out = np.array(out)
    return dict(u=u, weighting=weighting, dz0=dz0, mu0=mu0, npairs=len(gap), gap=gap, s=s, lognstar=lognstar,
                table=out, wD=math.sqrt(w.max()))


def report(res):
    t = res["table"]; ln = t[:, 0]; lmu = np.log(t[:, 4])
    C = np.gradient(lmu, ln * math.log(10))          # d ln mu / d ln n
    k = np.argmin(np.abs(ln - 61))
    sel = ln >= 48
    sl = np.polyfit(ln[sel] * math.log(10), np.log(np.maximum(C[sel], 1e-300)), 1)[0] if C[sel].min() > 0 else float('nan')
    return C, k, sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512); ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--dphi", default="3e-3,1e-2,3e-2"); ap.add_argument("--u", default="1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--weighting", default="zp,th"); ap.add_argument("--out", default="aging_N512.npz")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]; dphis = [float(x) for x in a.dphi.split(",")]
    us = [float(x) for x in a.u.split(",")]; ws = a.weighting.split(",")
    t0 = time.time(); log = lambda m: print(m + f"  [{time.time()-t0:.0f} с]", flush=True)
    recs = []
    for seed in seeds:
        p0, phi_j = find_phi_j(a.N, seed, False, verbose=False)
        for dphi in dphis:
            b, dz0 = build(a.N, seed, dphi, p0, phi_j)
            d = 2 * b.rad.mean()
            log(f"\n--- seed={seed} dphi={dphi:g} dz0={dz0:+.3f} ---")
            for wgt in ws:
                for u in us:
                    r = kinetics(b, dz0, u, wgt, d, log)
                    r.update(seed=seed, dphi=dphi)
                    C, k, sl = report(r)
                    log(f"    => u={u:g} {wgt}: C(10^61)={C[k]:.4f}  z(61)={r['table'][k,3]:.3f}  mu/mu0(61)={r['table'][k,4]:.3f}  dlnC/dlnt(48..62)={sl:+.3f}  пар<0.6d: {r['npairs']}")
                    recs.append(r)
            np.savez(a.out, recs=np.array(recs, dtype=object), allow_pickle=True)
    log("готово")


if __name__ == "__main__":
    main()
