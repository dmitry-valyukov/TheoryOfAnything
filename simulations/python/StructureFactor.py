"""Статический структурный фактор упаковки: числовой S_N(k) и объёмно-взвешенный S_V(k) (T-14).

Для бидисперсной упаковки гипероднородность проявляется в ОБЪЁМНОМ структурном факторе (флуктуации доли
объёма), а не в числовом: S_N(k -> 0) у смеси стремится к константе (композиционные флуктуации), тогда как
chi_V(k) -> 0 (Berthier et al. 2011; Zachary, Jiao, Torquato 2011). Считаем оба по кэшированным упаковкам
packing_N*_s*_dphi*.npz (DispersionLarge.py): S_N = |sum_j e^{ikr_j}|^2 / N,
S_V = |sum_j v_j e^{ikr_j}|^2 / (N <v^2>), где v_j = 4/3 pi R_j^3; усреднение по оболочкам |k| по всем
целочисленным n с |n| <= nmax; подгонка степени alpha в S ∝ k^alpha при k d < k_fit d.

Запуск: python StructureFactor.py packing_N8192_s1_dphi0.01.npz [...] [--nmax 14]
"""

import argparse
import math
import os
import sys

import numpy as np


def sk_shells(x, L, w, nmax):
    rng = range(-nmax, nmax + 1)
    ks, Sv = [], []
    w2 = (w ** 2).mean()
    for nx in rng:
        for ny in rng:
            for nz in rng:
                if nx == ny == nz == 0: continue
                if nx < 0 or (nx == 0 and ny < 0) or (nx == 0 and ny == 0 and nz < 0): continue
                if nx * nx + ny * ny + nz * nz > nmax * nmax: continue
                kv = 2 * math.pi / L * np.array([nx, ny, nz])
                rk = (w * np.exp(1j * (x @ kv))).sum()
                ks.append(np.linalg.norm(kv)); Sv.append(abs(rk) ** 2 / (len(x) * w2))
    ks, Sv = np.array(ks), np.array(Sv)
    uk, inv = np.unique(np.round(ks, 6), return_inverse=True)
    cnt = np.bincount(inv)
    return uk, np.array([Sv[inv == q].mean() for q in range(len(uk))]), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--nmax", type=int, default=14)
    ap.add_argument("--kfit", type=float, default=1.0)
    args = ap.parse_args()
    for f in args.files:
        z = np.load(f); s, L, rad = z["s"], float(z["L"]), z["rad"]
        x = np.mod(s, L)
        d = 2 * rad.mean(); N = len(rad)
        v = 4 / 3 * math.pi * rad ** 3
        uk, SN, cnt = sk_shells(x, L, np.ones(N), args.nmax)
        _, SV, _ = sk_shells(x, L, v, args.nmax)
        print(f"\n{os.path.basename(f)}: N={N}, L={L:.2f}, d={d:.3f}, k_min d={uk[0]*d:.3f}")
        print(f"  {'k d':>6} {'n_vec':>5} {'S_N':>7} {'S_V':>7}")
        for u, sn, sv, c in zip(uk[:14], SN[:14], SV[:14], cnt[:14]):
            print(f"  {u*d:>6.3f} {c:>5d} {sn:>7.4f} {sv:>7.4f}")
        m = uk * d < args.kfit
        if m.sum() >= 4:
            aN = np.polyfit(np.log(uk[m] * d), np.log(SN[m]), 1)[0]
            aV = np.polyfit(np.log(uk[m] * d), np.log(SV[m]), 1)[0]
            print(f"  показатели при k d < {args.kfit}: S_N ∝ k^{aN:.2f}, S_V ∝ k^{aV:.2f}; S_V(k_min)/S_V(k d~1) = {SV[0]/np.interp(1.0/d, uk, SV):.3f}")


if __name__ == "__main__":
    main()
