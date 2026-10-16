"""Проверка схемы «фазового зацепления через мгновенное давление» (глава 100.6.4, п. 8).

Схема: фазы двух узлов связаны каналом с силой eps = gamma0/D,
    phi_A = phi0A + eps*cos 2(phi_B - b),   phi_B = phi0B + eps*cos 2(phi_A - a),
исходы A = sign cos 2(phi_A - a), B = sign cos 2(phi_B - b);
начальные фазы термостата phi0A, phi0B независимы и равномерны на [0, 2pi).
Система решается итерациями до неподвижной точки (сжатие при eps < 1/2).

Результат (N = 6e4): при eps = 0.01…0.45 корреляция E и CHSH S — на уровне шума выборки
(max|S| по сетке углов ~0.02). Причина: сдвиг равномерной фазы на величину, зависящую
от чужих переменных, оставляет её равномерной и независимой.

Запуск: python BellPhaseLock.py [--N 60000] [--step 11.25]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=60_000)
ap.add_argument("--step", type=float, default=11.25, help="шаг сетки углов, градусы")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
p0A = rng.uniform(0, 2*np.pi, args.N)
p0B = rng.uniform(0, 2*np.pi, args.N)

def corr(a, b, eps, it=60):
    pA, pB = p0A.copy(), p0B.copy()
    for _ in range(it):
        pA_new = p0A + eps*np.cos(2*(pB - b))
        pB_new = p0B + eps*np.cos(2*(pA - a))
        pA, pB = 0.5*(pA + pA_new), 0.5*(pB + pB_new)
    A = np.sign(np.cos(2*(pA - a)))
    B = np.sign(np.cos(2*(pB - b)))
    return (A*B).mean(), A.mean(), B.mean()

d = np.pi/180
for eps in [0.01, 0.1, 0.3, 0.45]:
    a, a2, b, b2 = 0, 45*d, 22.5*d, 67.5*d
    Es = [corr(x, y, eps) for x, y in [(a, b), (a, b2), (a2, b), (a2, b2)]]
    S = Es[0][0] - Es[1][0] + Es[2][0] + Es[3][0]
    grid = np.arange(0, 90, args.step)*d
    tab = {(x, y): corr(x, y, eps, it=30)[0] for x in grid for y in grid}
    best = max(abs(tab[(x, y)] - tab[(x, y2)] + tab[(x2, y)] + tab[(x2, y2)])
               for x in grid for x2 in grid for y in grid for y2 in grid)
    print(f"eps={eps:5.2f}  E(0,22.5)={Es[0][0]:+.4f}  S(std)={S:+.4f}  "
          f"max|S|(grid)={best:.3f}  <A>={Es[0][1]:+.3f} <B>={Es[0][2]:+.3f}  "
          f"(шум ~{1/np.sqrt(args.N):.3f} на E)")
