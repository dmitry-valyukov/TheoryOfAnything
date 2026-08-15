"""Закон размягчения по амплитуде mu_s(gamma) на аморфной сети (T-44, T-34, T-4).

Программа T-44: если среда стоит в критической точке заклинивания, то
безразмерные константы модели могут быть универсальными амплитудными
отношениями этой точки. Первый вход — закон размягчения модуля сдвига по
амплитуде деформации: в 2.9.6 показано, что узел требует квартичного
РАЗМЯГЧЕНИЯ (g' > 0) и секстического упрочнения (h' > 0), а путь 10 к alpha
(4.14) требует глубины размягчения delta_max ~ 0.06–0.08 у заряда.

Что считается. Аморфная сеть центральных гармонических пружин (та же, что
в LorentzProtection.py), периодическая, с координационным числом z,
которое прореживанием связей доводится от ~15 до окрестности изостатики
z_c = 6. Ячейка подвергается конечному простому сдвигу gamma (градиент
деформации F применяется ко всей ячейке, узлы релаксируют НЕАФФИННО —
минимизация энергии по внутренним смещениям при каждом gamma).
Напряжение sigma(gamma) = (1/V) dE/dgamma; касательный модуль
mu(gamma) = d sigma/d gamma. Подгонка

    mu(gamma)/mu(0) = 1 - a gamma² + b gamma⁴

даёт знак и величину квартичного члена (a > 0 — размягчение) и
секстического (b). Проверка универсальности: a при разных z, N, затравках.
Дополнительно — сеть с предварительным сжатием связей (пружины короче
равновесной длины на delta0), как в заклиненных упаковках.

Запуск:  python SofteningLaw.py [N]
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78)


def build_network(N=500, L=8.0, r_cut=1.6, seed=1):
    r = np.random.default_rng(seed)
    pts = r.uniform(0, L, (N, 3))
    d0 = 1.15 * L / N ** (1 / 3)
    for _ in range(80):
        diff = pts[:, None, :] - pts[None, :, :]
        diff -= L * np.round(diff / L)
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e9)
        close = dist < d0
        unit = diff / (dist[:, :, None] + 1e-12)
        push = np.where(close[:, :, None], unit * (d0 - dist)[:, :, None], 0.0).sum(axis=1)
        pts = np.mod(pts + 0.3 * push, L)
    diff = pts[:, None, :] - pts[None, :, :]
    diff -= L * np.round(diff / L)
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1e9)
    iu = np.triu_indices(N, 1)
    mask = dist[iu] < r_cut
    ii, jj = iu[0][mask], iu[1][mask]
    edges = np.stack([ii, jj], axis=1)
    b0 = diff[jj, ii]                     # x_j - x_i (минимальный образ, опорная ячейка)
    return pts, edges, b0, L


def prune_to_z(edges, b0, N, z_target, seed=1):
    """Случайное прореживание до среднего координационного числа z_target,
    не оставляя узлов с < 4 связями (чтобы не плодить локальные флопы)."""
    r = np.random.default_rng(seed)
    m_target = int(round(z_target * N / 2))
    idx = np.arange(len(edges))
    r.shuffle(idx)
    keep = np.ones(len(edges), bool)
    deg = np.bincount(edges.ravel(), minlength=N)
    for e in idx:
        if keep.sum() <= m_target:
            break
        i, j = edges[e]
        if deg[i] > 4 and deg[j] > 4:
            keep[e] = False; deg[i] -= 1; deg[j] -= 1
    return edges[keep], b0[keep]


class ShearedNetwork:
    def __init__(self, N, edges, b0, L, k=1.0, prestrain=0.0):
        self.N, self.edges, self.b0, self.L, self.k = N, edges, b0, L, k
        self.l0 = np.linalg.norm(b0, axis=1) * (1.0 - prestrain)   # длина покоя
        self.V = L ** 3
        self.i, self.j = edges[:, 0], edges[:, 1]

    def F(self, g):
        return np.array([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def energy_grad(self, uflat, g):
        u = uflat.reshape(self.N, 3)
        b = self.b0 @ self.F(g).T + u[self.j] - u[self.i]
        r = np.linalg.norm(b, axis=1)
        dl = r - self.l0
        E = 0.5 * self.k * (dl ** 2).sum()
        f = (self.k * dl / r)[:, None] * b          # dE/db
        grad = np.zeros((self.N, 3))
        np.add.at(grad, self.j, f)
        np.add.at(grad, self.i, -f)
        # закрепление трансляции: штраф на среднее смещение
        um = u.mean(axis=0)
        E += 0.5 * 1e3 * (um @ um) * self.N
        grad += 1e3 * um[None, :]
        return E, grad.ravel()

    def relax(self, g, u0):
        res = minimize(lambda x: self.energy_grad(x, g), u0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=4000, gtol=1e-10, ftol=1e-15))
        return res.x, res.fun

    def stress(self, u, g):
        """sigma_xy = (1/V) dE/dgamma при фиксированных u (теорема о виртуальной работе:
        релаксированные u не дают вклада первого порядка)."""
        uu = u.reshape(self.N, 3)
        b = self.b0 @ self.F(g).T + uu[self.j] - uu[self.i]
        r = np.linalg.norm(b, axis=1)
        dl = r - self.l0
        dbdg = np.zeros_like(b); dbdg[:, 0] = self.b0[:, 1]     # d(F b0)/dgamma = (b0_y, 0, 0)
        return float((self.k * dl / r * (b * dbdg).sum(axis=1)).sum()) / self.V


def modulus_curve(net, gammas):
    u = np.zeros(3 * net.N)
    sig = []
    for g in gammas:
        u, _ = net.relax(g, u)
        sig.append(net.stress(u, g))
    sig = np.array(sig)
    return sig


def fit_softening(gammas, sig):
    """sigma(g) = mu0 (g - a g³/3 + b g⁵/5)  <=>  mu(g) = mu0 (1 - a g² + b g⁴)."""
    A = np.stack([gammas, -gammas ** 3 / 3, gammas ** 5 / 5], axis=1)
    coef, *_ = np.linalg.lstsq(A, sig, rcond=None)
    mu0 = coef[0]
    return mu0, coef[1] / mu0, coef[2] / mu0


def run_case(N, z_target, seed, prestrain=0.0, gmax=0.30, ng=13, label=""):
    pts, edges, b0, L = build_network(N=N, seed=seed)
    if z_target is not None:
        edges, b0 = prune_to_z(edges, b0, N, z_target, seed)
    z = 2 * len(edges) / N
    net = ShearedNetwork(N, edges, b0, L, prestrain=prestrain)
    gammas = np.linspace(0.0, gmax, ng)
    sig = modulus_curve(net, gammas)          # первая точка gamma = 0: релаксация предсжатия
    sig = sig[1:] - sig[0]; gammas = gammas[1:]
    mu0, a, b = fit_softening(gammas, sig)
    # касательный модуль на концах для наглядности
    mu_tan = np.gradient(sig, gammas)
    print(f"  {label:26s} z = {z:5.2f}  mu0 = {mu0:8.4f}   a = {a:+8.3f}   b = {b:+8.2f}   "
          f"mu(0.1)/mu0 = {mu_tan[np.argmin(abs(gammas-0.1))]/mu0:.4f}  mu(0.3)/mu0 = {mu_tan[-1]/mu0:.4f}")
    return z, mu0, a, b, gammas, sig


def main(N=500):
    head("ЗАКОН РАЗМЯГЧЕНИЯ ПО АМПЛИТУДЕ: mu(gamma)/mu0 = 1 - a gamma² + b gamma⁴")
    print(f"\n  N = {N}, гармонические центральные пружины, периодический сдвиг, неаффинная релаксация\n")
    print("  --- зависимость от координационного числа (без предсжатия) ---")
    rows = []
    for zt in (None, 10.0, 8.0, 7.0, 6.5):
        rows.append(run_case(N, zt, 1, label=f"z_target={zt}"))
    print("\n  --- универсальность: другие затравки и размер при z ~ 7 ---")
    for seed, n in ((2, N), (3, N), (1, int(N * 1.6))):
        run_case(n, 7.0, seed, label=f"seed={seed}, N={n}")
    print("\n  --- преднапряжение связей при z ~ 7: delta0 > 0 — растяжение, delta0 < 0 — сжатие (заклиненная упаковка) ---")
    for ps in (0.05, 0.02, -0.01, -0.02, -0.04, -0.06):
        run_case(N, 7.0, 1, prestrain=ps, label=f"delta0={ps:+}")
    print("""
  Чтение. a > 0 — квартичное размягчение (то, что нужно узлу и пути 10 к
  alpha), a < 0 — упрочнение. Универсальность: сравнивать a при одном z и
  разных затравках/N; критичность — тренд a при z -> 6. Глубина
  размягчения на плато оценивается как delta = a²/(4b) при b > 0
  (максимум 1 - a g² + b g⁴ по g² даёт delta_max = a²/4b); сравнить с
  требуемыми 0.06–0.08 (4.14, путь 10).
""")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
