"""Взаимодействие ДВУХ настоящих Q-шаров через нулевые колебания их собственного поля
(узкая часть долга 12, 100.17.2).

Отличие от EntropyPairSign.py / LatticePairForce.py: там включение было искусственным
возмущением жёсткости, здесь — настоящий узел главы [2.5]: профиль f(r) кубично-квинтического
Q-шара (CubicQuintic.find_ground_state + _shoot), а возмущение для флуктуаций — его собственный
потенциал второй вариации

    H = -nabla^2 + lam + V(r),   V(r) = U''(f) - U''(0) = -3 g f(r)^2 + 5 h f(r)^4,
    lam = m^2 - omega^2 (гэп флуктуаций),  U(f) = m^2 f^2/2 - g f^4/4 + h f^6/6.

Важное структурное замечание, которое и есть ответ по дальнодействию: у модели 2.5 поле
флуктуаций ЩЕЛЕВОЕ (lam > 0 — условие существования Q-шара), поэтому взаимодействие пары
обязано спадать экспоненциально с показателем 2 sqrt(lam), а не по степенному закону.
Безщелевой канал — акустический сектор среды, и он считался отдельно (100.17.2).

Величина — та же вторая разность:
    U_int(D) = Q(пара) - Q(узел 1) - Q(узел 2) + Q(без узлов),  Q = E_zp, F(T), S(T).
Знак силы: f = -dU_int/dD; притяжение, если U_int растёт с D.

Запуск: python QballPairForce.py [--n 20] [--dx 1.0] [--lam 0.4] [--T 0,0.3] [--dim 3|1]
"""
import argparse, math, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import CubicQuintic as cq


def qball_well(lam, g=2.0, h=1.0):
    """Профиль узла и его потенциал второй вариации V(r) (интерполятор)."""
    f0 = cq.find_ground_state(lam, g, h)
    if f0 is None:
        raise SystemExit(f"при lam={lam} ветви Q-шара нет (окно: lam < {0.75*1.0:.2f} при g=2,h=1)")
    _, sol = cq._shoot(f0, lam, g, h, fast=False)
    r, f = sol.t, sol.y[0]
    cut = int(np.argmin(np.abs(f)))
    r, f = r[:cut + 1], np.abs(f[:cut + 1])
    V = -3.0 * g * f ** 2 + 5.0 * h * f ** 4
    rhalf = float(r[np.argmax(f < f0 / 2)])
    return f0, rhalf, (lambda rr: np.interp(rr, r, V, right=0.0))


def lap_potential(n, dx, dim, centers, Vf, lam):
    """H = -nabla^2 + lam + sum_c V(|x - c|) на периодической решётке n^dim."""
    N = n ** dim
    H = np.zeros((N, N))
    if dim == 3:
        idx = lambda x, y, z: ((x % n) * n + (y % n)) * n + (z % n)
        coord = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3) * dx
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    a = idx(x, y, z)
                    H[a, a] += 6.0 / dx ** 2
                    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        b = idx(x + d[0], y + d[1], z + d[2])
                        H[a, b] -= 1.0 / dx ** 2
                        H[b, a] -= 1.0 / dx ** 2
    else:
        coord = (np.arange(n) * dx).reshape(-1, 1)
        for x in range(n):
            H[x, x] += 2.0 / dx ** 2
            b = (x + 1) % n
            H[x, b] -= 1.0 / dx ** 2
            H[b, x] -= 1.0 / dx ** 2
    V = np.zeros(N)
    L = n * dx
    for c in centers:
        d = coord - np.asarray(c, float)
        d -= L * np.round(d / L)
        V += Vf(np.sqrt((d ** 2).sum(axis=1)))
    H[np.diag_indices_from(H)] += lam + V
    return H


def potentials(H, Ts, nneg_expected=None):
    """Потенциалы по положительному сектору. У Q-шара в НЕограниченной задаче есть ровно одна
    отрицательная (дилатационная) мода на узел — она снимается сохранением заряда, поэтому
    отбрасывается. В разности счёт сходится: пара даёт 2, каждый одиночный 1, фон 0."""
    w2 = np.linalg.eigvalsh(H)
    nneg = int((w2 <= 1e-12).sum())
    if nneg_expected is not None and nneg != nneg_expected:
        print(f"      [внимание: отрицательных/нулевых мод {nneg}, ожидалось {nneg_expected}]", flush=True)
    w2 = w2[nneg:]
    w = np.sqrt(w2)
    out = {"E": float(0.5 * w.sum()), "nneg": nneg}
    for T in Ts:
        if T <= 0:
            out[("F", T)] = out["E"]; out[("S", T)] = 0.0
            continue
        x = w / T
        l = np.log1p(-np.exp(-x))
        out[("F", T)] = float(0.5 * w.sum() + T * l.sum())
        out[("S", T)] = float((x / np.expm1(x) - l).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--T", type=str, default="0,0.3")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--Ds", type=str, default="")
    args = ap.parse_args()
    Ts = [float(s) for s in args.T.split(",")]
    n, dx, lam, dim = args.n, args.dx, args.lam, args.dim
    t0 = time.time()
    f0, rhalf, Vf = qball_well(lam)
    mu = math.sqrt(lam)
    print(f"Q-шар: lam={lam}, f0={f0:.4f}, R_1/2={rhalf:.2f}; гэп флуктуаций mu=sqrt(lam)={mu:.3f}, "
          f"ожидаемый закон спада e^(-2 mu D), 2mu={2*mu:.3f}", flush=True)
    print(f"решётка {n}^{dim}, шаг dx={dx} (R_1/2 = {rhalf/dx:.1f} узлов, 1/mu = {1/mu/dx:.1f} узлов), "
          f"размер {n*dx:.1f}", flush=True)
    base = potentials(lap_potential(n, dx, dim, [], Vf, lam), Ts)
    print(f"  фон: E_zp={base['E']:.9f}  [{time.time()-t0:.0f} с]", flush=True)
    L = n * dx
    if args.Ds:
        Ds = [float(x) for x in args.Ds.split(",")]
    else:
        Ds = [round(x, 3) for x in np.arange(2 * rhalf, 0.45 * L + 1e-9, dx)]
    cen = np.array([L / 2] * dim)
    for D in Ds:
        off = np.zeros(dim); off[0] = D / 2
        c1, c2 = np.mod(cen - off, L), np.mod(cen + off, L)
        q1 = potentials(lap_potential(n, dx, dim, [c1], Vf, lam), Ts, 1)
        q2 = potentials(lap_potential(n, dx, dim, [c2], Vf, lam), Ts, 1)
        qp = potentials(lap_potential(n, dx, dim, [c1, c2], Vf, lam), Ts, 2)
        uE = qp["E"] - q1["E"] - q2["E"] + base["E"]
        msg = (f"    D={D:5.2f} (D/R={D/rhalf:4.2f}): U_int(E_zp)={uE:+.4e}"
               f" [отр.мод: {qp['nneg']}/{q1['nneg']}/{q2['nneg']}]")
        for T in Ts:
            if T <= 0: continue
            uF = qp[("F", T)] - q1[("F", T)] - q2[("F", T)] + base[("F", T)]
            uS = qp[("S", T)] - q1[("S", T)] - q2[("S", T)] + base[("S", T)]
            msg += f"  F(T={T:g})={uF:+.4e}  dS(T={T:g})={uS:+.4e}"
        print(msg + f"   [{time.time()-t0:.0f} с]", flush=True)
    print("""
  Чтение: U_int растёт с D — притяжение. Ожидание по структуре модели: экспоненциальный спад
  с показателем 2 sqrt(lam) (щелевое поле), а не степенной. Степенного 1/D здесь быть не может
  по построению, и это ответ по дальнодействию для настоящего узла 2.5.""")


if __name__ == "__main__":
    main()
