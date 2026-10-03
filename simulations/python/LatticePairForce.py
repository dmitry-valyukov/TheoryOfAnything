"""Взаимодействие двух включений через дрожание на УПОРЯДОЧЕННОЙ решётке (тест 2, 100.12.8).

Зачем отдельно от спаянной сети: в неупорядоченном образце второй разностью
U_int = Q(пара) - Q(c1) - Q(c2) + Q(нет) ловится сигнал меньше разброса по месту, и знак не
определяется. На кубической решётке неупорядоченности нет вовсе, поэтому вторая разность —
точная физика, ограниченная только машинной точностью.

Модель: простая кубическая решётка n^3 узлов, 3 компоненты смещения, пружины между соседями
(k=1); kp — слабое закрепление узлов. ВАЖНО: kp>0 даёт среде щель (omega_min = sqrt(kp)) и,
значит, конечную длину корреляции xi ~ sqrt(k/kp) — спад взаимодействия тогда экспоненциальный,
а не степенной. Реальная среда модели бесщелевая (акустическая), поэтому основной расчёт идёт
при kp = 0, а нулевые моды (трансляции) отбрасываются. Включение: пружины, чей центр
попал в шар r0 вокруг центра включения, умножены на lam (lam>1 — ужесточение, lam<1 —
размягчение; гравитации нужно размягчение, 7.4/7.7).

Считаются те же потенциалы, что в EntropyPairSign.py: E_zp = (1/2) sum w, F(T), S(T).
Знак силы: f = -dU_int/dD; притяжение, если U_int растёт с D.

Запуск: python LatticePairForce.py [--n 10] [--lam 4,0.25] [--T 0,0.1] [--r0 1.2] [--kp 0.05]
"""
import argparse, math, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np


def build_scalar(n, k, kp, incl=(), r0=1.2, lam=1.0, count=False):
    """Скалярное безщелевое поле на кубической решётке: 1 степень свободы на узел,
    H = -nabla^2 (взвешенный по связям) + kp. Контроль показателя: позволяет большие ящики
    (n^3 вместо 3 n^3 степеней свободы), чтобы отделить физику от конечного размера."""
    N = n ** 3
    nhot = 0
    idx = lambda x, y, z: ((x % n) * n + (y % n)) * n + (z % n)
    H = np.zeros((N, N))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                a = idx(x, y, z)
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    b = idx(x + d[0], y + d[1], z + d[2])
                    mid = np.array([x, y, z], float) + 0.5 * np.array(d, float)
                    kk = k
                    for ce in incl:
                        dm = mid - ce
                        dm -= n * np.round(dm / n)
                        if (dm ** 2).sum() < r0 * r0:
                            kk = k * lam
                            nhot += 1
                            break
                    H[a, a] += kk; H[b, b] += kk
                    H[a, b] -= kk; H[b, a] -= kk
    H[np.diag_indices_from(H)] += kp
    return (H, nhot) if count else H


def build(n, k, kp, incl=(), r0=1.2, lam=1.0, count=False):
    """Гессиан кубической решётки; incl — список центров включений (в единицах шага).
    count=True — вернуть (H, число задетых связей) для проверки, что включения не пересеклись."""
    nhot = 0
    N = n ** 3
    idx = lambda x, y, z: ((x % n) * n + (y % n)) * n + (z % n)
    H = np.zeros((3 * N, 3 * N))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                a = idx(x, y, z)
                for dvec, c in (((1, 0, 0), 0), ((0, 1, 0), 1), ((0, 0, 1), 2)):
                    b = idx(x + dvec[0], y + dvec[1], z + dvec[2])
                    mid = np.array([x, y, z], float) + 0.5 * np.array(dvec, float)
                    kk = k
                    for ce in incl:
                        dm = mid - ce
                        dm -= n * np.round(dm / n)
                        if (dm ** 2).sum() < r0 * r0:
                            kk = k * lam
                            nhot += 1
                            break
                    # продольная пружина вдоль оси связи: только компонента c
                    H[3 * a + c, 3 * a + c] += kk
                    H[3 * b + c, 3 * b + c] += kk
                    H[3 * a + c, 3 * b + c] -= kk
                    H[3 * b + c, 3 * a + c] -= kk
                    # слабая поперечная жёсткость связи (чтобы решётка не была вырожденной)
                    for c2 in range(3):
                        if c2 == c:
                            continue
                        kt = 0.3 * kk
                        H[3 * a + c2, 3 * a + c2] += kt
                        H[3 * b + c2, 3 * b + c2] += kt
                        H[3 * a + c2, 3 * b + c2] -= kt
                        H[3 * b + c2, 3 * a + c2] -= kt
    H[np.diag_indices_from(H)] += kp
    return (H, nhot) if count else H


def potentials(w, Ts):
    out = {"E": 0.5 * w.sum()}
    for T in Ts:
        if T <= 0:
            out[("F", T)] = out["E"]; out[("S", T)] = 0.0
            continue
        x = w / T
        l = np.log1p(-np.exp(-x))
        out[("F", T)] = float(0.5 * w.sum() + T * l.sum())
        out[("S", T)] = float((x / np.expm1(x) - l).sum())
    return out


def freqs(H):
    """Частоты; нулевые моды (трансляции при kp=0) отбрасываются, как в EntanglementArea.build_XP."""
    w2 = np.linalg.eigvalsh(H)
    w2 = w2[w2 > 1e-8 * w2.max()]
    return np.sqrt(w2)


def jitter_profile(n, kp, r0, lam, Ts, cen=None):
    """Профиль модуляции <u^2>(r) вокруг ОДНОГО включения на упорядоченной решётке.
    Чистая проверка показателя 7.9.6 (r^-3 как подпись бессилового источника) без неупорядоченности."""
    H0 = build(n, 1.0, kp)
    H1 = build(n, 1.0, kp, [cen if cen is not None else np.array([n/2.]*3)], r0, lam)
    out = {}
    for H, tag in ((H0, "fon"), (H1, "incl")):
        w2, E = np.linalg.eigh(H)
        keep = w2 > 1e-8 * w2.max()
        w = np.sqrt(w2[keep]); E = E[:, keep]
        for T in Ts:
            cth = np.ones_like(w) if T <= 0 else 1.0 / np.tanh(w / (2 * T))
            # <u^2> на узел: диагональ X, суммированная по 3 компонентам
            diag = ((E ** 2) * (cth / (2 * w))[None, :]).sum(axis=1)
            ncomp = diag.size // n ** 3
            out[(tag, T)] = diag.reshape(n ** 3, ncomp).sum(axis=1)
    # расстояния узлов от центра
    idx = np.arange(n ** 3)
    x = np.stack([(idx // (n * n)) % n, (idx // n) % n, idx % n], axis=1).astype(float)
    c = cen if cen is not None else np.array([n / 2.] * 3)
    dx = x - c; dx -= n * np.round(dx / n)
    r = np.sqrt((dx ** 2).sum(axis=1))
    res = {}
    for T in Ts:
        d = out[("incl", T)] - out[("fon", T)]
        bins = np.arange(1.0, 0.45 * n, 1.0)
        prof = []
        for b in bins:
            m = (r >= b - 0.5) & (r < b + 0.5)
            prof.append(np.mean(d[m]) if m.sum() else np.nan)
        res[T] = (bins, np.array(prof))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--lam", type=str, default="4,0.25")
    ap.add_argument("--T", type=str, default="0,0.1,0.3")
    ap.add_argument("--r0", type=float, default=1.2)
    ap.add_argument("--kp", type=float, default=0.0)
    ap.add_argument("--Dmax", type=float, default=0.0)
    ap.add_argument("--profile", action="store_true", help="профиль <u^2>(r) вокруг одного включения")
    ap.add_argument("--scalar", action="store_true", help="скалярное поле (1 ст.св. на узел): большие ящики")
    args = ap.parse_args()
    lams = [float(s) for s in args.lam.split(",")]
    if args.scalar:
        global build
        build = build_scalar
    Ts = [float(s) for s in args.T.split(",")]
    n = args.n
    t0 = time.time()
    print(f"решётка {n}^3 ({(n**3 if args.scalar else 3*n**3)} степеней свободы), r0={args.r0}, kp={args.kp}, T={Ts}", flush=True)
    H0 = build(n, 1.0, args.kp)
    w0 = freqs(H0)
    base = potentials(w0, Ts)
    print(f"  фон: E_zp={base['E']:.6f}, w в [{w0.min():.4f}, {w0.max():.4f}]  [{time.time()-t0:.0f} с]", flush=True)
    cen = np.array([n / 2.0, n / 2.0, n / 2.0])
    if args.profile:
        for lam in lams:
            res = jitter_profile(n, args.kp, args.r0, lam, Ts)
            for T in Ts:
                b, prof = res[T]
                m = np.isfinite(prof) & (np.abs(prof) > 0) & (b >= 1.5)
                sl = np.polyfit(np.log(b[m]), np.log(np.abs(prof[m])), 1)[0] if m.sum() >= 3 else float("nan")
                print(f"  lam={lam:g} T={T:g}: d<u^2>(r) = " + " ".join(f"{v:+.2e}" for v in prof)
                      + f"  (r/a = " + ",".join(f"{x:.0f}" for x in b) + f");  показатель {sl:.2f}", flush=True)
        return
    Dmax = args.Dmax or (0.45 * n)
    Ds = [D for D in np.arange(2.0, Dmax + 1e-9, 1.0)]
    for lam in lams:
        s1 = potentials(freqs(build(n, 1.0, args.kp, [cen], args.r0, lam)), Ts)
        print(f"  lam={lam:g}: одиночное включение dE_zp={s1['E']-base['E']:+.6f}", flush=True)
        prev = {}
        for D in Ds:
            c1 = cen - np.array([D / 2, 0, 0]); c2 = cen + np.array([D / 2, 0, 0])
            H1, n1 = build(n, 1.0, args.kp, [c1], args.r0, lam, count=True)
            H2, n2 = build(n, 1.0, args.kp, [c2], args.r0, lam, count=True)
            Hp, npair = build(n, 1.0, args.kp, [c1, c2], args.r0, lam, count=True)
            if n1 + n2 != npair:
                print(f"    D={D:4.1f}: включения задевают общие связи ({n1}+{n2} != {npair}) — пропуск", flush=True)
                continue
            q1 = potentials(freqs(H1), Ts); q2 = potentials(freqs(H2), Ts); qp = potentials(freqs(Hp), Ts)
            uE = qp["E"] - q1["E"] - q2["E"] + base["E"]
            msg = f"    D={D:4.1f}: U_int(E_zp)={uE:+.3e}"
            for T in Ts:
                if T <= 0: continue
                uF = qp[("F", T)] - q1[("F", T)] - q2[("F", T)] + base[("F", T)]
                uS = qp[("S", T)] - q1[("S", T)] - q2[("S", T)] + base[("S", T)]
                msg += f"  F(T={T:g})={uF:+.3e}  dS(T={T:g})={uS:+.3e}"
            print(msg + f"   [{time.time()-t0:.0f} с]", flush=True)
    print("""
  Чтение: U_int растёт с D — притяжение, падает — отталкивание. dS — ответ теста 2 напрямую.
  Ньютоновскому 1/r соответствовал бы показатель спада 1; казимировскому — заметно больше.""")


if __name__ == "__main__":
    main()
