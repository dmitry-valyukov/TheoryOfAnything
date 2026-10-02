"""Мост «какая энтропия»: классическая взаимная информация по функции Вигнера против квантовой
энтропии запутанности на одной и той же сети (долг 5 главы 100, [100.14.4]).

Постановка. Гауссово состояние мод дрожания спаянной сети задаётся X = <uu>, P = <pp>
(EntanglementArea.build_XP). Для двух непересекающихся шаров A и B считаются две разные величины:

  квантовая взаимная информация   I_Q = S(A) + S(B) - S(AB),   S — энтропия запутанности
                                  (симплектические числа; так считала 7.9);
  классическая по Вигнеру         I_W = (1/2)[ ln det Sig_A + ln det Sig_B - ln det Sig_AB ],
                                  Sig = blkdiag(X, P) — ковариация положительной гауссовой
                                  функции Вигнера (картина стохастической электродинамики,
                                  клетка фазового объёма 2 pi hbar).

Если две меры совпадают по показателю и с точностью до множителя порядка единицы, классическая
трактовка дрожания как термостата с конфигурационной энтропией (100.12.5) законна. Если нет —
термостат квантовый, и 100.12.5 надо переписывать.

Запуск: python WignerBridge.py [--N 1024] [--seeds 1,2] [--T 0,0.1]
"""

import argparse, math, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from JammedSoftening import head  # noqa: E402
from EntanglementArea import build_XP, ent_entropy, ball_indices  # noqa: E402
from EntropyPairSign import build_net  # noqa: E402


def sel_of(idx):
    s = np.concatenate([3 * idx + c for c in range(3)]); s.sort(); return s


def logdet(M):
    sign, ld = np.linalg.slogdet(M)
    return ld if sign > 0 else np.nan


def wigner_mi(X, P, iA, iB):
    a, b = sel_of(iA), sel_of(iB)
    ab = np.concatenate([a, b])
    t = 0.0
    for M in (X, P):
        t += logdet(M[np.ix_(a, a)]) + logdet(M[np.ix_(b, b)]) - logdet(M[np.ix_(ab, ab)])
    return 0.5 * t


def quantum_mi(X, P, iA, iB):
    return ent_entropy(X, P, iA) + ent_entropy(X, P, iB) - ent_entropy(X, P, np.concatenate([iA, iB]))


def run(N, seeds, dphi, Ts, nrep):
    rows = []
    for seed in seeds:
        t0 = time.time()
        b, H0, dz = build_net(N, seed, dphi)
        d = 2 * b.rad.mean(); L = b.L
        print(f"\n  --- N={N} seed={seed} dz={dz:.2f} d={d:.3f} L={L:.2f}  [{time.time()-t0:.0f} с] ---", flush=True)
        rng = np.random.default_rng(7000 + seed)
        for T in Ts:
            Tabs = T * math.sqrt(H0.diagonal().mean())
            X, P, w = build_XP(H0, Tabs)
            print(f"    T={T:g} (абс. {Tabs:.4f}) — ковариации готовы [{time.time()-t0:.0f} с]", flush=True)
            # (1) соседние шары радиуса R, зазор 0.2d: закон площади для обеих мер
            for R in np.array([1.0, 1.3, 1.7, 2.2]) * d:
                if 2 * R + 0.2 * d > 0.42 * L: continue
                iq = []; iw = []; na = []
                for _ in range(nrep):
                    c = rng.uniform(0, L, 3)
                    off = np.array([2 * R + 0.2 * d, 0, 0])
                    iA = ball_indices(b, np.mod(c, L), R)
                    iB = ball_indices(b, np.mod(c + off, L), R)
                    if len(iA) < 6 or len(iB) < 6 or len(np.intersect1d(iA, iB)): continue
                    iq.append(quantum_mi(X, P, iA, iB)); iw.append(wigner_mi(X, P, iA, iB))
                    na.append(0.5 * (len(iA) + len(iB)))
                if not iq: continue
                rows.append(dict(N=N, seed=seed, T=T, kind="area", R=R / d, gap=0.2,
                                 I_Q=float(np.mean(iq)), I_W=float(np.mean(iw)), n=float(np.mean(na)),
                                 sQ=float(np.std(iq)), sW=float(np.std(iw))))
                print(f"      R/d={R/d:4.2f} (n_A~{np.mean(na):.0f}): I_Q={np.mean(iq):.4f}±{np.std(iq):.4f}  "
                      f"I_W={np.mean(iw):.4f}±{np.std(iw):.4f}  I_W/I_Q={np.mean(iw)/np.mean(iq):.3f}", flush=True)
            # (2) спад с расстоянием при фиксированном R
            R = 1.3 * d
            for gap in [0.2, 1.0, 2.0, 3.0]:
                off0 = 2 * R + gap * d
                if off0 > 0.42 * L: continue
                iq = []; iw = []
                for _ in range(nrep):
                    c = rng.uniform(0, L, 3)
                    iA = ball_indices(b, np.mod(c, L), R)
                    iB = ball_indices(b, np.mod(c + np.array([off0, 0, 0]), L), R)
                    if len(iA) < 6 or len(iB) < 6 or len(np.intersect1d(iA, iB)): continue
                    iq.append(quantum_mi(X, P, iA, iB)); iw.append(wigner_mi(X, P, iA, iB))
                if not iq: continue
                rows.append(dict(N=N, seed=seed, T=T, kind="dist", R=R / d, gap=gap,
                                 I_Q=float(np.mean(iq)), I_W=float(np.mean(iw)), n=0.0,
                                 sQ=float(np.std(iq)), sW=float(np.std(iw))))
                print(f"      зазор={gap:3.1f}d: I_Q={np.mean(iq):.4f}  I_W={np.mean(iw):.4f}  "
                      f"I_W/I_Q={np.mean(iw)/np.mean(iq):.3f}", flush=True)
    return rows


def summarise(rows):
    head("СВОДКА: мост «какая энтропия» — Вигнер против запутанности (долг 5)")
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["N"], r["seed"], r["T"], r["kind"])].append(r)
    for key, rs in sorted(g.items()):
        N, seed, T, kind = key
        if kind == "area":
            rs.sort(key=lambda r: r["R"])
            R = np.array([r["R"] for r in rs]); q = np.array([r["I_Q"] for r in rs]); wv = np.array([r["I_W"] for r in rs])
            aq = np.polyfit(np.log(R), np.log(q), 1)[0] if len(R) >= 3 else np.nan
            aw = np.polyfit(np.log(R), np.log(wv), 1)[0] if len(R) >= 3 else np.nan
            print(f"\n  N={N} seed={seed} T={T:g} соседние шары: I_Q ∝ R^{aq:.2f}, I_W ∝ R^{aw:.2f}; "
                  f"I_W/I_Q = " + ", ".join(f"{a/b:.2f}" for a, b in zip(wv, q)))
        else:
            rs.sort(key=lambda r: r["gap"])
            gp = np.array([r["gap"] for r in rs]); q = np.array([r["I_Q"] for r in rs]); wv = np.array([r["I_W"] for r in rs])
            print(f"  N={N} seed={seed} T={T:g} спад по зазору (d): " +
                  ", ".join(f"{g_:.1f}: I_Q={a:.4f}/I_W={bv:.4f}" for g_, a, bv in zip(gp, q, wv)))
    print("""
  Чтение. Площадной показатель ~2 у обеих мер и отношение I_W/I_Q порядка единицы — мост стоит:
  классическая конфигурационная энтропия дрожания и квантовая запутанность различаются множителем,
  а не законом. Расхождение в показателе означало бы, что термостат квантовый (100.12.5 переписать).
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--seeds", type=str, default="1,2")
    ap.add_argument("--dphi", type=float, default=3e-2)
    ap.add_argument("--T", type=str, default="0,0.1")
    ap.add_argument("--nrep", type=int, default=6)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    Ts = [float(s) for s in args.T.split(",")]
    head(f"МОСТ ЭНТРОПИИ: N={args.N}, затравки {seeds}, T={Ts}")
    rows = run(args.N, seeds, args.dphi, Ts, args.nrep)
    out = args.out or os.path.join(HERE, f"wigner_bridge_N{args.N}.npz")
    np.savez(out, rows=np.array(rows, dtype=object))
    summarise(rows)


if __name__ == "__main__":
    main()
