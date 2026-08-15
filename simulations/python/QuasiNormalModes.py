"""Квазинормальные моды узла (T-18): спектр малых колебаний вокруг
минимайзера — устойчивость, частоты, ширины.

Исходная постановка T-18 (комплексный спектр конических функций Лежандра
на каустике) опиралась на лестницу 4.5, которая отозвана (T-7, T-16).
Ядро предложения — «каждый уровень даёт Re (масса) и Im (ширина)» —
переносится на настоящий объект модели: минимайзер сектора n = 1
(TorusBVP, 2.8). Линеаризация вокруг него — задача Боголюбова - де Женна:

    W = (V + delta) e^{-i mu t},   delta = a e^{-i Om t} + conj(b) e^{+i conj(Om) t}
    L (a, b)^T = Om (a, b)^T,
    L = [[ H0 - mu + 2 U1 - U2 |V|^2 ...   ,   U1 V^2 - ... ],
         [ -conj(...)                        ,  -(...)      ]]

где H0 — линейный оператор сектора (лапласиан + центробежные + штраф div),
нелинейность f(rho) = -g rho + h rho^2 (rho = |W|^2), и
    d/dW* [f(rho) W]  ->  диагональ f(rho) + rho f'(rho),  внедиагональ  V^2 f'(rho).

Секторная структура: возмущение с азимутальным числом m живёт в паре
секторов (n+m, n-m); L строится из операторов этих двух секторов.
Спектр Om: вещественный => устойчиво (колебание), Im Om != 0 =>
экспоненциальная неустойчивость (ширина). Om = 0 — нулевые моды
(фаза, трансляция по z, вращение по phi — по одной на симметрию).

Что получено (Q = 740, 400, 330, 270; секторы m = 0, 1, 2) — С УЧЁТОМ
разбора нулевых мод в QNMSymmetries.py:
  1. Всё, что не нулевая мода симметрии, вещественно (Im ~ 1e-12..1e-16)
     при всех Q и m = 0, 1, 2: узел линейно и энергетически устойчив,
     упругой ширины НЕТ. Это и есть содержание 17 порядков между
     ширинами лептонов и адронных резонансов: распад лептона идёт по
     постороннему каналу (слабому, 4.13). Программа T-18 «Im -> ширина
     лептона» закрывается.
  2. Пары у нуля с |Om| ~ 0.01–0.08 (в том числе ЧИСТО МНИМЫЕ: m = 0 при
     Q <= 330, m = 1 при всех Q) — расщеплённые дискретизацией жордановы
     клетки нулевых мод симметрий (фаза, сдвиг z; сдвиги и наклоны при
     m = 1): на 82–100 % лежат в оболочке генераторов. Это НЕ порог
     устойчивости и НЕ перенос заряда (для m != 0 первые вариации Q, H
     обнуляются тождественно). Ранее заявленный порог Q_st снят.
  3. Настоящие частоты колебаний узла: ~0.09 (смешанная), 0.30, 0.41 при
     Q = 740; 0.22, 0.46 при Q = 270.

Замечание о динамическом тесте: прямое интегрирование в реальном
времени (Странг-расщепление + Кранк-Николсон) срывается на t ~ 60-80 —
накопление несоленоидальной ошибки штрафной схемы, не физика.

Запуск:  python QuasiNormalModes.py [m_max]
"""

import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import TorusBVP as T

HERE = os.path.dirname(os.path.abspath(__file__))


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def linear_op(n):
    """Линейный оператор сектора n БЕЗ штрафа? — со штрафом: тот же L, что
    в релаксации (штраф div — часть функционала, значит и линеаризации)."""
    L, Div, w, R, Z = T.build_operators(n)
    return L, w, R, Z


def bdg_spectrum(V, mu, n, m, nev=12):
    """Собственные значения БдЖ для возмущений азимутального числа m."""
    NR, NZ = T.NR, T.NZ
    N3 = 3 * NR * NZ
    La, w, R, Z = linear_op(n + m)
    Lb, _, _, _ = linear_op(n - m)
    rho = (np.abs(V) ** 2).sum(axis=0)             # (NR, NZ)
    f = -T.G_CUB * rho + T.H_QUI * rho ** 2         # f(rho)
    fp = -T.G_CUB + 2 * T.H_QUI * rho               # f'(rho)
    # диагональные блоки: H0 - mu + f + rho f' (на каждую компоненту одинаково)
    # + перекрёстные по компонентам: V_i conj(V_j) f' (a) и V_i V_j f' (b)
    # Оператор действует на a (3 компоненты) и b (3 компоненты).
    Vf = V.reshape(3, NR * NZ)
    fF = f.ravel(); fpF = fp.ravel()
    diagA = sp.diags(np.concatenate([fF] * 3) - mu)
    # блок связи компонент через f': M_ij = V_i conj(V_j) f'  (Aab), N_ij = V_i V_j f' (Bab)
    def comp_block(kind):
        blocks = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if kind == "A":
                    d = Vf[i] * np.conj(Vf[j]) * fpF
                else:
                    d = Vf[i] * Vf[j] * fpF
                blocks[i][j] = sp.diags(d)
        return sp.bmat(blocks, format="csc")
    A_cross = comp_block("A")
    B_cross = comp_block("B")
    Hp = La + diagA + A_cross                # действует на a
    Hm = Lb + diagA + A_cross                # действует на b (тот же вид, сектор n-m)
    # БдЖ: [[Hp, B],[-conj(B), -conj(Hm)]] (a,b)^T = Om (a,b)^T
    top = sp.hstack([Hp, B_cross])
    bot = sp.hstack([-B_cross.conj(), -Hm.conj()])
    Lbdg = sp.vstack([top, bot]).tocsc()
    # спектр вблизи нуля (нижние моды): shift-invert около sigma = 0.02
    try:
        vals = spla.eigs(Lbdg, k=nev, sigma=0.0 + 0.0j, which="LM",
                         return_eigenvectors=False, tol=1e-8, maxiter=4000)
    except Exception as e:  # noqa: BLE001
        print("  eigs failed:", e)
        return np.array([])
    return vals


def relax_and_mu(Q0):
    """Минимайзер n = 1 при заряде Q0 и его mu (без записи в json)."""
    T.Q_FIX = Q0
    L, Div, w, R, Z = T.build_operators(1)
    N3 = 3 * T.NR * T.NZ
    solver = spla.splu((sp.identity(N3, format="csc", dtype=complex) + T.DT * L).tocsc())
    V = T.seed_tube(1, R, Z)
    wf = np.concatenate([w.ravel()] * 3)
    Vf = V.reshape(N3)
    Vf = Vf * math.sqrt(Q0 / float((np.abs(Vf) ** 2 * wf).sum()))
    Ep, hits = np.inf, 0
    for it in range(8000):
        V = Vf.reshape(3, T.NR, T.NZ)
        rho = (np.abs(V) ** 2).sum(0)
        Vn = solver.solve((V + T.DT * (T.G_CUB * rho - T.H_QUI * rho ** 2) * V).reshape(N3))
        Vf = Vn * math.sqrt(Q0 / float((np.abs(Vn) ** 2 * wf).sum()))
        if it % 100 == 99:
            E = T.observables(Vf.reshape(3, T.NR, T.NZ), 1, Div, w, R, Z)["E"]
            if abs(E - Ep) < 1e-9 * abs(E):
                hits += 1
                if hits >= 3:
                    break
            else:
                hits = 0
            Ep = E
    V = Vf.reshape(3, T.NR, T.NZ)
    rho = (np.abs(V) ** 2).sum(0)
    act = (L @ Vf) - ((T.G_CUB * rho - T.H_QUI * rho ** 2) * V).reshape(N3)
    mu = float((np.real(np.conj(Vf) * act) * wf).sum()) / Q0
    return V, mu, E


def main(m_max=2, Qs=(740.0, 400.0, 330.0, 270.0)):
    head("КВАЗИНОРМАЛЬНЫЕ МОДЫ МИНИМАЙЗЕРА n = 1: УСТОЙЧИВОСТЬ ПО Q")
    print("""
  Собственные значения Om задачи Боголюбова - де Женна по азимутальному
  числу возмущения m (ближайшие к нулю). Re Om — частота, Im Om — темп
  роста. Om = 0 — нулевые моды симметрий. Печатается max|Im Om| и
  ближайшие моды.
  Моды у нуля опознаются в QNMSymmetries.py.
""")
    for Q0 in Qs:
        cache = os.path.join(HERE, f"_minimizer_n1_Q{int(Q0)}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            V = d["V"]
            if "mu" in d:
                mu = float(d["mu"])
            else:
                import json
                rec = json.load(open(os.path.join(HERE, "torusbvp_results.json")))
                mu = [r for r in rec.values() if r["n"] == 1 and abs(r["Q"] - Q0) < 1][0]["mu"]
        else:
            V, mu, _ = relax_and_mu(Q0)
        print(f"  Q = {Q0:.0f}, mu_1 = {mu:.4f}")
        for m in range(0, m_max + 1):
            vals = bdg_spectrum(V, mu, 1, m, nev=8)
            if len(vals) == 0:
                continue
            vals = vals[np.argsort(np.abs(vals))]
            maxim = float(np.max(np.abs(vals.imag)))
            flag = "устойчив" if maxim < 1e-6 else "НЕУСТОЙЧИВ"
            print(f"     m = {m}: max|Im Om| = {maxim:.2e}  {flag}   "
                  + ", ".join(f"{v.real:+.4f}{v.imag:+.1e}i" for v in vals[:4]))
        print()
    print("""
  Чтение. Пары у нуля (|Om| < ~0.08, вещественные или мнимые) — это
  расщеплённые сеткой нулевые моды симметрий, см. QNMSymmetries.py:
  не частоты и не неустойчивости. Всё остальное вещественно при всех Q
  и m: узел устойчив, ширины упругого сектора НЕТ — 17 порядков
  стабильности лептонов против резонансов означают распад по стороннему
  каналу (слабому, 4.13); программа T-18 «Im -> ширина» закрывается.
""")


if __name__ == "__main__":
    mm = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(mm)
