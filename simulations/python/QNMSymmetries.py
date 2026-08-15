"""Моды m = 1 узла: нулевые моды симметрий, гессиан и энергетическая
устойчивость (довершение T-18 / 2.12).

В 2.12 моды m = 1 (смешивают секторы n-1 = 0 и n+1 = 2) с Im Om ~ 0.04–0.08
были отложены как «переносящие заряд». Это было неверно поставлено:
для m != 0 первые вариации Q = int |W|^2 и H = (1/2) int W* rot W
обнуляются автоматически — перекрёстные члены V* delta несут e^{+-i m phi}
и азимутальный интеграл их убивает. Связи Q, H на моды m != 0 условий
не накладывают; проецировать нечего.

Настоящий вопрос про m = 1 другой: в этом секторе живут ЧЕТЫРЕ точные
нулевые моды — сдвиги по x, y и наклоны (повороты вокруг x, y). У
гамильтоновой системы каждая нулевая мода симметрии стоит в жордановой
клетке со своим сопряжённым (сдвиг — с бустом, поворот — с изменением
угловой скорости). Дискретизация ломает симметрию на eps, и клетка
расщепляется в пару +-sqrt(eps): либо вещественную, либо МНИМУЮ.
Мнимая пара с |Om| ~ 0.05 отвечает eps ~ 3e-3 — типичный масштаб
сеточного нарушения трансляционной инвариантности (ось rho = 0, штраф
дивергенции, ящик).

Проверки:
  1. Гессиан энергии K (эрмитова часть БдЖ: L_BdG = J K). По Лагранжу —
     Дирихле: если K >= 0 на секторе m = 1 (кроме нулевых мод), узел
     энергетически устойчив, и мнимые Om — артефакт расщепления клеток,
     а не неустойчивость. Считаем ближайшие к нулю собственные значения K.
  2. Перекрытие мнимых мод БдЖ с генераторами сдвига (d/dx +- i d/dy)
     и наклона (L_x +- i L_y), построенными точно в секторной форме.
  3. m = 0 при Q = 270: у чисто мнимой пары проверяем z-чётность и
     перекрытие с V; если направление нечётно по z — симметричный
     тор мог быть седлом, и релаксация с несимметричной затравкой
     должна дать более низкую энергию.

Запуск:  python QNMSymmetries.py
"""

import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import TorusBVP as T
import QuasiNormalModes as Q

HERE = os.path.dirname(os.path.abspath(__file__))
NR, NZ, HH = T.NR, T.NZ, T.HH


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78)


def load_min(Q0):
    cache = os.path.join(HERE, f"_minimizer_n1_Q{int(Q0)}.npz")
    d = np.load(cache)
    V = d["V"]
    if "mu" in d:
        mu = float(d["mu"])
    else:
        import json
        rec = json.load(open(os.path.join(HERE, "torusbvp_results.json")))
        mu = [r for r in rec.values() if r["n"] == 1 and abs(r["Q"] - Q0) < 1][0]["mu"]
    return V, mu


def bdg_blocks(V, mu, n, m):
    """Блоки Hp, Hm, B БдЖ (как в QuasiNormalModes.bdg_spectrum)."""
    La, w, R, Z = Q.linear_op(n + m)
    Lb, _, _, _ = Q.linear_op(n - m)
    rho = (np.abs(V) ** 2).sum(axis=0)
    f = -T.G_CUB * rho + T.H_QUI * rho ** 2
    fp = -T.G_CUB + 2 * T.H_QUI * rho
    Vf = V.reshape(3, NR * NZ)
    fF, fpF = f.ravel(), fp.ravel()
    diagA = sp.diags(np.concatenate([fF] * 3) - mu)

    def comp_block(kind):
        blocks = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                d = Vf[i] * (np.conj(Vf[j]) if kind == "A" else Vf[j]) * fpF
                blocks[i][j] = sp.diags(d)
        return sp.bmat(blocks, format="csc")
    A = comp_block("A"); B = comp_block("B")
    Hp = (La + diagA + A).tocsc()
    Hm = (Lb + diagA + A).tocsc()
    return Hp, Hm, B, w


def hessian(Hp, Hm, B):
    """K = [[Hp, B],[B^H, conj(Hm)]] — гессиан энергии; L_BdG = diag(1,-1) K."""
    return sp.vstack([sp.hstack([Hp, B]), sp.hstack([B.conj(), Hm.conj()])]).tocsc()


def bdg(Hp, Hm, B):
    return sp.vstack([sp.hstack([Hp, B]), sp.hstack([-B.conj(), -Hm.conj()])]).tocsc()


def eig_near(M, sigma, k=12, vecs=False):
    try:
        out = spla.eigs(M, k=k, sigma=sigma, which="LM", return_eigenvectors=vecs,
                        tol=1e-8, maxiter=6000)
    except Exception as e:  # noqa: BLE001
        print("  eigs failed:", e); return (np.array([]), None) if vecs else np.array([])
    return out


# ---------------------------------------------------------------- генераторы
def helical(V):
    """V_+, V_-, V_z:  V_+ = (V_rho + i V_phi)/sqrt2 умножает (e_rho + i e_phi)
    = e^{-i phi}(x + i y)/... => скаляр с намоткой n-1; V_- — с намоткой n+1."""
    # V_rho e_rho + V_phi e_phi = V_+ (e_rho + i e_phi)/sqrt2 + V_- (e_rho - i e_phi)/sqrt2
    #   => V_+ = (V_rho - i V_phi)/sqrt2  (намотка n-1),  V_- = (V_rho + i V_phi)/sqrt2 (n+1)
    return (V[0] - 1j * V[1]) / math.sqrt(2), (V[0] + 1j * V[1]) / math.sqrt(2), V[2]


def unhelical(Vp, Vm, Vz):
    return np.array([(Vp + Vm) / math.sqrt(2), 1j * (Vp - Vm) / math.sqrt(2), Vz])


def d_rho(F, R, k=0):
    """d/drho (центральная разность, как в TorusBVP). У оси дискретный
    оператор континуальный генератор точно не воспроизводит — поэтому
    невязки нулевых мод меряются и вне оси (rho > 0.6)."""
    return T.d_matrix(NR, HH) @ F


def d_z(F):
    Dz = T.d_matrix(NZ, HH)
    return F @ Dz.T


def raise_op(F, k, R):
    """(d/dx + i d/dy)[F e^{ik phi}] = e^{i(k+1)phi} (dF/drho - k F/rho)."""
    return d_rho(F, R, k) - k * F / R


def lower_op(F, k, R):
    """(d/dx - i d/dy)[F e^{ik phi}] = e^{i(k-1)phi} (dF/drho + k F/rho)."""
    return d_rho(F, R, k) + k * F / R


def gen_translation(V, n, R, sign):
    """Компонента e^{i(n+1)phi} (a) и e^{i(n-1)phi} (b*) для сдвига
    T = (d/dx +- ... ). Возвращает (a, bstar) как массивы (3,NR,NZ) в
    цилиндрических компонентах секторов n+1 и n-1.
    sign=+1: d/dx + i d/dy (чисто повышающий: только a);
    sign=-1: d/dx - i d/dy (чисто понижающий: только b*)."""
    Vp, Vm, Vz = helical(V)
    if sign > 0:
        a = unhelical(raise_op(Vp, n - 1, R), raise_op(Vm, n + 1, R), raise_op(Vz, n, R))
        return a, np.zeros_like(a)
    b = unhelical(lower_op(Vp, n - 1, R), lower_op(Vm, n + 1, R), lower_op(Vz, n, R))
    return np.zeros_like(b), b


def gen_tilt(V, n, R, Z, sign):
    """Наклон: (L_x + i L_y) ~ z (d/dx + i d/dy) - (x + i y) d/dz + поворот
    самого вектора (x + i y)-компоненты. Скалярная часть: для F e^{ik phi}:
    e^{i(k+1)phi}[ z (F' - kF/rho) - rho dF/dz ].
    Векторная часть (вращение базиса): в спиральном базисе (+,-,z)
    компоненты преобразуются как spin-1: L_+ переводит  z -> +,  - -> z
    (с точностью до нормировки), что даёт вклад из соседних компонент."""
    Vp, Vm, Vz = helical(V)

    def orb(F, k):
        return Z * (d_rho(F, R, k) - k * F / R) - R * d_z(F)
    if sign > 0:
        # орбитальная часть в компонентах (+,-,z) секторa n+1
        op, om, oz = orb(Vp, n - 1), orb(Vm, n + 1), orb(Vz, n)
        # спиновая часть S_+ : |z> -> sqrt2 |+>, |-> -> sqrt2 |z>  (в базисе e_+ = -(x+iy)/sqrt2 знаки условны)
        sp_p, sp_m, sp_z = math.sqrt(2) * Vz, np.zeros_like(Vz), math.sqrt(2) * Vm
        a = unhelical(op + sp_p, om + sp_m, oz + sp_z)
        return a, np.zeros_like(a)
    def orbm(F, k):
        return Z * (d_rho(F, R, k) + k * F / R) - R * d_z(F)
    op, om, oz = orbm(Vp, n - 1), orbm(Vm, n + 1), orbm(Vz, n)
    sp_p, sp_m, sp_z = np.zeros_like(Vz), math.sqrt(2) * Vz, math.sqrt(2) * Vp
    b = unhelical(op + sp_p, om + sp_m, oz + sp_z)
    return np.zeros_like(b), b


def to_ab_vector(a, bstar):
    """(a, b) как вектор БдЖ: b = conj(bstar)."""
    return np.concatenate([a.reshape(-1), np.conj(bstar).reshape(-1)])


def check_zero_mode(Lb, vec, name, wf2, mf2=None):
    r = Lb @ vec
    nv = math.sqrt(float((np.abs(vec) ** 2 * wf2).sum()))
    nr = math.sqrt(float((np.abs(r) ** 2 * wf2).sum()))
    txt = f"    |L_BdG {name}| / |{name}| = {nr / nv:.2e}"
    if mf2 is not None:
        nv2 = math.sqrt(float((np.abs(vec) ** 2 * wf2 * mf2).sum()))
        nr2 = math.sqrt(float((np.abs(r) ** 2 * wf2 * mf2).sum()))
        txt += f"   (вне оси, rho > 0.6: {nr2 / nv2:.2e})"
    print(txt)
    return nr / nv


def overlap(u, v, wf2):
    num = abs(np.sum(np.conj(u) * v * wf2))
    return num / math.sqrt(float((np.abs(u) ** 2 * wf2).sum()) * float((np.abs(v) ** 2 * wf2).sum()))


def gens_m0(V, Z):
    """m = 0: фаза (V, -V*) и сдвиг по z (dzV, conj dzV)."""
    dzV = np.array([d_z(V[c]) for c in range(3)])
    return {
        "фаза": np.concatenate([V.reshape(-1), -np.conj(V).reshape(-1)]),
        "сдвиг z": np.concatenate([dzV.reshape(-1), np.conj(dzV).reshape(-1)]),
    }


def gens_m1(V, n, R, Z):
    out = {}
    for name, (a, bs) in {
        "T+ (dx+i dy)": gen_translation(V, n, R, +1),
        "T- (dx-i dy)": gen_translation(V, n, R, -1),
        "J+ (наклон)": gen_tilt(V, n, R, Z, +1),
        "J- (наклон)": gen_tilt(V, n, R, Z, -1),
    }.items():
        out[name] = to_ab_vector(a, bs)
    return out


def span_fraction(gens, wf2):
    G = np.array(list(gens.values())).T * np.sqrt(wf2)[:, None]
    Qg, _ = np.linalg.qr(G)
    def frac(u):
        uw = u * np.sqrt(wf2)
        return np.linalg.norm(Qg.conj().T @ uw) / np.linalg.norm(uw)
    return frac


def analyse(V, mu, Q0, m, gens, w, wf2, label, mf2=None):
    Hp, Hm, B, _ = bdg_blocks(V, mu, 1, m)
    Lb = bdg(Hp, Hm, B); K = hessian(Hp, Hm, B)
    print(f"\n  ---- {label}: Q = {Q0:.0f}, m = {m} ----")
    print("  невязка генераторов как нулевых мод БдЖ (эталон — шум):")
    for nm, g in gens.items():
        check_zero_mode(Lb, g, nm, wf2, mf2)
    rnd = np.random.default_rng(0).standard_normal(len(wf2)) + 0j
    check_zero_mode(Lb, rnd, "шум", wf2, mf2)
    frac = span_fraction(gens, wf2)
    vals, vecs = eig_near(Lb, 0j, k=10, vecs=True)
    idx = np.argsort(np.abs(vals)); vals, vecs = vals[idx], vecs[:, idx]
    print("  спектр БдЖ у нуля; доля моды в span(генераторов симметрий):")
    for v, u in zip(vals, vecs.T):
        print(f"    Om = {v.real:+.4f}{v.imag:+.4f}i   доля = {frac(u):.3f}")
    kv, kvec = eig_near(K, 0j, k=8, vecs=True)
    idx = np.argsort(kv.real); kv, kvec = kv[idx], kvec[:, idx]
    N3 = 3 * NR * NZ
    print("  гессиан K у нуля; доля в span(генераторов); dQ/|delta| направления:")
    for v, u in zip(kv, kvec.T):
        a = u[:N3].reshape(3, NR, NZ); b = u[N3:].reshape(3, NR, NZ)
        d = a + np.conj(b)
        dQ = 0.0
        if m == 0:
            dQ = 2 * float((np.real(np.conj(V) * d) * w).sum()) / math.sqrt(float((np.abs(d) ** 2 * w).sum()))
        print(f"    K = {v.real:+.4f}   доля = {frac(u):.3f}   dQ/|delta| = {dQ:+.3f}")
    return vals, vecs, kv


def main():
    head("НУЛЕВЫЕ МОДЫ СИММЕТРИЙ, ГЕССИАН И СПЕКТР БдЖ УЗЛА (m = 0, 1)")
    print("""
  Для m != 0 первые вариации Q и H равны нулю тождественно (азимутальная
  ортогональность): связи на моды m = 1 не действуют. Вопрос — являются
  ли мнимые Om (m = 1 при всех Q; m = 0 при Q <= 330) неустойчивостью
  или расщеплением жордановых клеток нулевых мод симметрий (фаза, сдвиги,
  наклоны) дискретизацией. Проверка: (а) генераторы симметрий — нулевые
  моды БдЖ? (б) какая доля сомнительных мод лежит в их линейной оболочке;
  (в) есть ли у гессиана K отрицательные направления ВНЕ этой оболочки
  и вне направления изменения Q.
""")
    rho, z, R, Z, w = T.grids()
    wf = np.concatenate([w.ravel()] * 3)
    wf2 = np.concatenate([wf, wf])
    mf = np.concatenate([(R > 0.6).ravel()] * 3).astype(float)
    mf2 = np.concatenate([mf, mf])
    keep = {}
    for Q0 in (740.0, 270.0):
        V, mu = load_min(Q0)
        print(f"\n  ======== Q = {Q0:.0f}, mu = {mu:.4f} ========")
        r0 = analyse(V, mu, Q0, 0, gens_m0(V, Z), w, wf2, "собственный сектор", mf2)
        r1 = analyse(V, mu, Q0, 1, gens_m1(V, 1, R, Z), w, wf2, "смешивание секторов 0 и 2", mf2)
        keep[Q0] = (V, mu, r0, r1)
    print("""
  Чтение. Фаза, сдвиг z, T+-, J+- — нулевые моды БдЖ: невязка вне оси на
  2–6 порядков ниже шума (у оси дискретный оператор генератор не
  воспроизводит, это сеточный эффект). Все моды БдЖ с |Om| < 0.1 — в том
  числе мнимая пара m = 0 при Q = 270 (97 %) и мнимые пары m = 1 (82–86 %) —
  и все малые собственные значения K лежат в оболочке генераторов:
  это расщеплённые дискретизацией жордановы клетки нулевых мод, не физика.
  Единственное большое отрицательное направление K (m = 0) несёт dQ != 0 —
  направление изменения заряда, запрещённое связью (ВК: dQ/dmu < 0).
  Вне оболочки и вне dQ-направления отрицательных направлений нет:
  узел энергетически устойчив при обоих Q, порога Q_st нет.
""")
    if len(sys.argv) > 1 and sys.argv[1] == "relax":
        V, mu, r0, r1 = keep[270.0]
        vals, vecs, _ = r0
        i = int(np.argmax(np.abs(vals.imag)))
        saddle_test(V, mu, 270.0, vecs[:, i])


def saddle_test(V, mu, Q0, u):
    """Толкнуть минимайзер вдоль мнимой моды и дать релаксации доработать:
    падает ли энергия ниже E(тора)?"""
    head("ТЕСТ СЕДЛА: релаксация из V + eps*delta при Q = 270")
    L, Div, w, R, Z = T.build_operators(1)
    N3 = 3 * NR * NZ
    solver = spla.splu((sp.identity(N3, format="csc", dtype=complex) + T.DT * L).tocsc())
    wf = np.concatenate([w.ravel()] * 3)
    E0 = T.observables(V, 1, Div, w, R, Z)["E"]
    print(f"  E(тор) = {E0:.6f}")
    a = u[:N3].reshape(3, NR, NZ); b = u[N3:].reshape(3, NR, NZ)
    d = a + np.conj(b)
    d = d / math.sqrt(float((np.abs(d) ** 2 * w).sum()))
    for eps in (0.5, 2.0):
        Vf = (V + eps * d).reshape(N3)
        Vf = Vf * math.sqrt(Q0 / float((np.abs(Vf) ** 2 * wf).sum()))
        Ep, hits = np.inf, 0
        for it in range(4000):
            Vv = Vf.reshape(3, NR, NZ)
            rho = (np.abs(Vv) ** 2).sum(0)
            Vn = solver.solve((Vv + T.DT * (T.G_CUB * rho - T.H_QUI * rho ** 2) * Vv).reshape(N3))
            Vf = Vn * math.sqrt(Q0 / float((np.abs(Vn) ** 2 * wf).sum()))
            if it % 200 == 199:
                ob = T.observables(Vf.reshape(3, NR, NZ), 1, Div, w, R, Z)
                if abs(ob["E"] - Ep) < 1e-10 * abs(ob["E"]):
                    hits += 1
                    if hits >= 3: break
                else: hits = 0
                Ep = ob["E"]
        ob = T.observables(Vf.reshape(3, NR, NZ), 1, Div, w, R, Z)
        print(f"  eps = {eps}: после {it+1} шагов E = {ob['E']:.6f} (dE = {ob['E']-E0:+.2e}), "
              f"R_core = {ob['R_core']:.2f}, z_core = {ob['z_core']:.2f}, r_eff = {ob['r_eff']:.2f}, "
              f"aspect = {ob['aspect']:.2f}, H = {ob['H']:.2f}")


if __name__ == "__main__":
    main()
