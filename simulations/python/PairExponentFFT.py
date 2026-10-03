"""Показатель спада взаимодействия двух слабых включений в БЕЗЩЕЛЕВОЙ среде — без диагонализации
(широкая часть долга 12, 100.17).

Зачем. Плотная диагонализация ящика $n^3$ не даёт показателя: у периодического безщелевого ящика
есть ИК-обрезание $\\sim2\\pi/L$, и уже при $D\\gtrsim L/2\\pi$ спад идёт быстрее истинного.
Поэтому показатель считается вторым порядком теории возмущений (формула лифшицевского типа),
где функция Грина берётся на большой решётке через БПФ.

Тождество: (1/2) Tr sqrt(H) = (1/2 pi) \\int_0^inf dxi Tr ln(1 + H/xi^2), откуда для H = H0+V1+V2
перекрёстный член второго порядка

    U_int(D) = -(1/2 pi) \\int_0^inf dxi Tr[ G(xi) V1 G(xi) V2 ],   G = (H0 + xi^2)^{-1}.

Для включения «массового» типа (диагональное V = kappa на узле): Tr[...] = kappa^2 G(D)^2.
Для включения «жёсткостного» типа (возмущение связи, V = kappa (e_a - e_b)(e_a - e_b)^T):
Tr[...] = kappa^2 [двойная конечная разность G по обоим концам связи]^2.

Ожидание: массовое включение даёт U ~ 1/D^3, жёсткостное — 1/D^7 (закон Казимира — Полдера:
производная связь подавляет низкие частоты). Среда модели — упругая, и массового члена у неё
нет по трансляционной инвариантности, поэтому она структурно сидит на 1/D^7.

Запуск: python PairExponentFFT.py [--n 128] [--nxi 400]
"""
import argparse, math
import numpy as np


def green(n, xi2):
    k = 2 * math.pi * np.fft.fftfreq(n)
    lam = (4 * np.sin(k / 2) ** 2)
    L = lam[:, None, None] + lam[None, :, None] + lam[None, None, :] + xi2
    inv = 1.0 / L
    inv[0, 0, 0] = 0.0   # однородная мода отброшена (как нулевая мода трансляций в среде)
    return np.fft.ifftn(inv).real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--nxi", type=int, default=400)
    ap.add_argument("--xmax", type=float, default=6.0)
    args = ap.parse_args()
    n, nxi = args.n, args.nxi
    Ds = np.array([4, 6, 8, 10, 12, 16, 20, 24, 28, 32])
    Ds = Ds[Ds < 0.45 * n]
    # сетка по xi: равномерно по log, плотно у нуля (вклад низких частот важен для массового типа)
    xi = np.concatenate([np.linspace(1e-4, 0.2, nxi // 2), np.geomspace(0.2, args.xmax, nxi // 2)])
    Imass = np.zeros(len(Ds)); Istiff = np.zeros(len(Ds))
    for i in range(len(xi) - 1):
        x0, x1 = xi[i], xi[i + 1]
        xm = 0.5 * (x0 + x1); w = x1 - x0
        G = green(n, xm ** 2)
        for j, D in enumerate(Ds):
            g = G[D, 0, 0]
            Imass[j] += w * g ** 2
            # жёсткостное: связь вдоль x на каждом конце -> двойная разность G
            dd = G[D, 0, 0] - G[D + 1, 0, 0] - G[D - 1, 0, 0] + G[D, 0, 0]
            Istiff[j] += w * dd ** 2
    Umass = -Imass / (2 * math.pi)
    Ustiff = -Istiff / (2 * math.pi)
    print(f"решётка {n}^3 (БПФ), узлов по xi: {len(xi)}")
    print("  D    U_масс(D)        U_жёстк(D)")
    for D, a, b in zip(Ds, Umass, Ustiff):
        print(f"  {D:3d}  {a:+.6e}   {b:+.6e}")
    # классический (высокотемпературный) предел: доминирует нулевая мацубаровская частота
    G0 = green(n, 1e-8)
    cm, cs = [], []
    for D in Ds:
        g = G0[D, 0, 0]
        dd = 2 * G0[D, 0, 0] - G0[D + 1, 0, 0] - G0[D - 1, 0, 0]
        cm.append(-0.5 * g ** 2); cs.append(-0.5 * dd ** 2)
    cm, cs = np.array(cm), np.array(cs)
    print("\n  классический предел (T >> 1/D, нулевая мацубаровская частота), в единицах T:")
    print("  D    U_масс            U_жёстк          локальные наклоны (масс / жёстк)")
    for j in range(len(Ds)):
        extra = ""
        if j < len(Ds) - 1:
            lm = math.log(cm[j] / cm[j + 1]) / math.log(Ds[j + 1] / Ds[j])
            ls = math.log(cs[j] / cs[j + 1]) / math.log(Ds[j + 1] / Ds[j])
            extra = f"   {-lm:6.2f} / {-ls:6.2f}"
        print(f"  {Ds[j]:3d}  {cm[j]:+.5e}   {cs[j]:+.5e}{extra}")
    print("  (ожидание: массовое -2, жёсткостное -6)")

    print("\n  локальные наклоны (по соседним точкам): D-окно | массовое | жёсткостное")
    for j in range(len(Ds) - 1):
        lm = math.log(Umass[j] / Umass[j + 1]) / math.log(Ds[j + 1] / Ds[j])
        ls = math.log(Ustiff[j] / Ustiff[j + 1]) / math.log(Ds[j + 1] / Ds[j])
        print(f"    {Ds[j]:3d}-{Ds[j+1]:3d}   {-lm:6.2f}   {-ls:6.2f}")
    m = (Ds >= 8) & (Ds <= 0.2 * n)
    if m.sum() >= 3:
        sm = np.polyfit(np.log(Ds[m]), np.log(-Umass[m]), 1)[0]
        ss = np.polyfit(np.log(Ds[m]), np.log(-Ustiff[m]), 1)[0]
        print(f"\n  фит по 8 <= D <= {0.2*n:.0f} (вне влияния периодических образов): "
              f"массовое {sm:.2f} (ожидание -3), жёсткостное {ss:.2f} (ожидание -7)")
    print("""
  Чтение. Знак отрицателен в обоих случаях — притяжение (согласуется с прямым счётом 100.17.2).
  Показатель задаётся типом связи источника со средой, а не величиной включения. Упругая среда
  не имеет массового члена (трансляционная инвариантность), поэтому ей доступен только
  жёсткостной канал — и это 1/D^7. Ньютоновское 1/D ни тем, ни другим не получается.""")


if __name__ == "__main__":
    main()
