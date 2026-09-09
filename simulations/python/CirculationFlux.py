"""Монополь плотности циркуляции у узла: проверка кандидата «керн = ёж циркуляции» (T-27).

Гипотеза (обсуждение онтологии, гл. 3.6): заряд может быть не потоком самого смещения
(закрыт теоремой о нулевом потоке — линейный эффект), а потоком КОМПОЗИТНОЙ, квадратичной
по u величины — плотности циркуляции (спиновой плотности)
    s = <u x du/dt>_t = (w/2) Im(V* x V)     (u = Re[V(rho,z) e^{i n phi} e^{-iwt}], n=1),
которая у гирации стационарна и на теорему не попадает (фазовый множитель сокращается).
Ёж поля s (радиальный монополь, поток != 0, знак = наружу/внутрь) дал бы: квантование
заряда степенью отображения, знак без хиральности, рождение парой (ёж+антиёж).

Считается на минимайзерах сектора n=1 (TorusBVP, Q=740 и 270), w=1 (ед.):
  s_rho = Im(V_phi* V_z),  s_phi = Im(V_z* V_rho),  s_z = Im(V_rho* V_phi);
  поток Phi(r) = int s_r dA по сферическим оболочкам; разложение s_r на оболочке на
  монополь (среднее) и диполь (момент ~cos theta); полный спин S_z = int s_z dV;
  та же диагностика для псевдовектора завихрённости circ = Im(V* x rot V) не считается
  (нужны производные) — только смещенческая s.

Запуск: python CirculationFlux.py
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def analyse(tag):
    z = np.load(os.path.join(HERE, f"_minimizer_n1_{tag}.npz"))
    V, R, Z, w = z["V"], z["R"], z["Z"], z["w"]
    Rs = np.sqrt(R ** 2 + Z ** 2)
    ct = np.divide(Z, Rs, out=np.zeros_like(Z), where=Rs > 0)
    s_rho = np.imag(np.conj(V[1]) * V[2])
    s_phi = np.imag(np.conj(V[2]) * V[0])
    s_z = np.imag(np.conj(V[0]) * V[1])
    s_r = s_rho * R / Rs + s_z * Z / Rs
    Sz = (s_z * w).sum()
    Q = ((np.abs(V) ** 2).sum(axis=0) * w).sum()
    print(f"\n== минимайзер n=1, {tag}:  Q = {Q:.1f},  S_z = int s_z dV = {Sz:.3f},  S_z/Q = {Sz/Q:.3f} ==")
    print(f"  {'r':>4} {'Phi=int s_r dA':>15} {'int |s_r| dA':>13} {'монополь/|.|':>13} {'диполь<s_r ct>':>15} {'int s_z dA':>11}")
    dr = 0.15
    for rs in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        sh = (Rs > rs - dr / 2) & (Rs <= rs + dr / 2)
        if sh.sum() < 8:
            continue
        A = w[sh] / dr
        Phi = (s_r[sh] * A).sum()
        Aabs = (np.abs(s_r[sh]) * A).sum()
        dip = (s_r[sh] * ct[sh] * A).sum()
        Fz = (s_z[sh] * A).sum()
        mono = Phi / Aabs if Aabs > 0 else 0.0
        print(f"  {rs:>4.1f} {Phi:>15.3e} {Aabs:>13.3e} {mono:>13.3f} {dip:>15.3e} {Fz:>11.3e}")
    # чётность по z: s_r(theta) против s_r(pi-theta)
    sh = (Rs > 2.0 - dr / 2) & (Rs <= 2.0 + dr / 2)
    up = (s_r[sh])[Z[sh] > 0].sum(); dn = (s_r[sh])[Z[sh] < 0].sum()
    print(f"  чётность на r=2: сумма s_r при z>0 = {up:.3e}, при z<0 = {dn:.3e}  (диполь: противоположные знаки)")


if __name__ == "__main__":
    print("Плотность циркуляции s = (w/2) Im(V* x V) на минимайзерах узла (w=1).")
    print("Ёж (заряд-кандидат) => монопольная доля порядка 1; спин => дипольная структура, поток ~ 0.")
    for tag in ("Q740", "Q270"):
        analyse(tag)
