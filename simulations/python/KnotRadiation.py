"""Неизлучающая конфигурация? (100.6.3 п. 8) Спектр нелинейного источника узла n=1 на
резонансной оболочке |k| = k0 в бесщелевой среде поперечного звука.

Берётся минимайзер сектора n=1 (TorusBVP, _minimizer_n1_Q*.npz): V = (V_rho, V_phi, V_z)(rho,z) e^{i n phi}.
Источник для дальнего поля в бесщелевой среде: S = (-g|V|^2 + h|V|^4) V (нелинейность узла).
Считается трёхмерный фурье-образ S на сфере |k|=k0 (азимутальный интеграл — через J_m),
поперечная проекция (в несжимаемой среде излучает только поперечная часть), и
мощность P(k0) ∝ k0^2 ∫ |S_perp(k0 n)|^2 dOmega. Ищутся нули по k0 — «резонансные конфигурации».
Также Γ/ω в единицах профиля для k0 = ω (c=1): Γ/ω = ∫|S_perp|^2 dΩ /(32 π^2 ω Q).
"""
import sys, math, numpy as np
from scipy.special import jv
sys.path.insert(0, '.')
import TorusBVP as T

def shell_power(V, R, Z, w, n, k0, nth=61):
    g, h = T.G_CUB, T.H_QUI
    rho_d = (np.abs(V)**2).sum(axis=0)
    S = (-g*rho_d + h*rho_d**2) * V           # (S_rho, S_phi, S_z)
    cxp = 0.5*(S[0] + 1j*S[1]); cxm = 0.5*(S[0] - 1j*S[1])   # S_x: m=n+1, n-1
    cyp = 0.5*(S[1] - 1j*S[0]); cym = 0.5*(S[1] + 1j*S[0])   # S_y
    th = np.linspace(0, math.pi, nth)
    Pperp = np.zeros(nth); Ptot = np.zeros(nth)
    for i, t in enumerate(th):
        a, kz = k0*math.sin(t), k0*math.cos(t)
        ez = np.exp(-1j*kz*Z)
        def F(c, m):   # 2π ∫∫ c J_m(a rho) e^{-i kz z} rho drho dz ; w = 2π rho h^2
            return (-1j)**m * (c * jv(m, a*R) * ez * w).sum()
        Sx = F(cxp, n+1) + F(cxm, n-1)
        Sy = F(cyp, n+1) + F(cym, n-1)
        Sz = F(S[2], n)
        khat = np.array([math.sin(t), 0.0, math.cos(t)])
        Svec = np.array([Sx, Sy, Sz])
        tot = (np.abs(Svec)**2).sum()
        lon = abs((khat*Svec).sum())**2
        Ptot[i] = tot; Pperp[i] = tot - lon
    dOm = 2*math.pi*np.sin(th)
    Ip = np.trapezoid(Pperp*dOm, th); It = np.trapezoid(Ptot*dOm, th)
    return Ip, It, th, Pperp

for fn in ('_minimizer_n1_Q270.npz', '_minimizer_n1_Q740.npz'):
    d = np.load(fn); V = d['V']; R = d['R']; Z = d['Z']; w = d['w']
    n = 1
    L, Div, w2, R2, Z2 = T.build_operators(n)
    ob = T.observables(V, n, Div, w, R, Z)
    Q = ob['Q']; reff = ob['r_eff']; Rc = ob['R_core']
    mu = float(d['mu']) if 'mu' in d.files else float('nan')
    print(f"\n{fn}: Q={Q:.1f} E={ob['E']:.3f} mu={mu:.4f} R_core={Rc:.2f} r_eff={reff:.2f} aspect={ob['aspect']:.2f}")
    k0s = np.linspace(0.02, 3.0, 150)
    Ps = []; 
    for k0 in k0s:
        Ip, It, th, Pp = shell_power(V, R, Z, w, n, k0, nth=41)
        Ps.append((k0, Ip, It))
    Ps = np.array(Ps)
    Pk = Ps[:,0]**2 * Ps[:,1]          # ∝ излучаемая мощность (k0^2 от меры оболочки и ω k0 -> k0^2 при ω=k0)
    imax = Pk.argmax()
    print(f"  max power at k0={Ps[imax,0]:.3f} (k0*r_eff={Ps[imax,0]*reff:.2f}, k0*R_core={Ps[imax,0]*Rc:.2f})")
    # local minima of shell power
    rel = Pk/Pk.max()
    mins = [i for i in range(1,len(rel)-1) if rel[i]<rel[i-1] and rel[i]<rel[i+1]]
    for i in mins:
        print(f"  local min: k0={Ps[i,0]:.3f} k0*r_eff={Ps[i,0]*reff:.2f} k0*R_core={Ps[i,0]*Rc:.2f}  P/Pmax={rel[i]:.2e}  perp/total={Ps[i,1]/Ps[i,2]:.2f}")
    # Γ/ω at k0 = ω scan
    print("  k0   k0*r_eff  P/Pmax   Gamma/omega(k0=omega)")
    for k0 in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        i = np.argmin(abs(Ps[:,0]-k0)); k = Ps[i,0]
        gw = Ps[i,1] / (32*math.pi**2 * k * Q)
        print(f"  {k:.2f}  {k*reff:6.2f}  {rel[i]:.2e}  {gw:.2e}")
    # comparison: generic gaussian scalar source of the same r_eff
