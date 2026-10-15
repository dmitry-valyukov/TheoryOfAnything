"""T-34 (100.6.3): числа для электронного узла в балансе с термостатом.

Входы: hbar = rho d^4 c (нормировка книги, 1.5/1.8), E = hbar*omega (баланс Бойера, 2.10),
размер узла L = c/omega в единицах ячеек, кубично-квинтическая форма (2.5).
Печатает: амплитуду узла u = d^2/L (смещение и деформацию), иерархию коэффициентов
(m^2, g, h) = omega_D^2 (alpha_G, 1, 1/alpha_G) в единицах u/d и в единицах деформации,
инвариантную по уровню частоту omega_x = m^2 d/c, которой требует m^2 ∝ c, проверку
источников книги (Хехл — Датта, ангармонизм, синус-Гордон, слом Гука), бракет щели из
устойчивости вакуума (x = m^2 h/g^2 >= 3/16) при Lambda_e = 0.05..0.075 и давление,
которое потребовала бы развилка «ячейка масштабируется с уровнем».
Чистая арифметика, без моделирования.
"""
import math
hbar=1.054571817e-34; c=2.99792458e8; G=6.67430e-11
me=9.1093837e-31; lP=math.sqrt(hbar*G/c**3)
lamC=hbar/(me*c)                      # reduced Compton length
we=me*c**2/hbar                       # electron frequency
alphaG=G*me**2/(hbar*c)
print(f"l_P={lP:.3e} m  lamC={lamC:.3e} m  omega_e={we:.3e} 1/s  alpha_G={alphaG:.3e}")
for d_over_lP in (1.0, 2.9):
    d=d_over_lP*lP
    rho=hbar/(d**4*c)                 # book normalization
    wD=c/d
    L=lamC                            # knot size in the Boyer balance: L=c/omega
    u=d**2/L                          # displacement amplitude from E=hbar*omega
    F=u/d; eps=u/L
    # check: E = rho*omega^2*u^2*L^3
    E=rho*we**2*u**2*L**3
    print(f"\n d={d_over_lP} l_P: rho={rho:.2e} kg/m^3  omega_D={wD:.2e} 1/s")
    print(f"   u=d^2/L={u:.2e} m  F=u/d={F:.2e}  strain=(d/L)^2={eps:.2e}   E/(hbar w_e)={E/(hbar*we):.3f}")
    print(f"   hierarchy m^2:g:h = omega_D^2 x ({we**2/wD**2:.1e} : 1 : {(L/d)**2:.1e})  [u/d units]")
    print(f"   same in strain units: g_strain/omega_D^2={(L/d)**2:.1e}, h_strain/omega_D^2={(L/d)**6:.1e}")
    for mfac in (1.0,1.29):
        m=mfac*we; wx=m**2*d/c
        print(f"   m={mfac} w_e: invariant omega_x=m^2 d/c={wx:.3e} 1/s, hbar*omega_x={hbar*wx/1.602e-19:.2e} eV, period={2*math.pi/wx:.0f} s")
    # sources
    print(f"   Hehl-Datta cubic/mass ~ l_P^2/lamC^2 = {lP**2/lamC**2:.1e}  (needed O(1))")
    print(f"   lattice anharmonicity on strain, |a|<=1e5: relative {(d/L)**4*1e5:.1e} (needed O(1))")
    print(f"   sine-Gordon cubic of gap potential: relative m^2/6/omega_D^2 ~ {we**2/6/wD**2:.1e}")
    print(f"   Hooke break sextic (1/omega_D^2): short by (L/d)^2 = {(L/d)**2:.1e}")
    print(f"   needed sextic scale: 1/m_e^2 i.e. h = omega_D^2 (L/d)^2 = {wD**2*(L/d)**2:.2e} 1/s^2 vs omega_D^2={wD**2:.2e}")
# gap bracket from vacuum stability: x=m^2 h/g^2 >= 3/16 ; omega^2/m^2 = 1 - Lambda/x
print("\nGap bracket (stable vacuum x>=3/16):")
for Lam in (0.05,0.06,0.075):
    r=1-Lam/(3/16)
    print(f"   Lambda_e={Lam}: omega_e^2/m^2 >= {r:.3f}  -> m/omega_e <= {1/math.sqrt(r):.3f}, hbar m <= {0.51099895/math.sqrt(r):.3f} MeV")
# cell-compression fork: pressure needed to shrink cells by U/2 at Earth's surface
U=G*5.972e24/(6.371e6*c**2)
print(f"\nFork 'cell scales with level': U_Earth={U:.2e}, dV/V=1.5U={1.5*U:.1e}; with K/mu>=1e22 pressure p/mu>={1.5*U*1e22:.1e}")
