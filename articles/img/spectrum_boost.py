"""Рис. к разделу 1 статьи «Энтропия»: что делает лоренц-буст с тремя изотропными спектрами.

Одно правило для всех панелей: спектральная интенсивность (плотность энергии на единицу
частоты и телесного угла) преобразуется как I'(w') = D^3 I(w'/D), где D — доплер-фактор
в данном направлении (инвариант I/w^3, Rybicki & Lightman, §4.9). Навстречу движению D>1,
вдогон D<1. Плоский спектр меняет высоту, планковский — температуру (T' = D T),
кубический w^3 совпадает сам с собой в любом направлении.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v = 0.38                      # v/c
D_fwd = np.sqrt((1 + v) / (1 - v))   # навстречу
D_bwd = 1.0 / D_fwd                  # вдогон
w = np.linspace(0.02, 10, 600)

def boost(I, D):
    return D**3 * I(w / D)

spectra = [
    ("Плоский спектр", lambda x: np.full_like(x, 1.0), "высота растёт/падает"),
    ("Планковский спектр", lambda x: x**3 / np.expm1(x / 2.0), "температура $T\\to DT$"),
    ("Кубический спектр $\\omega^3$", lambda x: 0.02 * x**3, "совпадает сам с собой"),
]

plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, (title, I, note) in zip(axs, spectra):
    ax.plot(w, I(w), color="#1f4e9c", lw=2.6, label="в покое")
    ax.plot(w, boost(I, D_fwd), "--", color="#c0392b", lw=1.8, label=f"буст навстречу, $D={D_fwd:.2f}$")
    ax.plot(w, boost(I, D_bwd), ":", color="#2e8b57", lw=2.2, label=f"буст вдогон, $D={D_bwd:.2f}$")
    ax.set_title(f"{title}\n({note})", fontsize=12)
    ax.set_xlabel("частота $\\omega$")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=9.5)
axs[0].set_ylabel("спектральная интенсивность $I(\\omega)$")
axs[0].set_ylim(0, 4.5)
fig.suptitle(f"Один и тот же буст $v={v}c$ по правилу $I'(\\omega')=D^3\\,I(\\omega'/D)$", y=1.02)
plt.tight_layout()
plt.savefig("spectrum_boost.png", dpi=200, bbox_inches="tight")
print("saved spectrum_boost.png", D_fwd, D_bwd)
