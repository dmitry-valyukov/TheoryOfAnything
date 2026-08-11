import matplotlib.pyplot as plt
import numpy as np

N_points_x = 200
N_samples = 5000
x_vals = np.linspace(0, 4 * np.pi, N_points_x)

theory_ab = []
cshs_ab = []

def thorus_spin(phase, x):
  phase2 = np.cos(phase/2)*(np.pi/4)
  return np.sign(np.sin(phase - x/2)**2 + 2*np.cos(phase2/2 - x/2)**2 - 1.5)

for x in x_vals:
    phase = rng.uniform(0, 4 * np.pi, N_samples)

    A = thorus_spin(phase, 0)
    B = thorus_spin(phase, x)

    cshs_ab.append(np.mean(A * B))
    theory_ab.append(np.cos(x))

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(x_vals, theory_ab, label=r"Квантовая теория: $\cos(x)$", color="teal", lw=2)
plt.plot(x_vals, cshs_ab, label="Локальный CHSH с независимым ZPF", color="crimson")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
