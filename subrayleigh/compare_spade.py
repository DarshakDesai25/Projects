"""Compare how precisely direct imaging and mode sorting can measure separation."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spade as sp

os.makedirs("results", exist_ok=True)
N = 1000                     # photons
R = sp.RAYLEIGH_IN_SIGMA     # sigma per Rayleigh unit
d_ray = np.linspace(0.02, 1.5, 120)
d = d_ray * R                # separation in units of sigma

crb = lambda fi: 1.0 / np.sqrt(N * fi) / R   # back to Rayleigh units

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(d_ray, crb(sp.fisher_direct(d)), "k-", lw=2, label="direct imaging (camera)")
ax.plot(d_ray, crb(sp.fisher_spade_full(d)), "C2-", lw=2, label="ideal SPADE, all modes")
ax.plot(d_ray, crb(sp.fisher_spade_binary(d)), "C0--", lw=1.8, label="binary SPADE, no crosstalk")
for eps, c in [(0.01, "C1"), (0.001, "C3")]:
    ax.plot(d_ray, crb(sp.fisher_spade_binary(d, crosstalk=eps)), ":", color=c, lw=1.8,
            label=f"binary SPADE, {eps:.1%} crosstalk")

# Monte Carlo check of simple estimators at a few separations
rng = np.random.default_rng(0)
pts = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
mc = {"direct": [], "full": [], "binary": []}
for v in pts:
    dv = v * R
    for key, est in [("direct", sp.estimate_direct(dv, N, 2000, rng)),
                     ("full", sp.estimate_spade_full(dv, N, 20000, rng)),
                     ("binary", sp.estimate_spade_binary(dv, N, 20000, rng))]:
        mc[key].append(np.sqrt(np.mean((est - dv) ** 2)) / R)
ax.plot(pts, mc["direct"], "ko", ms=6, label="simulated, camera estimator")
ax.plot(pts, mc["full"], "s", color="C2", ms=6, label="simulated, SPADE estimator")

ax.axvline(1.0, color="gray", ls="--", lw=1)
ax.text(1.02, 0.0065, "Rayleigh limit", color="gray")
ax.set_yscale("log")
ax.set_ylim(5e-3, 1.0)
ax.set_xlabel("true separation (Rayleigh units, approximate)")
ax.set_ylabel("best possible separation error\n(Rayleigh units, 1,000 photons)")
ax.set_title("Rayleigh's curse: cameras lose information as sources merge")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("results/spade_vs_direct.png", dpi=150)

print("separation | camera CRB | SPADE CRB | ratio  (Rayleigh units)")
for v in [0.05, 0.1, 0.2, 0.5, 1.0]:
    dv = np.array([v * R])
    a, b = crb(sp.fisher_direct(dv))[0], crb(sp.fisher_spade_full(dv))[0]
    print(f"{v:>9} | {a:10.4f} | {b:9.4f} | {a / b:5.1f}x")
print("Monte Carlo RMSE:", {k: np.round(vals, 4).tolist() for k, vals in mc.items()})
