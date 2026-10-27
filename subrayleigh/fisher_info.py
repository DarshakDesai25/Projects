"""Fisher information for the separation of two incoherent point sources seen by a camera.

Fisher information says how much one image can tell you about the separation. The
Cramer Rao bound turns it into the smallest error any unbiased method can reach, which
is 1 over the square root of the information.

The first version of this project plotted a bound that assumed everything except the
separation was known. A real measurement also has to find the center of the pair, its
orientation, the total brightness and the background light, and sometimes the brightness
split between the two sources. Not knowing these costs information. This script computes
the bound for each of these cases.

Model, the same as the simulator in psf.py
  incoherent sources, so intensities add and the phase between them does not matter
  Airy pattern for a circular lens, integrated over square pixels
  Poisson photon noise plus a flat background
  32 by 32 pixels, Rayleigh radius of 5 pixels, 1000 signal photons, 2 background per pixel

Usage
  python fisher_info.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psf import Simulator

ALL = ("cx", "cy", "sep", "theta", "flux", "bg", "ratio")
STEPS = dict(cx=1e-3, cy=1e-3, sep=1e-3, theta=1e-3, flux=1.0, bg=1e-2, ratio=1e-3)


def information_matrix(sim, sep, ratio=0.5, free=("sep",), theta=0.0):
    """Fisher information matrix for the parameters in `free`, at the true values given."""
    truth = dict(cx=0.0, cy=0.0, sep=sep, theta=theta, flux=sim.photons, bg=sim.background, ratio=ratio)

    def mean(p):
        return sim.mean_image(p["cx"], p["cy"], p["sep"], p["theta"], p["flux"], p["bg"], p["ratio"])[0].ravel()

    mu = mean(truth)
    grads = []
    for name in free:
        up, dn = dict(truth), dict(truth)
        up[name] += STEPS[name]
        dn[name] -= STEPS[name]
        grads.append((mean(up) - mean(dn)) / (2 * STEPS[name]))
    g = np.array(grads)                     # parameters by pixels
    return (g / mu) @ g.T                   # sum over pixels of g_i g_j / mu


def separation_bound(sim, sep, ratio=0.5, free=("sep",)):
    """Smallest possible standard deviation for the separation, in Rayleigh units."""
    info = information_matrix(sim, sep, ratio, free)
    inv = np.linalg.inv(info)
    return float(np.sqrt(inv[free.index("sep"), free.index("sep")])), float(1.0 / inv[free.index("sep"), free.index("sep")])


CASES = [
    ("only separation unknown (old bound)", 0.5, ("sep",)),
    ("also center, angle, brightness, background unknown", 0.5, ("cx", "cy", "sep", "theta", "flux", "bg")),
    ("same, and the brightness split unknown", 0.5, ALL),
    ("70/30 sources, split known", 0.7, ("cx", "cy", "sep", "theta", "flux", "bg")),
    ("70/30 sources, split unknown", 0.7, ALL),
    ("90/10 sources, split known", 0.9, ("cx", "cy", "sep", "theta", "flux", "bg")),
    ("90/10 sources, split unknown", 0.9, ALL),
]


def main():
    os.makedirs("results", exist_ok=True)
    sim = Simulator(photons=1000, background=2)
    seps = [0.1, 0.25, 0.5, 1.0]

    lines = ["| Case | " + " | ".join(f"d = {d}" for d in seps) + " |",
             "| --- | " + " | ".join("---" for _ in seps) + " |"]
    for name, ratio, free in CASES:
        cells = []
        for d in seps:
            sd, info = separation_bound(sim, d, ratio, free)
            shown = f"{info:.1f}" if info < 10 else f"{info:.0f}"
            cells.append(f"{sd:.3f} (info {shown})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("Smallest possible error in Rayleigh units, with the effective Fisher information in brackets\n")
    print(table)
    with open("results/fisher_information.md", "w") as f:
        f.write(table + "\n")

    d = np.linspace(0.05, 1.5, 40)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for (name, ratio, free), style in zip([CASES[0], CASES[1], CASES[2], CASES[4], CASES[6]],
                                         ["k-", "C0-", "C0--", "C3-", "C3--"]):
        ax.plot(d, [separation_bound(sim, v, ratio, free)[0] for v in d], style, label=name, lw=1.8)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("true separation (Rayleigh units)")
    ax.set_ylabel("smallest possible separation error (Rayleigh units)")
    ax.set_title("What a camera can measure, depending on what else is unknown")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/fisher_information.png", dpi=150)
    print("\nsaved results/fisher_information.md and results/fisher_information.png")


if __name__ == "__main__":
    main()
