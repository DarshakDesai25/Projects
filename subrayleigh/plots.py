"""Make the figures. Works with only the classical baseline, or with the CNN too."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from psf import Simulator

R = "results"
BINS = np.linspace(0.05, 1.5, 11)
CENTERS = 0.5 * (BINS[1:] + BINS[:-1])


def load(name):
    path = os.path.join(R, name)
    return dict(np.load(path)) if os.path.exists(path) else None


def binned(seps, values, labels):
    m = labels > 0.5
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = m & (seps >= lo) & (seps < hi)
        out.append(np.mean(values[sel]) if sel.any() else np.nan)
    return np.array(out)


def examples(sim):
    rng = np.random.default_rng(7)
    seps = [0.0, 0.25, 0.5, 0.75, 1.0, 1.4]
    fig, ax = plt.subplots(2, len(seps), figsize=(2.1 * len(seps), 4.4))
    for j, d in enumerate(seps):
        mu = sim.mean_image(0.0, 0.0, d, 0.0)[0]
        ax[0, j].imshow(mu, cmap="magma")
        ax[1, j].imshow(rng.poisson(mu), cmap="magma")
        ax[0, j].set_title("one source" if d == 0 else f"two, d = {d}")
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("noiseless"); ax[1, 0].set_ylabel(f"{int(sim.photons)} photons")
    fig.suptitle("Separation d in Rayleigh units. Below d = 1 the two sources blur into one spot.")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "examples.png"), dpi=150)


def detection(glrt, cnn):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if glrt is not None:
        said_two = (glrt["stats"] > glrt["threshold"]).astype(float)
        fa = said_two[glrt["labels"] < 0.5].mean()
        ax.plot(CENTERS, binned(glrt["seps"], said_two, glrt["labels"]), "o-",
                label=f"likelihood ratio test (false alarm {fa:.1%})")
    if cnn is not None:
        said_two = (cnn["prob"] > 0.5).astype(float)
        fa = said_two[cnn["labels"] < 0.5].mean()
        ax.plot(CENTERS, binned(cnn["seps"], said_two, cnn["labels"]), "s-",
                label=f"CNN (false alarm {fa:.1%})")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.text(1.02, 0.05, "Rayleigh limit", color="gray")
    ax.set_xlabel("true separation (Rayleigh units)")
    ax.set_ylabel("detection rate for two sources")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "detection_vs_separation.png"), dpi=150)


def separation_error(sim, glrt, cnn):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    d = np.linspace(0.05, 1.5, 40)
    crb = [1 / np.sqrt(sim.fisher_separation(v)) for v in d]
    ax.plot(d, crb, "k-", lw=1.5, label="classical Cramer Rao bound (direct imaging)")
    for res, name, mk in [(glrt, "maximum likelihood fit", "o-"), (cnn, "CNN", "s-")]:
        if res is None:
            continue
        sq = (res["sep_hat"] - res["seps"]) ** 2
        ax.plot(CENTERS, np.sqrt(binned(res["seps"], sq, res["labels"])), mk, label=name)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("true separation (Rayleigh units)")
    ax.set_ylabel("separation RMSE (Rayleigh units)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "separation_error.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    glrt, cnn = load("glrt.npz"), load("cnn.npz")
    ref = glrt or cnn
    photons = float(ref["photons"]) if ref else 1000.0
    background = float(ref["background"]) if ref else 2.0
    sim = Simulator(photons=photons, background=background)
    examples(sim)
    detection(glrt, cnn)
    separation_error(sim, glrt, cnn)
    print("saved figures to", R)
