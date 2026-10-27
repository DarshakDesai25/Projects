"""Robustness test. What happens when the real world does not match the simulator?

The CNN was trained on one simulator with a perfect lens, equal brightness sources,
a centered pair, 1000 photons and 2 background photons per pixel. Real cameras are
messier. This script tests the trained CNN and the classical maximum likelihood fit on
images that break those assumptions one at a time, and then all together.

Both methods see the same kind of shifted images. The classical fit still assumes the
nominal lens model, so it is mismatched too. That keeps the comparison fair.

Separations are always reported in nominal Rayleigh units (1.0 means 5 pixels), even
when the true lens is wider or narrower than the model.

Example
  python robustness.py                      # CNN and classical, about 15 minutes
  python robustness.py --no-classical       # CNN only, about 1 minute
  python robustness.py --n-classical 60     # faster classical comparison
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from psf import Simulator, airy

HARD = 0.6  # separations below this (in Rayleigh units) count as the hard region


@dataclass
class Condition:
    name: str
    psf_scale: float = 1.0    # 1.2 means the real lens blurs 20 percent more than the model
    astig: float = 1.0        # above 1 stretches the blur along x, below 1 along y
    blur_px: float = 0.0      # extra Gaussian blur in pixels, like defocus or vibration
    ratio: float = 0.5        # share of the light from the first source
    jitter_px: float = 1.0    # how far the pair center can wander from the image center
    photons: float = 1000.0
    background: float = 2.0


CONDITIONS = [
    Condition("matches training"),
    Condition("lens blurs 20% more", psf_scale=1.2),
    Condition("astigmatism", astig=1.35),
    Condition("defocus blur", blur_px=0.8),
    Condition("brightness 70/30", ratio=0.7),
    Condition("brightness 90/10", ratio=0.9),
    Condition("center off by up to 3 px", jitter_px=3.0),
    Condition("dim and noisy", photons=300.0, background=5.0),
    Condition("everything at once", astig=1.25, blur_px=0.5, ratio=0.7, jitter_px=2.0, photons=600.0, background=3.0),
]


class ShiftedSimulator(Simulator):
    """Same as the training simulator, but the lens can be wider, narrower or astigmatic."""

    def __init__(self, psf_scale=1.0, astig=1.0, **kw):
        super().__init__(**kw)
        self.ax = psf_scale * np.sqrt(astig)
        self.ay = psf_scale / np.sqrt(astig)

    def _psf(self, x, y):
        x = np.asarray(x, dtype=np.float64)[:, None, None]
        y = np.asarray(y, dtype=np.float64)[:, None, None]
        r = np.hypot((self.gx[None] - x) / self.ax, (self.gy[None] - y) / self.ay)
        fine = airy(r, self.rayleigh_px)
        n, s, k = fine.shape[0], self.size, self.ss
        coarse = fine.reshape(n, s, k, s, k).mean(axis=(2, 4))
        return coarse / coarse.sum(axis=(1, 2), keepdims=True)


def mean_images(cond, cx, cy, seps, theta):
    sim = ShiftedSimulator(psf_scale=cond.psf_scale, astig=cond.astig,
                           photons=cond.photons, background=cond.background)
    mu = sim.mean_image(cx, cy, seps, theta, ratio=cond.ratio)
    if cond.blur_px > 0:
        mu = gaussian_filter(mu, sigma=(0, cond.blur_px, cond.blur_px), mode="nearest")
    return mu


def make_data(cond, n, rng, sep_range=(0.05, 1.5), chunk=1000):
    labels = (rng.random(n) < 0.5).astype(np.int64)
    seps = np.where(labels == 1, rng.uniform(*sep_range, size=n), 0.0)
    theta = rng.uniform(0, np.pi, size=n)
    cx = rng.uniform(-cond.jitter_px, cond.jitter_px, size=n)
    cy = rng.uniform(-cond.jitter_px, cond.jitter_px, size=n)
    imgs = np.empty((n, 32, 32), dtype=np.float32)
    for i in range(0, n, chunk):
        j = slice(i, min(i + chunk, n))
        imgs[j] = rng.poisson(mean_images(cond, cx[j], cy[j], seps[j], theta[j]))
    return imgs, labels, seps.astype(np.float32)


# Metrics ------------------------------------------------------------------

def auc(scores, labels):
    """Chance that a two source image scores higher than a one source image."""
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(scores, sep_hat, labels, seps, rng, boot=200):
    def one(idx):
        l, s, sh, sp = labels[idx], scores[idx], sep_hat[idx], seps[idx]
        m = (l == 1) & (sp < HARD)
        rmse = np.sqrt(np.mean((sh[m] - sp[m]) ** 2)) if m.any() else np.nan
        return auc(s, l), rmse

    base = one(np.arange(len(labels)))
    draws = np.array([one(rng.integers(0, len(labels), len(labels))) for _ in range(boot)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {"auc": base[0], "auc_ci": [float(lo[0]), float(hi[0])],
            "rmse": float(base[1]), "rmse_ci": [float(lo[1]), float(hi[1])]}


# The two methods --------------------------------------------------------------

def load_cnn(path):
    import torch
    from model import ResolveNet
    from train import pick_device, predict
    from model import preprocess

    device = pick_device()
    model = ResolveNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    def run(imgs):
        prob, sep = predict(model, preprocess(imgs), device)
        return np.asarray(prob), np.asarray(sep)

    return run


def run_classical(imgs):
    """Maximum likelihood fit that assumes the nominal lens. Slow, so use few images."""
    from baseline_glrt import initial_guess, fit_one, fit_two

    sim = Simulator()
    stats, sep_hat = np.zeros(len(imgs)), np.zeros(len(imgs))
    for i, img in enumerate(imgs.astype(np.float64)):
        cx, cy, f0, b0, th0 = initial_guess(sim, img)
        n1 = fit_one(sim, img, cx, cy, f0, b0)
        n2, d = fit_two(sim, img, cx, cy, f0, b0, th0)
        stats[i] = 2 * max(n1 - n2, 0.0)
        sep_hat[i] = d
    return stats, sep_hat


def evaluate(cond, idx, n_cnn, n_cls, cnn_fn, rng, classical_fn=run_classical):
    imgs, labels, seps = make_data(cond, n_cnn, np.random.default_rng(1000 + idx))
    row = {"condition": asdict(cond)}
    if cnn_fn is not None:
        prob, sep_hat = cnn_fn(imgs)
        row["cnn"] = metrics(prob, sep_hat, labels, seps, rng)
        row["cnn"]["false_alarm_at_half"] = float(np.mean(prob[labels == 0] > 0.5))
        hard = (labels == 1) & (seps < HARD)
        row["cnn"]["detect_hard_at_half"] = float(np.mean(prob[hard] > 0.5))
    if n_cls > 0:
        sub = slice(0, n_cls)
        stats, sep_hat = classical_fn(imgs[sub])
        row["classical"] = metrics(stats, sep_hat, labels[sub], seps[sub], rng)
    return row


# Output -----------------------------------------------------------------------

def fmt(v, ci):
    return f"{v:.3f} ({ci[0]:.3f} to {ci[1]:.3f})"


def write_table(rows, path):
    lines = ["| Condition | CNN AUC | Classical AUC | CNN error, small d | Classical error, small d | CNN false alarm |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        c, k = r.get("cnn"), r.get("classical")
        lines.append("| " + " | ".join([
            r["condition"]["name"],
            fmt(c["auc"], c["auc_ci"]) if c else "n/a",
            fmt(k["auc"], k["auc_ci"]) if k else "n/a",
            fmt(c["rmse"], c["rmse_ci"]) if c else "n/a",
            fmt(k["rmse"], k["rmse_ci"]) if k else "n/a",
            f"{c['false_alarm_at_half']:.1%}" if c else "n/a"]) + " |")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def plot(rows, path):
    names = [r["condition"]["name"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, key, title, better in [(axes[0], "auc", "Telling one source from two (AUC)", "higher is better"),
                                   (axes[1], "rmse", "Separation error below 0.6 Rayleigh", "lower is better")]:
        for off, who, label, color in [(-0.18, "cnn", "CNN", "C1"), (0.18, "classical", "Classical fit", "C0")]:
            have = [i for i, r in enumerate(rows) if who in r]
            if not have:
                continue
            vals = np.array([rows[i][who][key] for i in have])
            ci = np.array([rows[i][who][key + "_ci"] for i in have])
            ax.barh(y[have] + off, vals, height=0.34, color=color, label=label,
                    xerr=[vals - ci[:, 0], ci[:, 1] - vals], error_kw={"lw": 1})
        ax.set_title(f"{title}\n{better}", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.5, 1.0)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def examples(path, sep=0.5):
    rng = np.random.default_rng(3)
    n = len(CONDITIONS)
    fig, ax = plt.subplots(1, n, figsize=(1.7 * n, 2.3))
    for a, cond in zip(ax, CONDITIONS):
        mu = mean_images(cond, np.zeros(1), np.zeros(1), np.full(1, sep), np.zeros(1))[0]
        a.imshow(rng.poisson(mu), cmap="magma")
        a.set_title(cond.name, fontsize=6.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"The same pair, separated by {sep} Rayleigh units, under each condition", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/resolvenet.pt")
    ap.add_argument("--n-cnn", type=int, default=4000)
    ap.add_argument("--n-classical", type=int, default=120)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cnn_fn = None if a.no_cnn else load_cnn(a.model)
    n_cls = 0 if a.no_classical else a.n_classical
    rng = np.random.default_rng(0)

    rows, t0 = [], time.time()
    for i, cond in enumerate(CONDITIONS):
        rows.append(evaluate(cond, i, a.n_cnn, n_cls, cnn_fn, rng))
        print(f"[{i + 1}/{len(CONDITIONS)}] {cond.name} done after {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "robustness.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print()
    print(write_table(rows, os.path.join(a.out, "robustness.md")))
    plot(rows, os.path.join(a.out, "robustness.png"))
    examples(os.path.join(a.out, "robustness_examples.png"))
    print("\nsaved robustness.json, robustness.md, robustness.png and robustness_examples.png to", a.out)


if __name__ == "__main__":
    main()
