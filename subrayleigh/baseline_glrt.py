"""Classical baseline: maximum likelihood model fitting with a likelihood ratio test.

For each image we fit two models with Poisson maximum likelihood
  model 1  one source   (x, y, flux, background)
  model 2  two sources  (centroid x, centroid y, separation, angle, flux, background)
The statistic 2 * (NLL1 - NLL2) decides one vs two sources, and model 2's
fitted separation is the classical separation estimate.

This is the standard approach an optics person would use, so it is the bar
the CNN has to meet.
"""
import argparse, json, time
import numpy as np
from scipy.optimize import minimize
from psf import Simulator


def nll(mu, k):
    mu = np.maximum(mu, 1e-9)
    return float(np.sum(mu - k * np.log(mu)))


def initial_guess(sim, img):
    bg0 = max(float(np.median(img[[0, -1], :])), 0.1)
    sig = np.clip(img - bg0, 0, None)
    flux0 = max(float(sig.sum()), 1.0)
    c = np.arange(sim.size) + 0.5 - sim.size / 2.0
    gx, gy = np.meshgrid(c, c)
    cx = float((sig * gx).sum() / flux0)
    cy = float((sig * gy).sum() / flux0)
    # orientation from second moments
    mxx = float((sig * (gx - cx) ** 2).sum() / flux0)
    myy = float((sig * (gy - cy) ** 2).sum() / flux0)
    mxy = float((sig * (gx - cx) * (gy - cy)).sum() / flux0)
    theta0 = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    return cx, cy, flux0, bg0, theta0


def fit_one(sim, img, cx, cy, flux0, bg0):
    def f(p):
        mu = sim.mean_image(p[0], p[1], 0.0, 0.0, np.exp(p[2]), np.exp(p[3]))[0]
        return nll(mu, img)
    r = minimize(f, [cx, cy, np.log(flux0), np.log(bg0)], method="L-BFGS-B")
    return r.fun


def fit_two(sim, img, cx, cy, flux0, bg0, theta0, starts=(0.4, 1.0)):
    def f(p):
        mu = sim.mean_image(p[0], p[1], p[2], p[3], np.exp(p[4]), np.exp(p[5]))[0]
        return nll(mu, img)
    best = None
    bounds = [(None, None), (None, None), (0.02, 3.0), (None, None), (None, None), (None, None)]
    # several starting points avoid local minima in separation and angle
    for d0 in starts:
        for th in (theta0, theta0 + np.pi / 2):
            r = minimize(f, [cx, cy, d0, th, np.log(flux0), np.log(bg0)],
                         method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
    return best.fun, float(best.x[2])


def run(n, seed, photons, background, out):
    sim = Simulator(photons=photons, background=background)
    rng = np.random.default_rng(seed)
    imgs, labels, seps, _ = sim.sample(n, rng)
    stats, sep_hat = np.zeros(n), np.zeros(n)
    t = time.time()
    for i, img in enumerate(imgs.astype(np.float64)):
        cx, cy, f0, b0, th0 = initial_guess(sim, img)
        n1 = fit_one(sim, img, cx, cy, f0, b0)
        n2, d = fit_two(sim, img, cx, cy, f0, b0, th0)
        stats[i] = 2 * max(n1 - n2, 0.0)
        sep_hat[i] = d
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n} images, {time.time() - t:.0f}s")
    # pick the decision threshold on the first half, report on the second half
    half = n // 2
    cand = np.unique(stats[:half])
    accs = [np.mean((stats[:half] > c) == labels[:half]) for c in cand]
    thr = float(cand[int(np.argmax(accs))])
    np.savez(out, stats=stats[half:], sep_hat=sep_hat[half:], labels=labels[half:],
             seps=seps[half:], threshold=thr, photons=photons, background=background)
    acc = np.mean((stats[half:] > thr) == labels[half:])
    print(json.dumps({"threshold": thr, "test_accuracy": round(float(acc), 4)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--photons", type=float, default=1000.0)
    ap.add_argument("--background", type=float, default=2.0)
    ap.add_argument("--out", default="results/glrt.npz")
    a = ap.parse_args()
    import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    run(a.n, a.seed, a.photons, a.background, a.out)
