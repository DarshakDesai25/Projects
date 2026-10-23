"""Direct imaging vs spatial mode demultiplexing (SPADE) for two point sources.

Model: two equally bright incoherent sources at +d/2 and -d/2 along x, with a
Gaussian point spread function. The intensity PSF has standard deviation sigma,
so the field amplitude is exp(-x^2 / (4 sigma^2)).

SPADE sorts each photon into Hermite Gaussian modes HG_q instead of pixels.
For a source offset by s from the mode center, a photon lands in mode q with
Poisson probability exp(-Q) Q^q / q!, where Q = s^2 / (4 sigma^2). Both sources
sit at |s| = d/2, so Q = d^2 / (16 sigma^2) for every photon.

Reference: Tsang, Nair and Lu, Physical Review X 6, 031033 (2016).
Crosstalk model follows the idea in Gessner, Fabre and Treps, PRL 125, 100501 (2020).

Assumptions: the centroid and the separation axis are known and the sorter is
aligned to the centroid. Real devices must estimate these first.
"""
import numpy as np

# For an Airy pattern, the best fit Gaussian has sigma of about 0.42 lambda N and
# the Rayleigh radius is 1.22 lambda N, so one Rayleigh unit is about 2.9 sigma.
RAYLEIGH_IN_SIGMA = 1.22 / 0.42


def q_of(d, sigma=1.0):
    return d ** 2 / (16.0 * sigma ** 2)


# Fisher information per photon for the separation d -----------------------

def fisher_spade_full(d, sigma=1.0):
    """Ideal sorting into all HG modes. Constant: no Rayleigh's curse."""
    return np.full_like(np.asarray(d, dtype=float), 1.0 / (4.0 * sigma ** 2))


def fisher_spade_binary(d, sigma=1.0, crosstalk=0.0):
    """Only two outcomes: photon in HG0 or not. crosstalk is the chance a photon
    that belongs in HG0 leaks into the other port."""
    d = np.asarray(d, dtype=float)
    Q = q_of(d, sigma)
    p = (1 - np.exp(-Q)) * (1 - crosstalk) + np.exp(-Q) * crosstalk
    dp_dd = (1 - 2 * crosstalk) * np.exp(-Q) * d / (8.0 * sigma ** 2)
    return dp_dd ** 2 / (p * (1 - p))


def fisher_direct(d, sigma=1.0, pixel=0.25, half_width=8.0):
    """Direct intensity imaging along x with fine pixels and no background."""
    d = np.asarray(d, dtype=float)
    edges = np.arange(-half_width, half_width + pixel, pixel) * sigma
    from scipy.special import erf

    def cdf(x):
        return 0.5 * (1 + erf(x / (np.sqrt(2) * sigma)))

    def probs(dd):
        a, b = edges[:-1], edges[1:]
        p1 = cdf(b - dd / 2) - cdf(a - dd / 2)
        p2 = cdf(b + dd / 2) - cdf(a + dd / 2)
        return 0.5 * (p1 + p2)

    out = []
    for dd in np.atleast_1d(d):
        eps = 1e-4 * sigma
        p = probs(dd)
        dp = (probs(dd + eps) - probs(max(dd - eps, 0.0))) / (dd + eps - max(dd - eps, 0.0))
        keep = p > 1e-15
        out.append(np.sum(dp[keep] ** 2 / p[keep]))
    return np.array(out)


# Simulated measurements and estimators -------------------------------------

def estimate_spade_full(d, n_photons, trials, rng, sigma=1.0):
    """Each photon's mode number is Poisson(Q). The MLE of Q is the mean mode number."""
    Q = q_of(d, sigma)
    total = rng.poisson(n_photons * Q, size=trials)  # sum of mode numbers
    Q_hat = total / n_photons
    return 4.0 * sigma * np.sqrt(Q_hat)


def estimate_spade_binary(d, n_photons, trials, rng, sigma=1.0, crosstalk=0.0):
    Q = q_of(d, sigma)
    p = (1 - np.exp(-Q)) * (1 - crosstalk) + np.exp(-Q) * crosstalk
    k = rng.binomial(n_photons, p, size=trials)
    frac = np.clip((k / n_photons - crosstalk) / (1 - 2 * crosstalk), 0.0, 1 - 1e-12)
    Q_hat = -np.log(1 - frac)
    return 4.0 * sigma * np.sqrt(Q_hat)


def estimate_direct(d, n_photons, trials, rng, sigma=1.0):
    """Each photon lands at x drawn from the two source mixture. With a known
    centroid, the second moment gives a simple unbiased estimate of d^2."""
    which = rng.random((trials, n_photons)) < 0.5
    x = rng.normal(0.0, sigma, (trials, n_photons)) + np.where(which, d / 2, -d / 2)
    d2_hat = 4.0 * (np.mean(x ** 2, axis=1) - sigma ** 2)
    return np.sqrt(np.clip(d2_hat, 0.0, None))
