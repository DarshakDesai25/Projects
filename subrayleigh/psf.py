"""Simulation of diffraction limited images of one or two incoherent point sources.

Units
  Separations are expressed in Rayleigh units (1.0 = the Rayleigh limit, the
  radius of the first dark ring of the Airy pattern).
"""
import numpy as np
from scipy.special import j1

J1_ZERO = 3.8317059702075125  # first zero of the Bessel function J1


def airy(r, rayleigh_px):
    """Normalized Airy intensity for radius r (in pixels)."""
    x = J1_ZERO * np.asarray(r, dtype=np.float64) / rayleigh_px
    out = np.ones_like(x)
    nz = x > 1e-9
    out[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
    return out


class Simulator:
    """Renders noisy camera images of point sources through a circular aperture.

    size         image width and height in pixels
    rayleigh_px  Rayleigh radius in pixels (sets how blurry the optics are)
    photons      expected total signal photons per image
    background   expected background photons per pixel
    supersample  sub pixel grid used to integrate light over each pixel
    """

    def __init__(self, size=32, rayleigh_px=5.0, photons=2000.0,
                 background=1.0, supersample=3):
        self.size = size
        self.rayleigh_px = rayleigh_px
        self.photons = photons
        self.background = background
        self.ss = supersample
        n = size * supersample
        coords = (np.arange(n) + 0.5) / supersample - size / 2.0
        self.gx, self.gy = np.meshgrid(coords, coords)

    def _psf(self, x, y):
        """Pixel integrated PSF for sources at (x, y). x and y have shape (n,)."""
        x = np.asarray(x, dtype=np.float64)[:, None, None]
        y = np.asarray(y, dtype=np.float64)[:, None, None]
        r = np.hypot(self.gx[None] - x, self.gy[None] - y)
        fine = airy(r, self.rayleigh_px)
        n, s, k = fine.shape[0], self.size, self.ss
        coarse = fine.reshape(n, s, k, s, k).mean(axis=(2, 4))
        return coarse / coarse.sum(axis=(1, 2), keepdims=True)

    def mean_image(self, cx, cy, sep, theta, flux=None, bg=None, ratio=0.5):
        """Expected photon counts. sep is in Rayleigh units. Arrays of shape (n,)."""
        flux = self.photons if flux is None else flux
        bg = self.background if bg is None else bg
        cx, cy, sep, theta = map(np.atleast_1d, (cx, cy, sep, theta))
        half = 0.5 * sep * self.rayleigh_px
        dx, dy = half * np.cos(theta), half * np.sin(theta)
        p1 = self._psf(cx + dx, cy + dy)
        p2 = self._psf(cx - dx, cy - dy)
        flux = np.atleast_1d(flux)[:, None, None]
        bg = np.atleast_1d(bg)[:, None, None]
        return flux * (ratio * p1 + (1 - ratio) * p2) + bg

    def sample(self, n, rng, sep_range=(0.05, 1.5), p_two=0.5,
               jitter_px=1.0, chunk=2000):
        """Draw n noisy images.

        Returns images (n, size, size) float32 photon counts, labels (1 = two
        sources), separations in Rayleigh units (0 for single sources) and
        orientations.
        """
        labels = (rng.random(n) < p_two).astype(np.int64)
        seps = np.where(labels == 1, rng.uniform(*sep_range, size=n), 0.0)
        theta = rng.uniform(0, np.pi, size=n)
        cx = rng.uniform(-jitter_px, jitter_px, size=n)
        cy = rng.uniform(-jitter_px, jitter_px, size=n)
        imgs = np.empty((n, self.size, self.size), dtype=np.float32)
        for i in range(0, n, chunk):
            j = slice(i, min(i + chunk, n))
            mu = self.mean_image(cx[j], cy[j], seps[j], theta[j])
            imgs[j] = rng.poisson(mu).astype(np.float32)
        return imgs, labels, seps.astype(np.float32), theta.astype(np.float32)

    def fisher_separation(self, sep, theta=0.0, eps=1e-3):
        """Classical Fisher information about the separation from one image,
        assuming Poisson pixel noise and known centroid, orientation and flux."""
        mu = self.mean_image(0.0, 0.0, sep, theta)[0]
        up = self.mean_image(0.0, 0.0, sep + eps, theta)[0]
        dn = self.mean_image(0.0, 0.0, max(sep - eps, 0.0), theta)[0]
        dmu = (up - dn) / (sep + eps - max(sep - eps, 0.0))
        return float(np.sum(dmu ** 2 / mu))
