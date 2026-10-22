"""Quick smoke test. Run this first to make sure everything is wired up."""
import numpy as np
import torch
from psf import Simulator
from model import ResolveNet, preprocess

sim = Simulator()
imgs, labels, seps, _ = sim.sample(8, np.random.default_rng(0))
x = preprocess(imgs)
logit, sep = ResolveNet()(x)
assert x.shape == (8, 1, 32, 32)
assert logit.shape == (8,) and sep.shape == (8,)
print("all good:", x.shape, logit.shape, sep.shape)
