"""Train the CNN on simulated images and evaluate it against the classical limit.

Example
  python train.py --epochs 15 --n-train 60000
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from psf import Simulator
from model import ResolveNet, preprocess


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_split(sim, n, seed):
    imgs, labels, seps, _ = sim.sample(n, np.random.default_rng(seed))
    return (preprocess(imgs), torch.as_tensor(labels, dtype=torch.float32),
            torch.as_tensor(seps, dtype=torch.float32))


@torch.no_grad()
def predict(model, x, device, bs=1024):
    model.eval()
    probs, seps = [], []
    for i in range(0, len(x), bs):
        logit, sep = model(x[i:i + bs].to(device))
        probs.append(torch.sigmoid(logit).cpu())
        seps.append(sep.cpu())
    return torch.cat(probs).numpy(), torch.cat(seps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sep-weight", type=float, default=5.0)
    ap.add_argument("--photons", type=float, default=1000.0)
    ap.add_argument("--background", type=float, default=2.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    sim = Simulator(photons=a.photons, background=a.background)
    t = time.time()
    xtr, ytr, dtr = make_split(sim, a.n_train, seed=1)
    xva, yva, dva = make_split(sim, a.n_val, seed=2)
    xte, yte, dte = make_split(sim, a.n_test, seed=3)
    print(f"simulated data in {time.time() - t:.0f}s")

    model = ResolveNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((a.n_train + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)

    best, history = 1e9, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(a.n_train)
        tot = 0.0
        for i in range(0, a.n_train, a.batch):
            idx = perm[i:i + a.batch]
            x, y, d = xtr[idx].to(device), ytr[idx].to(device), dtr[idx].to(device)
            # random flips and 90 degree rotations: the physics is symmetric
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
            logit, sep = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            mask = y > 0.5
            if mask.any():
                loss = loss + a.sep_weight * F.smooth_l1_loss(sep[mask], d[mask], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        p, s = predict(model, xva, device)
        yv, dv = yva.numpy(), dva.numpy()
        acc = float(np.mean((p > 0.5) == (yv > 0.5)))
        mae = float(np.mean(np.abs(s[yv > 0.5] - dv[yv > 0.5])))
        history.append({"epoch": ep + 1, "train_loss": tot / a.n_train, "val_acc": acc, "val_sep_mae": mae})
        print(history[-1])
        score = (1 - acc) + mae
        if score < best:
            best = score
            torch.save(model.state_dict(), os.path.join(a.out, "resolvenet.pt"))

    model.load_state_dict(torch.load(os.path.join(a.out, "resolvenet.pt"), map_location=device))
    p, s = predict(model, xte, device)
    np.savez(os.path.join(a.out, "cnn.npz"), prob=p, sep_hat=s, labels=yte.numpy(),
             seps=dte.numpy(), photons=a.photons, background=a.background)
    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("test accuracy:", float(np.mean((p > 0.5) == (yte.numpy() > 0.5))))
    print("now run: python plots.py")


if __name__ == "__main__":
    main()
