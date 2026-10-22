"""Small CNN with two heads: one vs two sources (logit) and separation (Rayleigh units)."""
import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class ResolveNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.features = nn.Sequential(
            block(1, width), nn.MaxPool2d(2),            # 32 -> 16
            block(width, 2 * width), nn.MaxPool2d(2),    # 16 -> 8
            block(2 * width, 4 * width), nn.MaxPool2d(2),  # 8 -> 4
        )
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(4 * width * 16, 256),
                                   nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.cls_head = nn.Linear(256, 1)
        self.sep_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(self.features(x))
        return self.cls_head(h).squeeze(1), self.sep_head(h).squeeze(1)


def preprocess(imgs):
    """Photon counts (n, H, W) -> normalized tensor (n, 1, H, W).
    Subtracts a rough background and scales each image by its peak, so the
    network sees shape rather than brightness."""
    x = torch.as_tensor(imgs, dtype=torch.float32)
    bg = x[:, [0, -1], :].flatten(1).median(dim=1).values[:, None, None]
    x = x - bg
    x = x / x.flatten(1).amax(dim=1).clamp(min=1.0)[:, None, None]
    return x.unsqueeze(1)
