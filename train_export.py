"""Train a small MNIST CNN in PyTorch and export everything the C++ engine needs.

Outputs (in ./model):
  model.txt         layer list, one op per line (the "graph")
  weights.bin       raw float32 tensors, little-endian, with a tiny header per tensor
  test_images.bin   10000 x 1 x 28 x 28 float32, already normalised
  test_labels.bin   10000 uint8
  ref_logits.bin    10000 x 10 float32, PyTorch's own outputs (the ground truth to match)
  pytorch.json      PyTorch accuracy and latency, for the comparison table

Run:  ../.venv/bin/python train_export.py
"""
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

OUT = Path(__file__).parent / "model"
DATA = Path(__file__).parent / "data"
MEAN, STD = 0.1307, 0.3081


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def to_tensor(ds):
    x = ds.data.float().div(255.0).sub(MEAN).div(STD).unsqueeze(1)
    return x, ds.targets


def write_tensor(f, name, t):
    a = t.detach().cpu().numpy().astype("<f4")
    f.write(struct.pack("<I", len(name)))
    f.write(name.encode())
    f.write(struct.pack("<I", a.ndim))
    f.write(struct.pack(f"<{a.ndim}I", *a.shape))
    f.write(a.tobytes())


def main():
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    xtr, ytr = to_tensor(datasets.MNIST(DATA, train=True, download=True))
    xte, yte = to_tensor(datasets.MNIST(DATA, train=False, download=True))

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  test acc {acc * 100:.2f}%")

    model.eval()
    with torch.no_grad():
        logits = model(xte)
    acc = (logits.argmax(1) == yte).float().mean().item()

    # --- export -----------------------------------------------------------
    with open(OUT / "weights.bin", "wb") as f:
        params = dict(model.named_parameters())
        f.write(b"TINY")
        f.write(struct.pack("<I", len(params)))
        for name, t in params.items():
            write_tensor(f, name, t)
    (OUT / "model.txt").write_text(
        "input 1 28 28\n"
        "conv conv1 relu\n"
        "maxpool\n"
        "conv conv2 relu\n"
        "maxpool\n"
        "flatten\n"
        "linear fc1 relu\n"
        "linear fc2 none\n"
    )
    xte.numpy().astype("<f4").tofile(OUT / "test_images.bin")
    yte.numpy().astype(np.uint8).tofile(OUT / "test_labels.bin")
    logits.numpy().astype("<f4").tofile(OUT / "ref_logits.bin")

    # --- PyTorch latency: batch 1, one thread, same conditions as the engine --
    torch.set_num_threads(1)
    lat = []
    with torch.no_grad():
        for i in range(200):  # warm-up
            model(xte[i : i + 1])
        for i in range(2000):
            t0 = time.perf_counter()
            model(xte[i : i + 1])
            lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model(xte)
        batch_1t = time.perf_counter() - t0
    lat = np.array(lat) * 1e6
    stats = {
        "torch_version": torch.__version__,
        "accuracy": acc,
        "latency_us_median": float(np.median(lat)),
        "latency_us_mean": float(lat.mean()),
        "latency_us_p99": float(np.percentile(lat, 99)),
        "batch10000_1thread_images_per_s": len(xte) / batch_1t,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (OUT / "pytorch.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
