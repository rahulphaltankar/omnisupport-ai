"""Return-image classification: data, models and training for the basic CNN and the transfer-learning model."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from omnisupport.config import IMAGE_LABELS, ROOT, SEED

CLASSES = ["damaged", "normal", "packaging_damage", "wrong_item"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG_SIZE = 96


def load_labels(path: Path = IMAGE_LABELS) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["class_idx"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    if df["class_idx"].isna().any():
        raise ValueError(f"unexpected labels: {sorted(df['label'].unique())}")
    return df


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified; identical for every model so the comparison is controlled."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["class_idx"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["class_idx"], random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def load_tensor(paths, root: Path = ROOT, size: int = IMG_SIZE) -> torch.Tensor:
    arr = []
    for p in paths:
        im = Image.open(root / p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        arr.append(np.asarray(im, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Batch augmentation: random affine (rotate +/-12deg, shift 8%, scale 0.9-1.1), horizontal flip, brightness/contrast jitter, light noise."""
    b = x.shape[0]
    ang = (torch.rand(b, generator=gen) * 2 - 1) * (12 * np.pi / 180)
    sc = 1 + (torch.rand(b, generator=gen) * 2 - 1) * 0.1
    tx, ty = [(torch.rand(b, generator=gen) * 2 - 1) * 0.16 for _ in range(2)]
    flip = (torch.rand(b, generator=gen) < 0.5).float() * 2 - 1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    br = 1 + (torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1) * 0.2
    ct = 1 + (torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1) * 0.2
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m) * br
    x = x + torch.randn(x.shape, generator=gen) * 0.01
    return x.clamp(0, 1)


def perturb(x: torch.Tensor, kind: str, level: float, seed: int = 0) -> torch.Tensor:
    """Test-time corruptions used for the robustness / failure analysis."""
    g = torch.Generator().manual_seed(seed)
    if kind == "noise":
        return (x + torch.randn(x.shape, generator=g) * level).clamp(0, 1)
    if kind == "blur":
        k = int(level) * 2 + 1
        w = torch.ones(3, 1, k, k) / (k * k)
        return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w, groups=3)
    if kind == "rotate":
        ang = torch.full((x.shape[0],), level * np.pi / 180)
        theta = torch.stack([torch.stack([torch.cos(ang), -torch.sin(ang), torch.zeros_like(ang)], 1), torch.stack([torch.sin(ang), torch.cos(ang), torch.zeros_like(ang)], 1)], 1)
        return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    if kind == "darken":
        return (x * level).clamp(0, 1)
    if kind == "occlude":
        out = x.clone(); s = int(x.shape[-1] * level); out[:, :, :s, :s] = 0.5
        return out
    raise ValueError(kind)


class BasicCNN(nn.Module):
    """3 conv blocks (16-32-64) + batch-norm + global average pooling + dropout. ~25k parameters."""

    def __init__(self, n_classes: int = 4, p_drop: float = 0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(n_classes: int = 4, pretrained: bool = True):
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_classes))
    return m


class Normalised(nn.Module):
    """Wraps a model so it accepts [0,1] images (ImageNet normalisation + optional resize happen inside)."""

    def __init__(self, net: nn.Module, imagenet_norm: bool = False, resize: int | None = None):
        super().__init__()
        self.net, self.imagenet_norm, self.resize = net, imagenet_norm, resize
        self.register_buffer("mean", MEAN.clone()); self.register_buffer("std", STD.clone())

    def forward(self, x):
        if self.resize:
            x = F.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        return self.net(x)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, bs: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); probs = []
    for i in range(0, len(X), bs):
        probs.append(torch.softmax(model(X[i:i + bs]), 1))
    p = torch.cat(probs).numpy()
    return p.argmax(1), p


def train_model(model: nn.Module, Xtr, ytr, Xva, yva, *, epochs: int, lr: float, wd: float = 1e-4, bs: int = 32, seed: int = SEED, augmenting: bool = True,
                param_groups=None, log_every: int = 1, tag: str = "") -> dict:
    """Generic loop: AdamW + cosine schedule, best-val-macro-F1 checkpointing, per-epoch history."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=epochs * int(np.ceil(len(Xtr) / bs)))
    best, best_state, hist = -1.0, None, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=gen); tl = 0.0; correct = 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], ytr[idx]
            if augmenting:
                xb = augment(xb, gen)
            opt.zero_grad(); out = model(xb); loss = F.cross_entropy(out, yb); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(idx); correct += (out.argmax(1) == yb).sum().item()
        pv, prob = predict(model, Xva); vloss = F.nll_loss(torch.log(torch.from_numpy(prob).clamp_min(1e-8)), yva).item()
        vf1 = f1_score(yva.numpy(), pv, average="macro"); vacc = float((pv == yva.numpy()).mean())
        hist.append({"epoch": ep, "train_loss": tl / len(Xtr), "train_acc": correct / len(Xtr), "val_loss": vloss, "val_acc": vacc, "val_macro_f1": vf1})
        if vf1 > best:
            best, best_state = vf1, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % log_every == 0:
            print(f"[{tag}] ep{ep:02d} loss {tl / len(Xtr):.3f} acc {correct / len(Xtr):.3f} | val loss {vloss:.3f} acc {vacc:.3f} f1 {vf1:.3f}", flush=True)
    model.load_state_dict(best_state)
    return {"history": pd.DataFrame(hist), "best_val_macro_f1": best, "train_seconds": time.perf_counter() - t0}


def evaluate(model, X, y: np.ndarray) -> dict:
    t0 = time.perf_counter(); pred, prob = predict(model, X); dt = time.perf_counter() - t0
    rep = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro")), "per_class": {c: rep[c] for c in CLASSES},
            "confusion": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(), "ms_per_image": dt / len(X) * 1000, "pred": pred, "prob": prob}


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
