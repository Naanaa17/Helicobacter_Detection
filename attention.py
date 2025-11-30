#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System2 - Patient Attention MIL over z' (triplet-MLP embeddings).

Input:
  zprime/
    healthy/*.npz
    unhealthy/*.npz

Each npz should contain:
  - "z" : (N_patches, emb_dim)  [preferred]
  optionally:
  - "paths": (N_patches,) list/array of patch paths

Trains:
  AttentionMILNet (AttentionUnits.py) to classify patient (bag) as:
    healthy=0 vs unhealthy=1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from AttentionUnits import Attention, NeuralNetwork


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_name(prefix: str = "attention_mil") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def load_npz_z(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if "z" in data:
        z = data["z"]
    elif "zprime" in data:
        z = data["zprime"]
    else:
        raise RuntimeError(f"[ERROR] {path} does not contain key 'z' (or 'zprime'). Keys={list(data.keys())}")

    paths = None
    if "paths" in data:
        paths = data["paths"]
    return z.astype(np.float32), paths


# ----------------------------
# Dataset
# ----------------------------
class PatientZPrimeDataset(Dataset):
    """
    Each item is one patient (one .npz):
      x: (N_patches, emb_dim) float32
      y: int64 (0 healthy, 1 unhealthy)
      pid: patient id (from filename stem)
      npz_path: str
    """
    def __init__(self, samples: List[Tuple[Path, int]], max_patches: Optional[int], seed: int):
        self.samples = samples
        self.max_patches = max_patches
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path, label = self.samples[idx]
        z, _paths = load_npz_z(npz_path)  # (N, D)

        if z.ndim != 2:
            raise RuntimeError(f"[ERROR] z must be 2D (N,D). Got {z.shape} in {npz_path}")

        # Optional sub-sample patches per patient
        if self.max_patches is not None and z.shape[0] > self.max_patches:
            sel = self.rng.choice(z.shape[0], self.max_patches, replace=False)
            z = z[sel]

        x = torch.from_numpy(z)  # (N,D)
        y = torch.tensor(label, dtype=torch.long)
        pid = npz_path.stem
        return x, y, pid, str(npz_path)


# ----------------------------
# Model
# ----------------------------
class AttentionMILNet(nn.Module):
    """
    Patient-level MIL:
      attention aggregates (N_patches, emb_dim) -> context Z
      classifier predicts patient class
    """
    def __init__(self, emb_dim: int, att_decomp_dim: int, att_branches: int, num_classes: int = 2):
        super().__init__()

        att_params = {
            "in_features": emb_dim,
            "decom_space": att_decomp_dim,
            "ATTENTION_BRANCHES": att_branches,
        }
        self.attention = Attention(att_params)

        clf_input_dim = emb_dim * att_branches
        clf_params = {"in_features": clf_input_dim, "out_features": num_classes}
        self.classifier = NeuralNetwork(clf_params)

    def forward(self, x: torch.Tensor):
        """
        x: (B=1, N, D)
        returns:
          logits: (1,2)
          A: attention weights (shape depends on your AttentionUnits)
        """
        Z, A = self.attention(x)      # Z: (branches, D)
        Z_flat = Z.view(1, -1)        # (1, branches*D)
        logits = self.classifier(Z_flat)
        return logits, A


# ----------------------------
# Index samples
# ----------------------------
def build_samples(zprime_dirs: Dict[int, Path]) -> Tuple[List[Tuple[Path, int]], int]:
    samples: List[Tuple[Path, int]] = []
    emb_dim: Optional[int] = None

    for label, d in zprime_dirs.items():
        if not d.is_dir():
            raise RuntimeError(f"[ERROR] zprime dir not found: {d}")

        files = sorted(d.glob("*.npz"))
        if not files:
            raise RuntimeError(f"[ERROR] no .npz found in {d}")

        print(f"[INFO] Found {len(files)} patients in {d} (label={label})")

        for f in files:
            if emb_dim is None:
                z, _ = load_npz_z(f)
                emb_dim = int(z.shape[1])
            samples.append((f, label))

    if emb_dim is None:
        raise RuntimeError("[ERROR] Could not infer emb_dim (no files read).")

    # Shuffle once (seeded outside)
    return samples, emb_dim


# ----------------------------
# Train
# ----------------------------
def train_attention(
    zprime_healthy: Path,
    zprime_unhealthy: Path,
    out_dir: Path,
    epochs: int,
    lr: float,
    att_decomp_dim: int,
    att_branches: int,
    max_patches: Optional[int],
    seed: int,
    val_frac: float,
    patience: int,
    min_delta: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] DEVICE:", device)

    set_seed(seed)

    zprime_dirs = {0: zprime_healthy, 1: zprime_unhealthy}
    samples, emb_dim = build_samples(zprime_dirs)

    rng = np.random.RandomState(seed)
    rng.shuffle(samples)

    n_total = len(samples)
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"[INFO] Patients total={n_total} train={len(train_samples)} val={len(val_samples)}")
    print(f"[INFO] emb_dim={emb_dim} att_decomp_dim={att_decomp_dim} branches={att_branches} max_patches={max_patches}")

    train_ds = PatientZPrimeDataset(train_samples, max_patches=max_patches, seed=seed + 1)
    val_ds = PatientZPrimeDataset(val_samples, max_patches=max_patches, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = AttentionMILNet(emb_dim=emb_dim, att_decomp_dim=att_decomp_dim, att_branches=att_branches, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    cfg = {
        "zprime_healthy": str(zprime_healthy),
        "zprime_unhealthy": str(zprime_unhealthy),
        "epochs": epochs,
        "lr": lr,
        "att_decomp_dim": att_decomp_dim,
        "att_branches": att_branches,
        "max_patches": max_patches,
        "seed": seed,
        "val_frac": val_frac,
        "patience": patience,
        "min_delta": min_delta,
        "emb_dim": emb_dim,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    no_improve = 0
    log_rows = []

    ckpt_best = out_dir / "checkpoints" / "attention_best.pt"
    ckpt_last = out_dir / "checkpoints" / "attention_last.pt"

    print("[INFO] Starting AttentionMIL training...")

    for ep in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for xb, yb, pid, _npz in tqdm(train_loader, desc=f"[TRAIN] {ep}/{epochs}"):
            xb = xb.to(device)  # (1, N, D)
            yb = yb.to(device)  # (1,)

            logits, _att = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_loss_sum += float(loss.item())
            preds = logits.argmax(dim=1)
            tr_correct += int((preds == yb).sum().item())
            tr_total += 1

        tr_loss = tr_loss_sum / max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # ---- VAL ----
        model.eval()
        va_loss_sum = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for xb, yb, pid, _npz in tqdm(val_loader, desc=f"[VAL]   {ep}/{epochs}"):
                xb = xb.to(device)
                yb = yb.to(device)

                logits, _att = model(xb)
                loss = criterion(logits, yb)

                va_loss_sum += float(loss.item())
                preds = logits.argmax(dim=1)
                va_correct += int((preds == yb).sum().item())
                va_total += 1

        va_loss = va_loss_sum / max(va_total, 1)
        va_acc = va_correct / max(va_total, 1)

        print(f"[EP {ep:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        # logs
        log_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })
        pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

        # save last
        torch.save({
            "state_dict": model.state_dict(),
            "epoch": ep,
            **cfg,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
        }, ckpt_last)

        # save best (early stopping)
        if va_loss < best_val - min_delta:
            best_val = va_loss
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep,
                "best_val": best_val,
                **cfg,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_acc": tr_acc,
                "val_acc": va_acc,
            }, ckpt_best)
            print(f"[SAVE] best -> {ckpt_best} (best_val={best_val:.4f})")
        else:
            no_improve += 1
            print(f"[INFO] No improve ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"[EARLY STOP] patience reached ({patience}).")
            break

    print("[DONE] Finished.")
    print("[OUT]", out_dir)
    print("[BEST]", ckpt_best)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--zprime-healthy", default="/export/fhome/maed04/sys2_nana/runs/triplet_mlp_20251127_003153/zprime/healthy")
    ap.add_argument("--zprime-unhealthy", default="/export/fhome/maed04/sys2_nana/runs/triplet_mlp_20251127_003153/zprime/unhealthy")

    ap.add_argument("--out-root", default="/export/fhome/maed04/sys2_nana/runs")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--att-decomp-dim", type=int, default=128)
    ap.add_argument("--att-branches", type=int, default=1)
    ap.add_argument("--max-patches", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)

    args = ap.parse_args()

    zph = Path(args.zprime_healthy)
    zpu = Path(args.zprime_unhealthy)

    out_root = Path(args.out_root)
    run_name = args.run_name.strip() or now_run_name("attention_mil")
    out_dir = out_root / run_name

    max_patches = None if args.max_patches <= 0 else int(args.max_patches)

    train_attention(
        zprime_healthy=zph,
        zprime_unhealthy=zpu,
        out_dir=out_dir,
        epochs=args.epochs,
        lr=args.lr,
        att_decomp_dim=args.att_decomp_dim,
        att_branches=args.att_branches,
        max_patches=max_patches,
        seed=args.seed,
        val_frac=args.val_frac,
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
