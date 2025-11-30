#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System2 (Sys2): Train an MLP on AE-latent vectors using Triplet Loss to get more
discriminative patch embeddings z' (per patch), and save z' per patient.

Input .npz format (your extractor):
  - feats: (N_patches, C)  [recommended]
  - paths: optional list of patch paths
  - patient_id: optional

We support legacy key:
  - z: (N_patches, C)

Expected folders:
  --latents-healthy  <dir_with_npz>  => label 0
  --latents-unhealthy <dir_with_npz> => label 1

Outputs under:
  /export/fhome/maed04/sys2_nana/
    runs/triplet_mlp_<timestamp>/
      mlp_best.pt
      mlp_last.pt
      train_log.csv
      zprime/healthy/*.npz
      zprime/unhealthy/*.npz
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_feats(npz_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
    """
    Returns:
      feats: (N,C) float32
      paths: (N,) object or None
      patient_id: str or None
    """
    data = np.load(npz_path, allow_pickle=True)
    if "feats" in data:
        feats = data["feats"]
    elif "z" in data:
        feats = data["z"]
    else:
        raise RuntimeError(f"[ERROR] {npz_path} has no 'feats' nor 'z'")

    feats = np.asarray(feats, dtype=np.float32)
    paths = data["paths"] if "paths" in data else None
    patient_id = str(data["patient_id"]) if "patient_id" in data else None
    return feats, paths, patient_id


# -------------------------
# Triplet dataset (in-memory)
# -------------------------
class TripletDatasetEmb(Dataset):
    """
    Builds (anchor, positive, negative) from embeddings X with labels y.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 1
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        self.cls_to_idx: Dict[int, np.ndarray] = {}
        for c in np.unique(y):
            self.cls_to_idx[int(c)] = np.where(y == c)[0]

        if len(self.cls_to_idx) < 2:
            raise RuntimeError("[ERROR] Need >=2 classes for triplet training.")

        # Precompute negatives pool per class
        self.classes = sorted(list(self.cls_to_idx.keys()))
        self.neg_pool: Dict[int, np.ndarray] = {}
        all_idx = np.arange(len(y))
        for c in self.classes:
            self.neg_pool[c] = all_idx[y != c]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        a = self.X[idx]
        ya = int(self.y[idx].item())

        pos_candidates = self.cls_to_idx[ya]
        if len(pos_candidates) < 2:
            # if a class has only 1 sample (rare), just pick itself (won't train well, but avoids crash)
            pos_idx = idx
        else:
            # choose a different positive
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = int(np.random.choice(pos_candidates))

        neg_candidates = self.neg_pool[ya]
        neg_idx = int(np.random.choice(neg_candidates))

        p = self.X[pos_idx]
        n = self.X[neg_idx]
        return a, p, n


# -------------------------
# MLP Embedder z -> z'
# -------------------------
class MLPEmbedder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z


# -------------------------
# Load all embeddings from two dirs
# -------------------------
def load_latents_two_dirs(latents_healthy: Path, latents_unhealthy: Path) -> Tuple[np.ndarray, np.ndarray]:
    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for label, d in [(0, latents_healthy), (1, latents_unhealthy)]:
        if not d.is_dir():
            raise RuntimeError(f"[ERROR] Missing dir: {d}")

        files = sorted(d.glob("*.npz"))
        if not files:
            raise RuntimeError(f"[ERROR] No .npz in: {d}")

        print(f"[INFO] Loading {len(files)} npz from {d} (label={label})")
        for f in tqdm(files, desc=f"load label={label}", leave=False):
            feats, _paths, _pid = load_npz_feats(f)
            all_feats.append(feats)
            all_labels.append(np.full((feats.shape[0],), label, dtype=np.int64))

    X = np.concatenate(all_feats, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)

    print(f"[INFO] X: {X.shape} | y: {y.shape} | classes: {np.unique(y).tolist()}")
    return X, y


# -------------------------
# Train
# -------------------------
def train_triplet_mlp(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    emb_dim: int = 256,
    margin: float = 1.0,
    batch_size: int = 512,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 123,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] DEVICE:", device)

    ds = TripletDatasetEmb(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=torch.cuda.is_available(), drop_last=True)

    input_dim = X.shape[1]
    model = MLPEmbedder(input_dim=input_dim, emb_dim=emb_dim, dropout=0.1).to(device)

    # Built-in TripletLoss
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    ckpt_best = out_dir / "mlp_best.pt"
    ckpt_last = out_dir / "mlp_last.pt"

    log_rows = []
    print(f"[INFO] Training Triplet MLP: input_dim={input_dim} emb_dim={emb_dim} margin={margin}")
    print(f"[INFO] epochs={epochs} bs={batch_size} lr={lr} wd={weight_decay}")

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        n = 0

        for a, p, nnv in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            nnv = nnv.to(device, non_blocking=True)

            za = model(a)
            zp = model(p)
            zn = model(nnv)

            loss = criterion(za, zp, zn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = a.size(0)
            run += loss.item() * bs
            n += bs

        ep_loss = run / max(1, n)
        print(f"[EP {ep:02d}] triplet_loss={ep_loss:.6f}")

        # save last
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "emb_dim": emb_dim,
            "margin": margin,
            "epoch": ep,
            "train_loss": ep_loss,
        }, ckpt_last)

        # save best
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "emb_dim": emb_dim,
                "margin": margin,
                "epoch": ep,
                "train_loss": ep_loss,
            }, ckpt_best)
            print(f"[SAVE] best -> {ckpt_best} (loss={best_loss:.6f})")

        log_rows.append({"epoch": ep, "triplet_loss": ep_loss})
        pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

    print("[DONE] Training finished. best_loss=", best_loss)
    return ckpt_best


# -------------------------
# Apply z -> z' and save per patient
# -------------------------
def apply_and_save_zprime(
    ckpt_path: Path,
    in_dir: Path,
    out_dir: Path,
    batch_size: int = 2048,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = int(ckpt["input_dim"])
    emb_dim = int(ckpt["emb_dim"])

    model = MLPEmbedder(input_dim=input_dim, emb_dim=emb_dim).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npz"))
    print(f"[INFO] Applying MLP to {len(files)} patients in {in_dir} -> {out_dir}")

    for f in tqdm(files, desc=f"zprime {in_dir.name}"):
        feats, paths, patient_id = load_npz_feats(f)
        if feats.shape[1] != input_dim:
            raise RuntimeError(f"[ERROR] {f}: feats dim {feats.shape[1]} != input_dim {input_dim}")

        x = torch.from_numpy(feats).float().to(device)

        outs = []
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                xb = x[i:i+batch_size]
                zb = model(xb).cpu().numpy().astype(np.float32)
                outs.append(zb)

        zprime = np.concatenate(outs, axis=0)
        mean_zprime = zprime.mean(axis=0)

        out_file = out_dir / f.name
        payload = {
            "zprime": zprime,
            "mean_zprime": mean_zprime,
        }
        if paths is not None:
            payload["paths"] = paths
        if patient_id is not None:
            payload["patient_id"] = patient_id

        np.savez_compressed(out_file, **payload)

    print("[DONE] Saved z' to:", out_dir)


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--latents-healthy", required=True, type=str)
    ap.add_argument("--latents-unhealthy", required=True, type=str)

    ap.add_argument("--root-out", default="/export/fhome/maed04/sys2_nana", type=str)

    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    root_out = Path(args.root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    run_dir = root_out / "runs" / f"triplet_mlp_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    pd.DataFrame([vars(args)]).to_csv(run_dir / "config.csv", index=False)
    print("[INFO] run_dir:", run_dir)

    lat_h = Path(args.latents_healthy)
    lat_u = Path(args.latents_unhealthy)

    X, y = load_latents_two_dirs(lat_h, lat_u)

    ckpt_best = train_triplet_mlp(
        X=X, y=y, out_dir=run_dir,
        emb_dim=args.emb_dim,
        margin=args.margin,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    # Save z' per patient, in two separate dirs
    zprime_root = run_dir / "zprime"
    apply_and_save_zprime(ckpt_best, lat_h, zprime_root / "healthy")
    apply_and_save_zprime(ckpt_best, lat_u, zprime_root / "unhealthy")


if __name__ == "__main__":
    main()

