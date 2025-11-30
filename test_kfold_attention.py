#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Fold evaluation for AttentionMILNet using precomputed z' per patient (.npz).

Inputs:
  --zprime-healthy   <dir with *.npz>
  --zprime-unhealthy <dir with *.npz>
  --checkpoint       attention_best.pt

Outputs:
  out_dir/
    kfold_metrics.csv
    all_patient_scores.csv
    roc_kfold.png
    confusion_fold_XX.png
    confusion_global.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

from AttentionUnits import Attention, NeuralNetwork


# ----------------------------
# Model (same as training)
# ----------------------------
class AttentionMILNet(nn.Module):
    def __init__(self, emb_dim: int, att_decomp_dim: int, att_branches: int, num_classes: int = 2):
        super().__init__()
        self.attention = Attention({
            "in_features": emb_dim,
            "decom_space": att_decomp_dim,
            "ATTENTION_BRANCHES": att_branches,
        })
        self.classifier = NeuralNetwork({
            "in_features": emb_dim * att_branches,
            "out_features": num_classes,
        })

    def forward(self, x: torch.Tensor):
        # x: (1, N, D)
        Z, A = self.attention(x)     # Z: (branches, D)
        Z_flat = Z.view(1, -1)       # (1, branches*D)
        logits = self.classifier(Z_flat)
        return logits, A


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_z(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "z" in data:
        z = data["z"]
    elif "zprime" in data:
        z = data["zprime"]
    else:
        raise RuntimeError(f"[ERROR] {path} missing 'z' (or 'zprime'). Keys={list(data.keys())}")
    z = z.astype(np.float32)
    if z.ndim != 2:
        raise RuntimeError(f"[ERROR] z must be 2D (N,D). Got {z.shape} in {path}")
    return z


def plot_confusion(cm: np.ndarray, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)  # no explicit colors

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy(0)", "unhealthy(1)"])
    ax.set_yticklabels(["healthy(0)", "unhealthy(1)"])

    # numbers
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Dataset
# ----------------------------
class PatientBagsDataset(Dataset):
    """
    Each item: patient bag (N_patches, emb_dim), label, patient_id, npz_path
    """
    def __init__(self, zprime_healthy: Path, zprime_unhealthy: Path, max_patches: Optional[int], seed: int):
        self.samples: List[Tuple[Path, int]] = []
        self.max_patches = max_patches
        self.rng = np.random.RandomState(seed)

        for lab, d in [(0, zprime_healthy), (1, zprime_unhealthy)]:
            if not d.is_dir():
                raise RuntimeError(f"[ERROR] Not a directory: {d}")
            files = sorted(d.glob("*.npz"))
            if not files:
                raise RuntimeError(f"[ERROR] No .npz found in {d}")
            for f in files:
                self.samples.append((f, lab))

        # shuffle once to avoid ordering effects
        self.rng.shuffle(self.samples)

        # infer emb_dim
        z0 = load_npz_z(self.samples[0][0])
        self.emb_dim = int(z0.shape[1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path, label = self.samples[idx]
        z = load_npz_z(npz_path)  # (N,D)

        if self.max_patches is not None and z.shape[0] > self.max_patches:
            sel = self.rng.choice(z.shape[0], self.max_patches, replace=False)
            z = z[sel]

        x = torch.from_numpy(z)  # (N,D)
        y = torch.tensor(label, dtype=torch.long)
        pid = npz_path.stem
        return x, y, pid, str(npz_path)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zprime-healthy", required=True)
    ap.add_argument("--zprime-unhealthy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-patches", type=int, default=0, help="0 = no limit")
    ap.add_argument("--threshold", type=float, default=0.5, help="prob threshold for confusion matrix & metrics")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] DEVICE:", device)

    zph = Path(args.zprime_healthy)
    zpu = Path(args.zprime_unhealthy)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_patches = None if args.max_patches <= 0 else int(args.max_patches)

    dataset = PatientBagsDataset(zph, zpu, max_patches=max_patches, seed=args.seed + 1)
    labels = [lab for _p, lab in dataset.samples]

    # robust k (avoid errors if few positives)
    n_pos = int(np.sum(np.array(labels) == 1))
    n_neg = int(np.sum(np.array(labels) == 0))
    k = min(args.k, len(labels), n_pos, n_neg)
    k = max(k, 2)
    print(f"[INFO] Patients={len(labels)} (pos={n_pos}, neg={n_neg}) -> using k={k}")

    # load model
    ckpt = torch.load(ckpt_path, map_location=device)
    emb_dim = int(ckpt.get("emb_dim", dataset.emb_dim))
    att_decomp_dim = int(ckpt.get("att_decomp_dim", ckpt.get("att_decomp_dim", 128)))
    att_branches = int(ckpt.get("att_branches", 1))
    num_classes = int(ckpt.get("num_classes", 2))

    model = AttentionMILNet(
        emb_dim=emb_dim,
        att_decomp_dim=att_decomp_dim,
        att_branches=att_branches,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"[INFO] Loaded model: emb_dim={emb_dim} att_decomp_dim={att_decomp_dim} branches={att_branches}")

    # KFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    fold_rows = []

    all_true = []
    all_prob = []
    all_pred = []
    all_pid = []
    all_npz = []
    all_fold = []

    # ROC plot skeleton
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)

    for fold, (_tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        test_subset = Subset(dataset, te)
        loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))

        y_true = []
        y_prob = []
        y_pid = []
        y_npz = []

        with torch.no_grad():
            for xb, yb, pid, npz_path in tqdm(loader, desc=f"[FOLD {fold:02d}]"):
                xb = xb.to(device)  # (1,N,D)
                logits, _att = model(xb)
                prob_unhealthy = torch.softmax(logits, dim=1)[:, 1].item()

                y_true.append(int(yb.item()))
                y_prob.append(float(prob_unhealthy))
                y_pid.append(pid[0])
                y_npz.append(npz_path[0])

        y_true_np = np.array(y_true, dtype=int)
        y_prob_np = np.array(y_prob, dtype=float)

        # ROC/AUC
        if len(np.unique(y_true_np)) == 2:
            fpr, tpr, _ = roc_curve(y_true_np, y_prob_np)
            fold_auc = auc(fpr, tpr)
        else:
            # edge-case fold with a single class (rare with small data)
            fpr, tpr, fold_auc = np.array([0, 1]), np.array([0, 1]), float("nan")

        aucs.append(float(fold_auc))
        ax.plot(fpr, tpr, lw=1.2, alpha=0.85, label=f"Fold {fold} (AUC={fold_auc:.3f})")

        # interpolate for mean ROC
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

        # confusion & metrics (fixed threshold)
        thr = float(args.threshold)
        y_pred_np = (y_prob_np >= thr).astype(int)

        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        prec = precision_score(y_true_np, y_pred_np, zero_division=0)
        rec = recall_score(y_true_np, y_pred_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

        fold_rows.append({
            "fold": fold,
            "n_test": int(len(y_true_np)),
            "auc": float(fold_auc),
            "precision@thr": float(prec),
            "recall@thr": float(rec),
            "f1@thr": float(f1),
            "threshold": thr,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

        plot_confusion(cm, out_dir / f"confusion_fold_{fold:02d}.png", title=f"Confusion - Fold {fold} (thr={thr})")

        # accumulate global
        all_true.extend(y_true_np.tolist())
        all_prob.extend(y_prob_np.tolist())
        all_pred.extend(y_pred_np.tolist())
        all_pid.extend(y_pid)
        all_npz.extend(y_npz)
        all_fold.extend([fold] * len(y_true_np))

    # Mean ROC ± std
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    tprs_arr = np.stack(tprs, axis=0)
    mean_tpr = tprs_arr.mean(axis=0)
    std_tpr = tprs_arr.std(axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, lw=2.2, label=f"Mean ROC (AUC={mean_auc:.3f})")
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), alpha=0.2, label="±1 std")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AttentionMIL ROC - KFold")
    ax.grid(alpha=0.35)
    ax.legend(loc="lower right", fontsize="small")

    fig.tight_layout()
    fig.savefig(out_dir / "roc_kfold.png", dpi=200)
    plt.close(fig)

    # Global confusion & metrics
    all_true_np = np.array(all_true, dtype=int)
    all_prob_np = np.array(all_prob, dtype=float)
    all_pred_np = np.array(all_pred, dtype=int)

    cm_g = confusion_matrix(all_true_np, all_pred_np, labels=[0, 1])
    plot_confusion(cm_g, out_dir / "confusion_global.png", title=f"Global Confusion (thr={args.threshold})")

    prec_g = precision_score(all_true_np, all_pred_np, zero_division=0)
    rec_g = recall_score(all_true_np, all_pred_np, zero_division=0)
    f1_g = f1_score(all_true_np, all_pred_np, zero_division=0)

    # Save CSVs
    df_scores = pd.DataFrame({
        "fold": all_fold,
        "patient_id": all_pid,
        "npz_path": all_npz,
        "y_true": all_true_np,
        "prob_unhealthy": all_prob_np,
        "y_pred": all_pred_np,
    })
    df_scores.to_csv(out_dir / "all_patient_scores.csv", index=False)

    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.loc[len(df_metrics)] = {
        "fold": "GLOBAL",
        "n_test": int(len(all_true_np)),
        "auc": float("nan"),
        "precision@thr": float(prec_g),
        "recall@thr": float(rec_g),
        "f1@thr": float(f1_g),
        "threshold": float(args.threshold),
        "tn": int(cm_g[0, 0]),
        "fp": int(cm_g[0, 1]),
        "fn": int(cm_g[1, 0]),
        "tp": int(cm_g[1, 1]),
    }
    df_metrics.to_csv(out_dir / "kfold_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print(f"[OUT] {out_dir}")
    print(f"[KFold] mean_auc={np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}  (mean ROC AUC={mean_auc:.4f})")
    print(f"[Global@thr={args.threshold}] Precision={prec_g:.4f} Recall={rec_g:.4f} F1={f1_g:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
