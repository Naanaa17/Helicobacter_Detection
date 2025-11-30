#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System1 - Threshold tuning (KFold by patient) using Annotated + Excel Presence labels,
for the NEW AE trained with RGB MSE + HSV aux.

- Computes patch-level scores on matched Annotated patches (disk ∩ excel).
- KFold split is by patient (stratified by patient label).
- Learns threshold on TRAIN fold (Youden or best-F1).
- Evaluates on TEST fold at PATCH and PATIENT level (patient aggregation).
- Saves:
    out_dir/
      config.json
      all_scores.csv
      kfold_summary.csv
      ROC_PATCH_with_table.png
      ROC_PATIENT_with_table.png
      folds/fold_k/...

Notes:
- Model checkpoint expected to have "state_dict" + "config" + "img_size" (like your ae_best.pt).
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

from Models.AEmodels import AutoEncoderCNN


# ============================================================
# Auto-detect helpers (avoid "wrong path" pain)
# ============================================================
def first_existing_path(candidates: List[Path], kind: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    # if none exists, return first (so argparse shows default), but will fail later with good msg
    return candidates[0] if candidates else Path(f"__MISSING_{kind}__")


# ============================================================
# DEFAULT PATHS (adapted to maed04 + your new ckpt)
# ============================================================
DEFAULT_ANNOTATED_ROOT = first_existing_path([
    Path("/export/fhome/maed04/Cross_validation/Annotated"),
    Path("/export/fhome/maed04/HelicoDataSet/CrossValidation/Annotated"),
    Path("/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated"),
], "ANNOTATED_ROOT")

DEFAULT_EXCEL_PATH = first_existing_path([
    Path("/export/fhome/maed04/Cross_validation/HP_WSI-CoordAllAnnotatedPatches.xlsx"),
    Path("/export/fhome/maed04/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"),
    Path("/export/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"),
], "EXCEL_PATH")

DEFAULT_CKPT_PATH = first_existing_path([
    Path("/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv/ae_best.pt"),
    Path("./ae_best.pt"),
], "CKPT_PATH")

DEFAULT_OUT_DIR = Path("/export/fhome/maed04/prova_sys1/results_system1/threshold_kfold_hue_paperhsv_v5_pretty")


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

FOLDER_RE = re.compile(r"^(?P<pat>.+?)[_-]{1,2}(?P<sec>\d+)$")  # B22-47_0 o B22-94__1

def parse_folder(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    m = FOLDER_RE.match(folder_name)
    if not m:
        return None, None
    return m.group("pat").strip(), str(int(m.group("sec")))

def norm_section(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def norm_window_id(w) -> str:
    """
    Normaliza Window_ID al estilo disco:
      521 / 521.0 -> 00521
      902_Aug1 / 902.0_Aug1 -> 00902_Aug1
      01653_Aug1 -> 01653_Aug1
    """
    s = str(w).strip()

    m = re.match(r"^(\d+)(?:\.0)?_(Aug\d+)$", s)
    if m:
        return f"{int(m.group(1)):05d}_{m.group(2)}"

    m2 = re.match(r"^(\d+)_Aug\d+$", s)
    if m2:
        num = int(m2.group(1))
        aug = s.split("_", 1)[1]
        return f"{num:05d}_{aug}"

    try:
        return f"{int(float(s)):05d}"
    except Exception:
        return s


# ============================================================
# AE CONFIGS (same as training script)
# ============================================================
def build_ae_configs(config: str, input_channels: int):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}

    net_paramsEnc["drop_rate"] = 0.0
    net_paramsDec["drop_rate"] = 0.0

    if config == "1":
        net_paramsEnc["block_configs"] = [[32, 32], [64, 64]]
        net_paramsEnc["stride"] = [[1, 2], [1, 2]]

        net_paramsDec["block_configs"] = [[64, 32], [32, input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "2":
        net_paramsEnc["block_configs"] = [[32], [64], [128], [256]]
        net_paramsEnc["stride"] = [[2], [2], [2], [2]]

        net_paramsDec["block_configs"] = [[128], [64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "3":
        net_paramsEnc["block_configs"] = [[32], [64], [64]]
        net_paramsEnc["stride"] = [[1], [2], [2]]

        net_paramsDec["block_configs"] = [[64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]
    else:
        raise ValueError(f"Unknown config: {config}")

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format unexpected: expected dict with state_dict/config/img_size.")

    if "state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'state_dict'. Keys={list(ckpt.keys())}")

    config = str(ckpt.get("config", "3"))
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = build_ae_configs(config, 3)

    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


# ============================================================
# Index disk + Read Excel + Merge
# ============================================================
def index_annotated_disk(root: Path) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Annotated root not found: {root}")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    rows = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        pat, sec = parse_folder(folder.name)
        if pat is None:
            continue
        for p in folder.iterdir():
            if not (p.is_file() and p.suffix.lower() in exts):
                continue
            rows.append({
                "img_path": str(p),
                "pat_id": pat,
                "section_id": sec,
                "window_id_norm": norm_window_id(p.stem),
            })
    return pd.DataFrame(rows)

def load_excel_labels(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"Pat_ID", "Section_ID", "Window_ID", "Presence"}
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"Excel missing columns: {miss}. Columns={list(df.columns)}")

    df = df[df["Presence"].isin([-1, 1])].copy()    # descartamos Presence=0
    df["label"] = (df["Presence"] == 1).astype(int)

    df["pat_id"] = df["Pat_ID"].astype(str).str.strip()
    df["section_id"] = df["Section_ID"].apply(norm_section)
    df["window_id_norm"] = df["Window_ID"].apply(norm_window_id)

    return df[["pat_id", "section_id", "window_id_norm", "label"]]

def build_merged_df(annotated_root: Path, excel_path: Path) -> pd.DataFrame:
    print("[STEP] Indexing disk Annotated...")
    df_disk = index_annotated_disk(annotated_root)
    if len(df_disk) == 0:
        raise RuntimeError(f"No images indexed under {annotated_root}")

    print(f"[INFO] Disk images indexed: {len(df_disk)} | patients={df_disk['pat_id'].nunique()}")

    print("[STEP] Loading Excel labels...")
    df_x = load_excel_labels(excel_path)
    print(f"[INFO] Excel label rows (Presence -1/1): {len(df_x)}")

    sd = set(df_disk["window_id_norm"].unique())
    sx = set(df_x["window_id_norm"].unique())
    print(f"[DEBUG] unique disk window_id_norm: {len(sd)}")
    print(f"[DEBUG] unique excel window_id_norm: {len(sx)}")
    print(f"[DEBUG] intersection window_id_norm: {len(sd & sx)}")

    df = df_disk.merge(df_x, on=["pat_id", "section_id", "window_id_norm"], how="inner")
    pos = int(df["label"].sum()) if len(df) else 0
    neg = int((df["label"] == 0).sum()) if len(df) else 0
    print(f"[INFO] Matched (disk ∩ excel): {len(df)} | Pos={pos} Neg={neg} patients={df['pat_id'].nunique() if len(df) else 0}")

    if len(df) == 0 or df["label"].nunique() < 2:
        raise RuntimeError("Need matched samples with BOTH classes to compute ROC/AUC.")

    return df


# ============================================================
# Dataset
# ============================================================
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

class AnnotatedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 256):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img = Image.open(r["img_path"]).convert("RGB")
        if self.img_size:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = pil_to_tensor(img)
        y = int(r["label"])
        pat = str(r["pat_id"])
        return x, y, pat, r["img_path"]


# ============================================================
# HSV + circular Hue distance
# ============================================================
def rgb_to_hsv_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)

    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)

    h = torch.zeros_like(maxc)
    mask = (delta > eps) & (maxc == r)
    h[mask] = (bc - gc)[mask]
    mask = (delta > eps) & (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = (delta > eps) & (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)

def hue_circular_diff(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    d = torch.abs(h1 - h2)
    return torch.minimum(d, 1.0 - d)

def red_mask_from_x(x_rgb: torch.Tensor, sat_thr: float = 0.20, val_thr: float = 0.20) -> torch.Tensor:
    hsv = rgb_to_hsv_torch(torch.clamp(x_rgb, 0, 1))
    h = hsv[:, 0] * 360.0
    s = hsv[:, 1]
    v = hsv[:, 2]
    rm = (((h <= 20.0) | (h >= 340.0)) & (s >= sat_thr) & (v >= val_thr)).float()
    return rm.unsqueeze(1)


# ============================================================
# Scoring functions
# ============================================================
def score_hue_mse(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    hsv_x = rgb_to_hsv_torch(torch.clamp(x, 0, 1))
    hsv_r = rgb_to_hsv_torch(torch.clamp(xhat, 0, 1))
    dh = hue_circular_diff(hsv_x[:, 0], hsv_r[:, 0])
    return (dh ** 2).mean(dim=(1, 2))  # [B]

def score_hue_p999(x: torch.Tensor, xhat: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    hsv_x = rgb_to_hsv_torch(torch.clamp(x, 0, 1))
    hsv_r = rgb_to_hsv_torch(torch.clamp(xhat, 0, 1))
    dh = hue_circular_diff(hsv_x[:, 0], hsv_r[:, 0])
    flat = (dh ** 2).reshape(dh.size(0), -1)
    return torch.quantile(flat, q, dim=1)

def score_hue_red_weighted(x: torch.Tensor,
                           xhat: torch.Tensor,
                           lam: float = 8.0,
                           sat_thr: float = 0.20,
                           val_thr: float = 0.20) -> torch.Tensor:
    hsv_x = rgb_to_hsv_torch(torch.clamp(x, 0, 1))
    hsv_r = rgb_to_hsv_torch(torch.clamp(xhat, 0, 1))
    dh2 = hue_circular_diff(hsv_x[:, 0], hsv_r[:, 0]) ** 2

    base = dh2.mean(dim=(1, 2))

    rm = red_mask_from_x(x, sat_thr=sat_thr, val_thr=val_thr)[:, 0]
    red_sum = (dh2 * rm).sum(dim=(1, 2))
    red_cnt = rm.sum(dim=(1, 2)).clamp_min(1.0)
    red_mean = red_sum / red_cnt

    return base + lam * red_mean


# ============================================================
# Threshold selection + metrics
# ============================================================
def best_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    best_f = -1.0
    best_t = float(thr[0])
    for t in thr:
        y_pred = (scores >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t

def best_threshold_by_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i])

def point_for_threshold(y_true: np.ndarray, scores: np.ndarray, thr_value: float) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = int(np.argmin(np.abs(thr - thr_value)))
    return float(fpr[idx]), float(tpr[idx])

def metrics_at_thr(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "sens": sens, "spec": spec, "acc": acc}


# ============================================================
# Patient aggregation
# ============================================================
def aggregate_patient_scores(pat_ids: np.ndarray,
                             labels: np.ndarray,
                             scores: np.ndarray,
                             agg: str = "p95",
                             topk: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dfp = pd.DataFrame({"pat": pat_ids, "y": labels, "s": scores})
    pats, y_pat, s_pat = [], [], []
    for pat, g in dfp.groupby("pat"):
        pats.append(pat)
        y_pat.append(int(g["y"].max()))
        ss = g["s"].to_numpy()
        if agg == "p95":
            s_pat.append(float(np.quantile(ss, 0.95)))
        elif agg == "max":
            s_pat.append(float(np.max(ss)))
        elif agg == "mean_topk":
            ss_sorted = np.sort(ss)[::-1]
            k = min(topk, len(ss_sorted))
            s_pat.append(float(ss_sorted[:k].mean()))
        else:
            raise ValueError("Unknown patient agg")
    return np.array(pats), np.array(y_pat, dtype=int), np.array(s_pat, dtype=float)


# ============================================================
# Pretty plot: ROC mean + table accuracy
# ============================================================
def plot_mean_roc(
    mean_fpr: np.ndarray,
    tprs: List[np.ndarray],
    aucs: List[float],
    thr_global: float,
    out_png: Path,
    title: str,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Curvas por fold + AUC en leyenda
    for i, tpr_fold in enumerate(tprs):
        auc_i = aucs[i] if i < len(aucs) else float("nan")
        plt.plot(mean_fpr, tpr_fold, lw=1, alpha=0.25, label=f"Fold {i+1} (AUC={auc_i:.3f})")

    # Media + banda
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc_val = float(np.nanmean(aucs))
    std_auc_val = float(np.nanstd(aucs))

    plt.plot(
        mean_fpr,
        mean_tpr,
        lw=2.5,
        label=f"MEAN (AUC={mean_auc_val:.3f} ± {std_auc_val:.3f}) | tau={thr_global:.4g}",
    )
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.18,
        label="± 1 std",
    )

    # Diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.grid(alpha=0.3)

    # Leyenda fuera (si hay muchos folds)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()




# ============================================================
# Main
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--annotated-root", default=str(DEFAULT_ANNOTATED_ROOT))
    ap.add_argument("--excel", default=str(DEFAULT_EXCEL_PATH))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    ap.add_argument("--img-size", type=int, default=0, help="0 -> use ckpt img_size")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--metric", type=str, default="hue_red_weighted",
                    choices=["hue_mse", "hue_p999", "hue_red_weighted"])
    ap.add_argument("--q", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--sat-thr", type=float, default=0.20)
    ap.add_argument("--val-thr", type=float, default=0.20)

    ap.add_argument("--thr-method", type=str, default="youden", choices=["youden", "f1"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--patient-agg", type=str, default="p95", choices=["p95", "max", "mean_topk"])
    ap.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()
    set_seed(args.seed)

    annotated_root = Path(args.annotated_root)
    excel_path = Path(args.excel)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    df = build_merged_df(annotated_root, excel_path)

    # patient stratification
    pat_y = df.groupby("pat_id")["label"].max()
    pats = pat_y.index.to_numpy()
    y_p = pat_y.values.astype(int)

    num_pos_pat = int((y_p == 1).sum())
    num_pat = len(pats)

    k = min(args.k, num_pat)
    if num_pos_pat > 0:
        k = min(k, num_pos_pat)
    k = max(k, 2)

    print(f"[INFO] Patients total={num_pat} pos_patients={num_pos_pat} -> using k={k}")
    print(f"[INFO] Metric={args.metric} thr_method={args.thr_method} patient_agg={args.patient_agg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(ckpt_path, device)

    ckpt_img_size = int(ckpt.get("img_size", 256))
    img_size = int(args.img_size) if int(args.img_size) > 0 else ckpt_img_size
    ckpt_config = str(ckpt.get("config", "3"))
    print(f"[INFO] Using checkpoint: {ckpt_path}")
    print(f"[INFO] Using img_size={img_size} | ckpt_img_size={ckpt_img_size} | config={ckpt_config}")

    ds = AnnotatedDataset(df, img_size=img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    scores = np.zeros(len(df), dtype=np.float32)
    labels = df["label"].to_numpy(dtype=int)
    pats_per_sample = df["pat_id"].astype(str).to_numpy()

    print(f"[STEP] Computing scores ({args.metric}) ...")
    with torch.no_grad():
        idx0 = 0
        for xb, yb, pb, paths in tqdm(loader, total=len(loader)):
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            xhat = out[0] if isinstance(out, (tuple, list)) else out

            if args.metric == "hue_mse":
                sc = score_hue_mse(xb, xhat)
            elif args.metric == "hue_p999":
                sc = score_hue_p999(xb, xhat, q=args.q)
            elif args.metric == "hue_red_weighted":
                sc = score_hue_red_weighted(xb, xhat, lam=args.lam, sat_thr=args.sat_thr, val_thr=args.val_thr)
            else:
                raise ValueError("Unknown metric")

            sc = torch.nan_to_num(sc, nan=0.0, posinf=1e6, neginf=0.0).detach().cpu().numpy()
            scores[idx0:idx0+len(sc)] = sc
            idx0 += len(sc)

    df_out = df.copy()
    df_out["score"] = scores
    df_out.to_csv(out_dir / "all_scores.csv", index=False)

    def pick_thr(y_true, sc):
        return best_threshold_by_f1(y_true, sc) if args.thr_method == "f1" else best_threshold_by_youden(y_true, sc)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

    folds_dir = out_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    mean_fpr = np.linspace(0, 1, 250)

    # PATCH
    tprs_patch, aucs_patch, accs_patch = [], [], []
    oof_patch_scores, oof_patch_labels = [], []

    # PATIENT
    tprs_pat, aucs_pat, accs_pat = [], [], []
    oof_pat_scores, oof_pat_labels = [], []

    summary_rows = []

    for fold, (tr_pat_idx, te_pat_idx) in enumerate(skf.split(pats, y_p), start=1):
        tr_pats = set(pats[tr_pat_idx].tolist())
        te_pats = set(pats[te_pat_idx].tolist())

        tr_idx = np.where(np.isin(pats_per_sample, list(tr_pats)))[0]
        te_idx = np.where(np.isin(pats_per_sample, list(te_pats)))[0]

        y_tr, s_tr = labels[tr_idx], scores[tr_idx]
        y_te, s_te = labels[te_idx], scores[te_idx]

        fold_dir = folds_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # ----- PATCH -----
        thr_patch = pick_thr(y_tr, s_tr)
        auc_patch = float(roc_auc_score(y_te, s_te)) if len(np.unique(y_te)) == 2 else float("nan")
        m_patch = metrics_at_thr(y_te, s_te, thr_patch)

        fpr, tpr, _ = roc_curve(y_te, s_te)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0

        tprs_patch.append(interp)
        aucs_patch.append(auc_patch)
        accs_patch.append(float(m_patch["acc"]))

        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(fold_dir / "roc_test_patch.csv", index=False)
        pd.DataFrame(df.iloc[te_idx]).assign(score=s_te, pred=(s_te >= thr_patch).astype(int)).to_csv(
            fold_dir / "preds_test_patch.csv", index=False
        )

        oof_patch_scores.append(s_te)
        oof_patch_labels.append(y_te)

        # ----- PATIENT -----
        _pats_tr_u, y_tr_pat, s_tr_pat = aggregate_patient_scores(
            pats_per_sample[tr_idx], y_tr, s_tr, agg=args.patient_agg, topk=args.topk
        )
        pats_te_u, y_te_pat, s_te_pat = aggregate_patient_scores(
            pats_per_sample[te_idx], y_te, s_te, agg=args.patient_agg, topk=args.topk
        )

        thr_pat = pick_thr(y_tr_pat, s_tr_pat)
        auc_pat = float(roc_auc_score(y_te_pat, s_te_pat)) if len(np.unique(y_te_pat)) == 2 else float("nan")
        m_pat = metrics_at_thr(y_te_pat, s_te_pat, thr_pat)

        fprp, tprp, _ = roc_curve(y_te_pat, s_te_pat)
        interp_p = np.interp(mean_fpr, fprp, tprp)
        interp_p[0] = 0.0

        tprs_pat.append(interp_p)
        aucs_pat.append(auc_pat)
        accs_pat.append(float(m_pat["acc"]))

        pd.DataFrame({"fpr": fprp, "tpr": tprp}).to_csv(fold_dir / "roc_test_patient.csv", index=False)
        pd.DataFrame({"pat_id": pats_te_u, "label": y_te_pat, "score": s_te_pat,
                      "pred": (s_te_pat >= thr_pat).astype(int)}).to_csv(
            fold_dir / "preds_test_patient.csv", index=False
        )

        oof_pat_scores.append(s_te_pat)
        oof_pat_labels.append(y_te_pat)

        summary_rows.append({
            "fold": fold,
            "thr_patch_train": float(thr_patch),
            "auc_patch_test": float(auc_patch),
            "acc_patch": float(m_patch["acc"]),
            "sens_patch": float(m_patch["sens"]),
            "spec_patch": float(m_patch["spec"]),
            "thr_patient_train": float(thr_pat),
            "auc_patient_test": float(auc_pat),
            "acc_patient": float(m_pat["acc"]),
            "sens_patient": float(m_pat["sens"]),
            "spec_patient": float(m_pat["spec"]),
        })

        print(
            f"[FOLD {fold:02d}] "
            f"PATCH AUC={auc_patch:.4f} thr={thr_patch:.6g} acc={m_patch['acc']:.4f} sens={m_patch['sens']:.4f} spec={m_patch['spec']:.4f} | "
            f"PAT AUC={auc_pat:.4f} thr={thr_pat:.6g} acc={m_pat['acc']:.4f} sens={m_pat['sens']:.4f} spec={m_pat['spec']:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "kfold_summary.csv", index=False)

    # --- GLOBAL thresholds (OOF)
    y_oof_patch = np.concatenate(oof_patch_labels)
    s_oof_patch = np.concatenate(oof_patch_scores)
    thr_global_patch = pick_thr(y_oof_patch, s_oof_patch)

    y_oof_pat = np.concatenate(oof_pat_labels)
    s_oof_pat = np.concatenate(oof_pat_scores)
    thr_global_pat = pick_thr(y_oof_pat, s_oof_pat)

    plot_mean_roc(
      mean_fpr=mean_fpr,
      tprs=tprs_patch,
      aucs=aucs_patch,
      thr_global=thr_global_patch,
      out_png=out_dir / "ROC_PATCH.png",
      title=f"ROC: AE {img_size}px - Metric: {args.metric} (PATCH)",
    )
    
    plot_mean_roc(
        mean_fpr=mean_fpr,
        tprs=tprs_pat,
        aucs=aucs_pat,
        thr_global=thr_global_pat,
        out_png=out_dir / "ROC_PATIENT.png",
        title=f"ROC: AE {img_size}px - Metric: {args.metric} (PATIENT, agg={args.patient_agg})",
    )


    print("\n" + "="*70)
    print(f"[DONE] out={out_dir}")
    print(f"[PATCH]  mean_auc={np.nanmean(aucs_patch):.4f} ± {np.nanstd(aucs_patch):.4f} | "
          f"mean_acc={np.mean(accs_patch):.4f} ± {np.std(accs_patch):.4f} | global_tau={thr_global_patch:.6g}")
    print(f"[PATIENT] mean_auc={np.nanmean(aucs_pat):.4f} ± {np.nanstd(aucs_pat):.4f} | "
          f"mean_acc={np.mean(accs_pat):.4f} ± {np.std(accs_pat):.4f} | global_tau={thr_global_pat:.6g}")
    print("="*70)


if __name__ == "__main__":
    main()
