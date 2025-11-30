#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Models.AEmodels import AutoEncoderCNN


# -------------------------
# CONFIG (tus rutas)
# -------------------------
CKPT_PATH = Path("/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv/ae_best.pt")
DATA_ROOT = Path("/export/fhome/maed04/Cross_validation/Separated/unhealthy")  # o unhealthy si quieres
OUT_DIR   = Path("/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv_latents")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 4

# OJO: pon el mismo CONFIG que usaste al entrenar (en tu script era CONFIG="3")
CONFIG = "3"


# -------------------------
# AE configs (igual que entreno)
# -------------------------
def build_ae_configs(config: str, input_channels: int):
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = {}, {}, {}
    net_paramsEnc["drop_rate"] = 0.0
    net_paramsDec["drop_rate"] = 0.0

    if config == "1":
        net_paramsEnc["block_configs"] = [[32, 32], [64, 64]]
        net_paramsEnc["stride"]        = [[1, 2],   [1, 2]]
        net_paramsDec["block_configs"] = [[64, 32], [32, input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "2":
        net_paramsEnc["block_configs"] = [[32], [64], [128], [256]]
        net_paramsEnc["stride"]        = [[2],  [2],  [2],   [2]]
        net_paramsDec["block_configs"] = [[128], [64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "3":
        net_paramsEnc["block_configs"] = [[32], [64], [64]]
        net_paramsEnc["stride"]        = [[1],  [2],  [2]]
        net_paramsDec["block_configs"] = [[64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]
    else:
        raise ValueError(f"Unknown config: {config}")

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


def load_model(ckpt_path: Path, device: torch.device):
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = build_ae_configs(CONFIG, 3)

    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # tu checkpoint guarda "state_dict"
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------------------------
# Dataset por paciente
# -------------------------
class ImgDataset(Dataset):
    def __init__(self, img_paths: List[Path], img_size: int):
        self.img_paths = img_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr), str(p)


def list_patient_dirs(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir()])


# -------------------------
# Main extraction
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] ckpt:", CKPT_PATH)
    print("[INFO] data_root:", DATA_ROOT)
    print("[INFO] out_dir:", OUT_DIR)
    print("[INFO] IMG_SIZE:", IMG_SIZE, "CONFIG:", CONFIG)

    model = load_model(CKPT_PATH, device)

    patient_dirs = list_patient_dirs(DATA_ROOT)
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found in {DATA_ROOT}")

    summary_rows: List[Dict] = []

    for pdir in patient_dirs:
        patient_id = pdir.name
        img_paths = [Path(x) for x in sorted(glob.glob(str(pdir / "*.png")))]
        if not img_paths:
            continue

        ds = ImgDataset(img_paths, IMG_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())

        feats = []
        paths_out = []

        with torch.no_grad():
            for xb, pb in tqdm(dl, desc=f"[{patient_id}] extracting", leave=False):
                xb = xb.to(device, non_blocking=True)

                # LATENT MAP: [B,C,H',W']
                z = model.encoder(xb)

                # convertir a embedding vector por patch: [B,C]
                z_vec = F.adaptive_avg_pool2d(z, output_size=1).squeeze(-1).squeeze(-1)

                feats.append(z_vec.detach().cpu().numpy())
                paths_out.extend(list(pb))

        feats = np.concatenate(feats, axis=0)  # [N_patches, C]
        mean_feat = feats.mean(axis=0)

        # guarda por paciente
        out_npz = OUT_DIR / f"{patient_id}.npz"
        np.savez_compressed(
            out_npz,
            patient_id=patient_id,
            paths=np.array(paths_out, dtype=object),
            feats=feats,          # (N,C)
            mean_feat=mean_feat,  # (C,)
        )

        summary_rows.append({
            "patient_id": patient_id,
            "n_patches": int(feats.shape[0]),
            "feat_dim": int(feats.shape[1]),
            "npz_path": str(out_npz),
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "patients_summary.csv", index=False)
    print("[DONE] Saved:", OUT_DIR / "patients_summary.csv")


if __name__ == "__main__":
    main()
