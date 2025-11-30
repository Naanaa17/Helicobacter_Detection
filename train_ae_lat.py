#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:26:14 2025

@author: mcases
"""

#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Extraer y guardar vectores z (bottleneck) del AutoEncoder entrenado,
agrupados por paciente.

Por cada paciente se guarda un .npz con:
    z      -> matriz (N_patches, latent_dim)
    paths  -> lista de rutas de cada patch en el mismo orden
"""

import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from AEmodels import AutoEncoderCNN  # mismo que en tu script de entrenamiento


# ============================================================
# 0. PATHS Y PARAMS
# ============================================================
HEALTHY_ROOT = Path("/export/fhome/maed04/Cross_validation/Separated/unhealthy")

RESULTS_DIR = Path.home() / "Codi_nana_results" / "ae_prof_fredish"
CKPT_PATH = RESULTS_DIR / "ae_prof_fredish_best.pt"

# Carpeta donde guardaremos los .npz de latentes
LATENTS_DIR = RESULTS_DIR / "latents_unhealthy"
LATENTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
BATCH_SIZE = 64


# ============================================================
# 1. MISMA FUNCIÓN DE CONFIG DEL AE
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

        net_paramsDec["block_configs"] = [
            [64, 32],
            [32, input_channels],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "2":
        # misma que usabas para entrenar
        net_paramsEnc["block_configs"] = [[32], [64], [128], [256]]
        net_paramsEnc["stride"] = [[2], [2], [2], [2]]

        net_paramsDec["block_configs"] = [
            [128],
            [64],
            [32],
            [input_channels],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "3":
        net_paramsEnc["block_configs"] = [[32], [64], [64]]
        net_paramsEnc["stride"] = [[1], [2], [2]]

        net_paramsDec["block_configs"] = [
            [64],
            [32],
            [input_channels],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]
    else:
        raise ValueError(f"Unknown config: {config}")

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


# ============================================================
# 2. DATASET SIMPLE PARA PARCHES DE UN PACIENTE
# ============================================================
class PatientPatchesDataset(Dataset):
    def __init__(self, patch_paths: List[Path], img_size: int = 128):
        self.paths = patch_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        x = torch.from_numpy(arr)
        return x


# ============================================================
# 3. CARGA DEL MODELO Y PREPARACIÓN DEL ENCODER
# ============================================================
def load_trained_encoder():
    print(f"[INFO] Cargando checkpoint desde {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Leemos la config usada al entrenar (la guardaste en el ckpt)
    config_name = ckpt.get("config", "2")  # por defecto "2" si no estuviera
    print(f"[INFO] Config del AE: {config_name}")

    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = build_ae_configs(
        config_name,
        input_channels=inputmodule_paramsEnc["num_input_channels"],
    )

    model = AutoEncoderCNN(
        inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ==== MUY IMPORTANTE ====
    # Aquí asumimos que el modelo tiene un método `encode(x)`
    # que devuelve el bottleneck. Si no existe, ver comentario más abajo.
    if not hasattr(model, "encode"):
        raise AttributeError(
            "AutoEncoderCNN no tiene método `encode(x)`. "
            "Añádelo en AEmodels.py o expón el encoder (por ejemplo self.encoder)."
        )

    return model


# ============================================================
# 4. EXTRAER Z POR PACIENTE
# ============================================================
def get_patient_folders(root: Path) -> List[Path]:
    if not root.is_dir():
        raise RuntimeError(f"Root no existe: {root}")
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        raise RuntimeError(f"No hay subcarpetas de pacientes en {root}")
    print("\n[INFO] Pacientes encontrados:")
    for d in subdirs:
        print("  -", d.name)
    print()
    return subdirs


def extract_and_save_latents():
    model = load_trained_encoder()
    patient_dirs = get_patient_folders(HEALTHY_ROOT)

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        out_file = LATENTS_DIR / f"{patient_name}.npz"

        # Si ya existe, puedes saltarlo (o sobreescribir, como prefieras)
        if out_file.exists():
            print(f"[SKIP] {out_file} ya existe, lo salto.")
            continue

        # Lista de patches del paciente
        patch_paths = sorted(
            Path(p) for p in glob.glob(str(patient_dir / "*.png"))
        )
        if not patch_paths:
            print(f"[WARN] Paciente {patient_name} sin .png, se omite.")
            continue

        print(f"[INFO] Procesando paciente {patient_name} con {len(patch_paths)} patches...")

        ds = PatientPatchesDataset(patch_paths, img_size=IMG_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_z = []

        with torch.no_grad():
            for x in tqdm(dl, desc=f"{patient_name}"):
                x = x.to(DEVICE, non_blocking=True)

                # Sacamos bottleneck
                z = model.encode(x)  # (B, latent_dim) o (B, C, H, W)

                # Si el encoder devuelve un mapa, lo aplanamos
                if z.dim() > 2:
                    z = torch.flatten(z, start_dim=1)  # (B, C*H*W)

                all_z.append(z.cpu())

        Z = torch.cat(all_z, dim=0).numpy().astype(np.float32)  # (N_patches, latent_dim)

        print(f"[INFO] Matriz Z para {patient_name}: {Z.shape}")
        paths_str = np.array([str(p) for p in patch_paths])

        # Guardamos en .npz comprimido
        np.savez_compressed(out_file, z=Z, paths=paths_str)
        print(f"[SAVE] Guardado {out_file}")


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    print("[INFO] DEVICE:", DEVICE)
    extract_and_save_latents()
    print("[DONE] Extracción de latentes completada.")
