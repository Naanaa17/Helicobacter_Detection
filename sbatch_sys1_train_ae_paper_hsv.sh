#!/bin/bash
#SBATCH --job-name=sys1_ae_hsv
#SBATCH --partition=tfg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o /export/fhome/maed04/Sortida/%x_%j.out
#SBATCH -e /export/fhome/maed04/Sortida/%x_%j.err

set -euo pipefail

source /export/fhome/maed04/MyVirtualEnv/bin/activate

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD before cd: $(pwd)"
echo "[INFO] SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-N/A}"

# Ir a la carpeta donde está tu script
cd /export/fhome/maed04/last_dance
echo "[INFO] PWD after cd: $(pwd)"
echo "[INFO] Listing:"
ls -lah

nvidia-smi || true

# Ejecutar el script (ruta relativa para que sea obvio)
python -u ./train_sys1_ae_paper_hsv.py
