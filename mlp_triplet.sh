#!/bin/bash
#SBATCH --job-name=sys2_triplet
#SBATCH --partition=tfg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH -o /export/fhome/maed04/Sortida/%x_%j.out
#SBATCH -e /export/fhome/maed04/Sortida/%x_%j.err

set -euo pipefail

source /export/fhome/maed04/MyVirtualEnv/bin/activate

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD before cd: $(pwd)"
nvidia-smi || true

cd /export/fhome/maed04/sys2_nana
echo "[INFO] PWD after cd: $(pwd)"
ls -lah

python -u ./mlp_triplet.py \
  --latents-healthy   /export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv_latents_healthy \
  --latents-unhealthy /export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv_latents_unhealthy \
  --emb-dim 128 --epochs 60 --batch-size 512 --margin 1.0
