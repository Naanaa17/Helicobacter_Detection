#!/bin/bash
#SBATCH --job-name=sys2_att
#SBATCH --partition=tfg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o /export/fhome/maed04/Sortida/%x_%j.out
#SBATCH -e /export/fhome/maed04/Sortida/%x_%j.err

set -euo pipefail
source /export/fhome/maed04/MyVirtualEnv/bin/activate

cd /export/fhome/maed04/sys2_nana
nvidia-smi || true

python -u ./attention.py \
  --zprime-healthy   /export/fhome/maed04/sys2_nana/runs/triplet_mlp_20251127_003153/zprime/healthy \
  --zprime-unhealthy /export/fhome/maed04/sys2_nana/runs/triplet_mlp_20251127_003153/zprime/unhealthy \
  --epochs 80 --lr 1e-4 --att-decomp-dim 128 --att-branches 1 --max-patches 0
