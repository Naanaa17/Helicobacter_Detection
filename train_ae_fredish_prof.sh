#!/bin/bash
#SBATCH -p tfg                 # particion TFG
#SBATCH -c 4                   # 4 CPUs
#SBATCH -N 1                   # 1 node
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --mem=32G              # 16 GB RAM
#SBATCH --job-name=ae_train_lat
#SBATCH --time=10:00:00        # hasta 10 horas
#SBATCH --output=/export/fhome/maed04/Codi_nana_debora/ae_fredish_prof_%j.out
#SBATCH --error=/export/fhome/maed04/Codi_nana_debora/ae_fredish_prof_%j.err

# activar entorno virtual
source ~/MyVirtualEnv/bin/activate

# ir a la carpeta del codigo
cd /export/fhome/maed04/sys2_alvarocases

# ejecutar entrenamiento
python -u attention.py
