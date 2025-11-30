# Helicobacter pylori Detection (WSI → patches)
## Sistema 1 (Autoencoder) + Sistema 2 (Triplet + MIL + Attention)

Projecte de detecció d’**Helicobacter pylori** en histologia (WSI) treballant a nivell de **patch**.  
El pipeline està dividit en dos sistemes:

- **Sistema 1 (AE / detecció per anomalia)**: entrenem un *Autoencoder* només amb **patches de pacients sans**. Quan el model veu patrons compatibles amb H. pylori (anomalia), acostuma a **reconstruir pitjor** → obtenim un **score** a partir de l’error de reconstrucció.
- **Sistema 2 (Triplet + MIL + Attention)**: partim dels latents del Sistema 1 (`z` per patch), els fem més discriminatius amb **Triplet Loss** (`z → z'`) i fem classificació **a nivell pacient** amb **MIL + Attention**.

**Assignatura:** *Mètodes Avançats de Processament de Senyal, Imatge i Vídeo (UAB)*  
**Informe:** `Hyplori_informe.pdf` (explicació completa + resultats)

---

## 1) Contingut del repositori (fitxers principals)

### Sistema 1 — Autoencoder + threshold
- `train_sys1_ae_paper_hsv.py`  
  Entrena el **teacher AutoEncoderCNN** només amb *healthy*.  
  Loss:
  - `loss = MSE_RGB + hsv_weight * MSE_H (Hue)` *(o HSV complet segons config)*

- `system1_reconstruct_grids_ae_best.py`  
  Genera **grids de reconstrucció** (healthy vs unhealthy) per inspecció visual.

- `extract_latent_by_patient.py`  
  Extreu l’espai latent `z` **per pacient** i el guarda en `.npz`.

- `system1_threshold.py`  
  Ajust de **threshold (tau)** amb **ROC + KFold** usant **Annotated + Excel**.

### Sistema 2 — Triplet + MIL + Attention
- `mlp_triplet.py`  
  Entrena una MLP amb **Triplet Loss** per transformar **`z → z'`** i desa `z'` per pacient en `.npz`.

- `attention.py`  
  Entrena **MIL + Attention** a nivell pacient usant `z'` com a “bag” de patches.

- `test_kfold_attention.py`  
  Test amb **KFold** del Sistema 2 (ROC/AUC, mètriques i figures).

### Fitxers de suport
- `AttentionUnits.py` (unitat d’atenció + classificador)
- `datasets.py` (TripletDataset)
- `triplet_loss.py` (TripletLoss)

### Scripts SLURM
- `sbatch_sys1_train_ae_paper_hsv.sh`
- `mlp_triplet.sh`
- `sbatch_attention.sh`

---

## 2) Dades i rutes (clúster)

### Dataset separats (train AE)
- **Healthy (train)**  
  `/export/fhome/maed04/Cross_validation/Separated/healthy/<PATIENT_SECTION>/*.png`

### Annotated + Excel (threshold / validació)
- **Annotated**  
  `/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/`
- **Excel labels**  
  `/export/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx`

---

## 3) Outputs (on es guarda tot)

### Sistema 1 — Model entrenat
- **AE best checkpoint**  
  `/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv/ae_best.pt`

### Sistema 2 — Runs
Tot el que és Triplet / Attention es guarda a:
- `/export/fhome/maed04/sys2_nana/runs/`

Exemples típics:
- **Triplet MLP (z')**
  - `.../runs/triplet_mlp_YYYYMMDD_HHMMSS/zprime/healthy/<PACIENT>.npz`
  - `.../runs/triplet_mlp_YYYYMMDD_HHMMSS/zprime/unhealthy/<PACIENT>.npz`

- **Attention MIL (checkpoints)**
  - `.../runs/attention_mil_YYYYMMDD_HHMMSS/checkpoints/attention_best.pt`

---

## 4) Com executar-ho (pipeline recomanat)

### 0) Activar entorn
```bash
source /export/fhome/maed04/MyVirtualEnv/bin/activate
````

### 1) Entrenar Autoencoder (Sistema 1)

```bash
sbatch sbatch_sys1_train_ae_paper_hsv.sh
```

### 2) Visualitzar reconstruccions (healthy vs unhealthy)

```bash
python -u system1_reconstruct_grids_ae_best.py
```

### 3) Extreure latents `z` per pacient

```bash
python -u extract_latent_by_patient.py
```

### 4) Ajustar threshold (tau) amb ROC/KFold

```bash
python -u system1_threshold.py
```

### 5) Entrenar Triplet MLP i generar `z'` (Sistema 2)

```bash
sbatch mlp_triplet.sh
```

### 6) Entrenar MIL + Attention (Sistema 2)

```bash
sbatch sbatch_attention.sh
```

### 7) Test KFold del Sistema 2

```bash
python -u test_kfold_attention.py
```

---

## 5) Notes importants

### AE vs VAE

Aquest projecte fa servir un **Autoencoder (AE)**, **NO** un VAE.
Un VAE tindria `mu`, `logvar` i sampling/reparametrization. Aquí no.

### Per què `.npz`?

Fem servir `.npz` perquè:

* és compacte (`np.savez_compressed`)
* és ràpid de carregar
* permet guardar `z`/`z'` i opcionalment `paths`

### Per què NO posar threshold fix 0.5?

0.5 només té sentit si les probabilitats estan calibrades i la distribució és estable.
Aquí triem `tau` amb **Youden** o **F1-optimal**, dins d’un **KFold** per robustesa.

---

## 6) Entrega (què s’entrega)

* **Codi**: scripts Python + sbatch
* **Informe**: `Hyplori_informe.pdf`
* **Resultats**: figures ROC / matrius de confusió / CSVs generats pels scripts

```
```
