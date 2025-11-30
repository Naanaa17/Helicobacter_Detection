
# Detecció d’*Helicobacter pylori* en WSI (pipeline per patches)  
## Sistema 1 (Autoencoder) + Sistema 2 (Triplet + MIL amb Attention)

Projecte de detecció d’*H. pylori* a partir de *patches* d’imatges d’histopatologia, dividit en dos blocs:

- **Sistema 1 (AE / detecció per anomalia):** entrenem un **Autoencoder (AE)** només amb pacients sans i usem l’error de reconstrucció (RGB + component HSV/Hue) per definir un *score* i un **threshold** via ROC.
- **Sistema 2 (Triplet + MIL + Attention):** fem servir l’espai latent del Sistema 1 (*z* per patch), el transformem a embeddings més discriminatius amb **Triplet loss** (*z → z’*) i finalment entrenem un **model MIL amb Attention** per classificar el pacient.

**Assignatura:** *Mètodes Avançats de Processament de Senyal, Imatge i Vídeo (UAB)*  
**Informe:** `Hyplori_informe.pdf` (explicació completa + resultats)

---

## Contingut del repositori (fitxers clau)

### Sistema 1 — Autoencoder + Threshold (ROC)
- `train_sys1_ae_paper_hsv.py`  
  Entrenament del **teacher AutoEncoderCNN** només amb *healthy patches*, amb loss: `MSE_RGB + w * MSE_Hue/HSV`.
- `sbatch_sys1_train_ae_paper_hsv.sh`  
  Script Slurm per entrenar l’AE al clúster.
- `system1_reconstruct_grids_ae_best.py`  
  Genera graelles de reconstrucció **healthy vs unhealthy** per inspecció visual.
- `extract_latent_by_patient.py` i `train_ae_lat.py`  
  Extracció i guardat de latents per pacient (`.npz`).
- `system1_threshold.py`  
  Càlcul de *scores* i selecció de **threshold** amb ROC (output: `ROC_PATCH.png`, `ROC_PATIENT.png`, `kfold_summary.csv`, `all_scores.csv`).
- Outputs ja pujats:
  - `ROC_PATCH.png`
  - `ROC_PATIENT.png`
  - `all_scores.csv`
  - `kfold_summary.csv`

### Sistema 2 — Triplet (z → z’) + MIL amb Attention
- `mlp_triplet.py`  
  Entrena una MLP amb **TripletLoss** per obtenir embeddings `z'` més separables.
- `mlp_triplet.sh`  
  Script Slurm per entrenar la MLP Triplet.
- `attention.py`  
  Entrena el model **MIL + Attention** a nivell de pacient sobre els `z'`.
- `sbatch_attention.sh`  
  Script Slurm per entrenar el model d’Attention.
- `test_kfold_attention.py`  
  Test K-Fold del model MIL amb mètriques i figures.
- Fitxers de suport:
  - `AttentionUnits.py`
  - `datasets.py`
  - `triplet_loss.py`
- Outputs ja pujats:
  - `roc_kfold.png`
  - `confusion_global.png`
  - `config.json`

---

## Estructura de dades (paths que fem servir)

**Patches separats (ja preparats):**
- Healthy: `/export/fhome/maed04/Cross_validation/Separated/healthy/<PATIENT_SECTION>/*.png`
- Unhealthy: `/export/fhome/maed04/Cross_validation/Separated/unhealthy/<PATIENT_SECTION>/*.png` *(si aplica)*

**Annotated (per validació/ROC/threshold):**
- `/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/`
- Excel etiquetes: `/export/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx`

**Models i runs (segons scripts):**
- Sistema 1 AE: `~/Codi_nana_results/...`
- Sistema 2 (triplet + attention): `/export/fhome/maed04/sys2_nana/runs/...`

> Nota: els scripts estan pensats per executar-se en entorn HPC amb Slurm, però també es poden executar manualment si les rutes existeixen.

---

## Com executar el pipeline (ordre recomanat)

### 0) Activar entorn
```bash
source /export/fhome/maed04/MyVirtualEnv/bin/activate
````

---

## 1) Sistema 1 — Entrenar Autoencoder (AE)

Opció HPC (Slurm):

```bash
sbatch sbatch_sys1_train_ae_paper_hsv.sh
```

Opció manual:

```bash
python -u train_sys1_ae_paper_hsv.py
```

---

## 2) Sistema 1 — Visualitzar reconstruccions (healthy vs unhealthy)

```bash
python -u system1_reconstruct_grids_ae_best.py
```

---

## 3) Sistema 1 — Extreure latents *z* per pacient

```bash
python -u extract_latent_by_patient.py
```

*(Si tens una variant/runner d’extracció: `train_ae_lat.py`)*

---

## 4) Sistema 1 — Calcular score + threshold amb ROC (K-Fold)

```bash
python -u system1_threshold.py
```

Això genera (entre d’altres):

* `ROC_PATCH.png`, `ROC_PATIENT.png`
* `all_scores.csv`, `kfold_summary.csv`

---

## 5) Sistema 2 — Triplet: entrenar MLP (z → z’)

Opció HPC:

```bash
sbatch mlp_triplet.sh
```

Opció manual:

```bash
python -u mlp_triplet.py
```

Sortida típica:

* carpeta de run a `/export/fhome/maed04/sys2_nana/runs/triplet_mlp_*/`
* `zprime/healthy` i `zprime/unhealthy` amb `.npz` per pacient

---

## 6) Sistema 2 — Entrenar MIL + Attention sobre z’

Opció HPC:

```bash
sbatch sbatch_attention.sh
```

Opció manual:

```bash
python -u attention.py
```

Sortida típica:

* `.../runs/attention_mil_*/checkpoints/attention_best.pt`

---

## 7) Sistema 2 — Test (K-Fold) + figures

```bash
python -u test_kfold_attention.py
```

Genera:

* `roc_kfold.png`
* `confusion_global.png`
* (i outputs per fold si ho tens activat)

---

## Resultats (què mirar ràpid)

### Sistema 1

* `ROC_PATCH.png` i `ROC_PATIENT.png` per veure separació per patch i per pacient
* `kfold_summary.csv` per resum de mètriques per fold
* `all_scores.csv` per auditar exemples i score per mostra

### Sistema 2

* `roc_kfold.png` (ROC mitjana K-Fold)
* `confusion_global.png` (matriu confusió global del test)

---

## Detalls importants / decisions del disseny

* **És un AE (no un VAE):** l’arquitectura `AutoEncoderCNN` no té mostreig probabilístic ni KL-divergence, per tant és un **Autoencoder clàssic**.
* **Per què HSV/Hue:** la component Hue ajuda a capturar canvis relacionats amb el “senyal vermell” (tinció) d’*H. pylori*. Per això al Sistema 1 fem servir una loss que combina RGB amb HSV/Hue.
* **Triplet abans d’Attention:** el Triplet força que patches similars quedin a prop i diferents lluny a l’espai embedding (`z’`), i això acostuma a ajudar que l’Attention tingui una representació més “neteja” a nivell pacient.

---

## Notes d’execució (HPC / Slurm)

* Outputs `.out/.err` habitualment a: `/export/fhome/maed04/Sortida/`
* Si un script dóna “No such file”, assegura que el `cd` del `.sh` apunta al directori correcte i que el nom del `.py` coincideix.

---


## Autors

Adrián Fuster, Marc Cases, Álvaro Bello, Namanmahi Kumar
