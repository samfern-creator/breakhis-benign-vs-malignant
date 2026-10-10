# BreakHis: Benign vs Malignant Classification

Classifying breast histopathology images as **benign** or **malignant** with a pretrained convolutional neural network (ResNet34, fastai). The model is evaluated with **patient-level cross-validation**, and every run is tracked with **MLflow** so any prediction can be traced back to the exact run, settings, and features that produced it.

**Headline result:** out-of-fold patient-level AUC of **0.954** across all 70 patients (5-fold CV, 200X magnification).

**Presentation:** [BreakHis-Classification-slides.pdf](BreakHis-Classification-slides.pdf), a 12-slide overview of the project.

> [!NOTE]
> Sections marked **🟧 MLflow** are where experiment tracking happens: logging parameters, metrics, artifacts, and models. They are the backbone of the project's reproducibility.

---

## Contents
| # | Section | Uses MLflow |
|---|---|---|
| 0 | [Setup](#0-setup) | 🟧 configures tracking |
| 1 | [Data overview](#1-data-overview) | |
| 2 | [Patient IDs and train/validation split](#2-patient-ids-and-trainvalidation-split) | |
| 3 | [DataLoaders](#3-dataloaders) | |
| 4 | [Train](#4-train--mlflow) | 🟧 **yes** |
| 5 | [Evaluation](#5-evaluation--mlflow) | 🟧 **yes** |
| 6 | [Error analysis by tumor type](#6-error-analysis-by-tumor-type) | |
| 7 | [Patient-level cross-validation](#7-patient-level-cross-validation--mlflow) | 🟧 **yes** |
| 8 | [Epochs comparison (2 vs 4)](#8-epochs-comparison-2-vs-4--mlflow) | 🟧 **yes** |
| 9 | [Feature export (traceability)](#9-feature-export-traceability--mlflow) | 🟧 **yes** |
| 10 | [Traceability demo](#10-traceability-demo--mlflow) | 🟧 **yes** |
| 11 | [Results and conclusions](#11-results-and-conclusions) | |

---

## Dataset
**BreakHis** (Spanhol et al., 2016, *IEEE Transactions on Biomedical Engineering*): 7,909 H&E-stained microscopy images of breast tumor tissue from 82 patients, collected at the P&D Laboratory in Paraná, Brazil.

- 2,480 benign and 5,429 malignant images; 700×460 px RGB PNG.
- Four magnifications: 40X, 100X, 200X, 400X.
- Eight tumor types. **Benign:** adenosis (A), fibroadenoma (F), phyllodes tumor (PT), tubular adenoma (TA). **Malignant:** ductal (DC), lobular (LC), mucinous (MC), papillary (PC) carcinoma.
- This project uses the **200X** images: 2,013 images from 70 patients (after grouping patient IDs by number; see Section 2).

Filename format: `SOB_B_TA-14-4659-200-001.png` → biopsy method _ class _ tumor type - year - patient/slide ID - magnification - image index.

## Tools
fastai · PyTorch · scikit-learn · pandas · MLflow (SQLite backend) · Kaggle notebook with a T4 GPU

---

## 0. Setup

**Purpose:** Install MLflow, load libraries, make results reproducible, locate the data, and point MLflow at a local tracking database.

**How it relates to deep learning:** Deep learning training involves randomness (weight initialization, data shuffling, augmentation). Fixing the seed makes runs repeatable, which is required before any comparison between runs means anything.

```python
%pip install -q mlflow

from fastai.vision.all import *
from collections import Counter
from sklearn.metrics import roc_auc_score
import re, random, mlflow

set_seed(42, reproducible=True)
print('GPU:', torch.cuda.is_available())

PATH = next(Path('/kaggle/input').rglob('histology_slides')) / 'breast'

mlflow.set_tracking_uri('sqlite:////kaggle/working/mlflow.db')
mlflow.set_experiment('breakhis');
```

- `%pip` installs into the notebook's own Python environment.
- `set_seed(42, reproducible=True)` fixes every source of randomness so results are the same each run.
- `rglob('histology_slides')` **searches** for the data folder instead of hard-coding it, so a change in where Kaggle mounts the dataset can't break the notebook.

> [!IMPORTANT]
> **🟧 MLflow:** `set_tracking_uri('sqlite:///...')` stores all run metadata (parameters, metrics) in a local SQLite database, and artifacts in an `mlruns/` folder next to it. MLflow's plain-folder file store is deprecated, so SQLite is the local option; the same API works unchanged with a remote tracking server. `set_experiment('breakhis')` groups every run in this project together.

---

## 1. Data overview

**Purpose:** Confirm the dataset loaded correctly and check image format before modeling.

**How it relates to deep learning:** A CNN expects inputs of a consistent shape and type. Checking sizes and pixel ranges up front prevents silent errors later.

```python
files = get_image_files(PATH)
print(len(files), 'images')
print(files[0].name)

random.seed(42)
sizes = Counter()
for f in random.sample(list(files), 300):
    with Image.open(f) as im:
        sizes[im.size] += 1
```

- `get_image_files` (fastai) finds every image under the folder.
- The size check uses a **random sample of 300** images rather than all 7,909, since opening every file on Kaggle's network drive is slow. `with Image.open(...)` closes each file after reading.

**Findings**
- 7,909 RGB PNG images (uint8, values 0–255), mostly 700×460; a few are 700×456, which `Resize` handles.
- Some patient folders share a number with different letter suffixes (e.g. `14-22549AB`, `14-22549CD`, `14-22549G`), which likely marks different slides from the same patient.

---

## 2. Patient IDs and train/validation split

**Purpose:** Give every image a patient ID and hold out whole patients for validation.

**How it relates to deep learning:** A deep network can memorize patient-specific details (staining, slide preparation). If the same patient appears in both training and validation, the model is rewarded for recognizing the patient rather than the disease, and results are inflated. This is **data leakage**, and splitting by patient prevents it.

```python
def get_patient(p): return re.search(r'-(\d+-\d+)', p.name).group(1)

MAG = '200X'
files_mag = files.filter(lambda f: f.parent.name == MAG)

df = pd.DataFrame({'path': files_mag})
df['patient'] = df.path.map(get_patient)
df['label'] = df.path.map(lambda p: 'M' if '_M_' in p.name else 'B')
print('Max labels per patient:', df.groupby('patient').label.nunique().max())

rng = random.Random(42)
pats = df.groupby('patient').label.first()
valid_pats = set()
for lab in ['B', 'M']:
    ps = sorted(pats[pats == lab].index)
    valid_pats |= set(rng.sample(ps, round(len(ps) * 0.2)))

def is_valid(p): return get_patient(p) in valid_pats
```

- `get_patient` keeps the **number** (`14-22549`) and drops the letter suffix, so all slides from one patient stay together.
- The "max labels per patient" check confirms no patient has both benign and malignant labels (prints 1).
- About 20% of patients **from each class** are sampled for validation, so benign patients appear in both sets.

---

## 3. DataLoaders

**Purpose:** Define how images are loaded, labeled, split, resized, and augmented for training.

**How it relates to deep learning:** The `DataBlock` is fastai's recipe for feeding a network. Resizing gives the CNN a fixed input size (224×224, what ImageNet-pretrained ResNets expect), and augmentation creates varied versions of each image so the model learns tissue patterns instead of memorizing exact pixels.

```python
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_y=RegexLabeller(r'SOB_([BM])_'),
    splitter=FuncSplitter(is_valid),
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(flip_vert=True))

dls = dblock.dataloaders(files_mag, bs=32)
dls.show_batch(max_n=8)
```

- `RegexLabeller` reads `B` or `M` from the filename (the parent folder is the magnification, so `parent_label` doesn't apply here).
- `FuncSplitter(is_valid)` sends a whole patient to validation, enforcing the patient-level split.
- `aug_transforms(flip_vert=True)` adds random flips, rotations, and zooms. Vertical flips are valid for tissue, which has no "up."
- `dls.vocab` is `['B', 'M']`, so class index 1 = malignant.

---

## 4. Train 🟧 MLflow

**Purpose:** Fine-tune a pretrained ResNet34 and record the run.

**How it relates to deep learning:** **Transfer learning**: the ResNet34 already learned general visual features (edges, textures, shapes) on ImageNet. `fine_tune` first trains only the new classification head, then unfreezes and trains the whole network at a lower learning rate, adapting those features to histopathology.

```python
ARCH, EPOCHS, IMG_SIZE, BS = 'resnet34', 4, 224, 32

with mlflow.start_run(run_name=f'{ARCH}_{MAG}') as run:
    mlflow.log_params({'arch': ARCH, 'epochs': EPOCHS, 'img_size': IMG_SIZE,
                       'batch_size': BS, 'magnification': MAG, 'seed': 42,
                       'n_valid_patients': len(valid_pats)})
    mlflow.log_text('\n'.join(sorted(valid_pats)), 'valid_patients.txt')

    learn = vision_learner(dls, resnet34, metrics=[error_rate, RocAucBinary()])
    learn.fine_tune(EPOCHS)

    for epoch, (trn, val, err, auc) in enumerate(learn.recorder.values):
        mlflow.log_metrics({'train_loss': trn, 'valid_loss': val,
                            'error_rate': err, 'image_auc': auc}, step=epoch)
    RUN_ID = run.info.run_id
```

- `vision_learner(..., resnet34)` loads the pretrained network and adds a new head for 2 classes.
- `RocAucBinary()` reports AUC every epoch. AUC is used instead of accuracy because the classes are imbalanced (about 2× more malignant images).

> [!IMPORTANT]
> **🟧 MLflow**
> - `mlflow.start_run(run_name=...)` opens a run; everything logged inside the `with` block belongs to it.
> - `log_params` records the settings (architecture, epochs, image size, batch size, magnification, seed).
> - `log_text(..., 'valid_patients.txt')` saves the **exact list of validation patients**, so anyone can rebuild the same split.
> - `log_metrics(..., step=epoch)` records loss, error rate, and AUC for every epoch, which shows up as learning curves in the MLflow UI.

---

## 5. Evaluation 🟧 MLflow

**Purpose:** Measure performance per patient (not just per image) and inspect the mistakes.

**How it relates to deep learning:** The network predicts one image at a time, but diagnosis happens per patient. Averaging a patient's image predictions gives a patient-level score. Looking at the highest-loss images shows where the network is confidently wrong.

```python
probs, targs = learn.get_preds()
pred_df = pd.DataFrame({
    'path': dls.valid.items,
    'patient': [get_patient(p) for p in dls.valid.items],
    'label': targs.numpy(),
    'prob_malignant': probs[:, 1].numpy()})

pat_df = pred_df.groupby('patient')[['label', 'prob_malignant']].mean()
patient_auc = roc_auc_score(pat_df.label, pat_df.prob_malignant)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(); fig_cm = plt.gcf()
interp.plot_top_losses(9, nrows=3); fig_tl = plt.gcf()

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metric('patient_auc', patient_auc)
    mlflow.log_figure(fig_cm, 'confusion_matrix.png')
    mlflow.log_figure(fig_tl, 'top_losses.png')
    mlflow.log_artifact('predictions.csv')
```

- `get_preds()` returns the predicted probabilities for every validation image; column 1 is p(malignant).
- `plot_top_losses` shows the images the model was **most confidently wrong** about.

> [!IMPORTANT]
> **🟧 MLflow:** `start_run(run_id=RUN_ID)` **reopens the training run** so evaluation results are stored with the model that produced them: the patient AUC, the confusion matrix and top-loss images (`log_figure`), and a per-image predictions table (`log_artifact`). `mlflow.search_runs(...)` then lists every run with its metrics as a table.

**Findings (single patient split)**
- Image AUC 0.758, patient AUC 0.818, error rate 26%.
- Benign images were the main failure: 45 of 111 correct (41%), versus 242 of 279 malignant (87%).
- Training loss kept falling while validation loss rose after epoch 1.

---

## 6. Error analysis by tumor type

**Purpose:** Find out *which* images the model gets wrong, by tumor type and by patient.

**How it relates to deep learning:** Aggregate metrics hide failure modes. A network can do well on common subtypes and poorly on rare ones. Breaking errors down by subgroup is how you find out what the model actually learned.

```python
pred_df['tumor_type'] = pred_df.path.map(lambda p: p.name.split('_')[2].split('-')[0])
pred_df['pred'] = (pred_df.prob_malignant > 0.5).astype(int)
pred_df['correct'] = pred_df.pred == pred_df.label

pred_df.groupby(['label', 'tumor_type']).correct.agg(['mean', 'count'])
pred_df.groupby(['tumor_type', 'patient']).correct.agg(['mean', 'count'])
```

- The tumor type is parsed straight from the filename.
- Grouping by **patient** as well as type separates "this tumor type is hard" from "this one patient is unusual."

**Findings**
- All fibroadenoma errors in this split came from a **single patient** (14-25197, 0 of 32 correct); tubular adenoma accuracy ranged from 45% to 90% across its two patients.
- The validation set contained only 3 benign patients and was missing several tumor types (no adenosis, phyllodes, or papillary carcinoma). One split was not representative, which motivated cross-validation.

---

## 7. Patient-level cross-validation 🟧 MLflow

**Purpose:** Test every patient exactly once, so the result doesn't depend on which patients happened to land in one validation set.

**How it relates to deep learning:** With small medical datasets, performance varies a lot between splits. K-fold cross-validation trains a **fresh network** on each fold and gives a mean ± standard deviation, a far more reliable estimate than a single split. Out-of-fold predictions give every image a prediction from a model that never saw its patient.

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name=f'cv5_{ARCH}_{MAG}'):
    mlflow.log_params({..., 'n_folds': 5,
                       'split': 'StratifiedGroupKFold(y=tumor_type, groups=patient)'})

    for fold, (_, va_idx) in enumerate(sgkf.split(df, df.tumor_type, df.patient)):
        va_pats = set(df.patient.iloc[va_idx])
        # DataBlock with splitter=FuncSplitter(lambda p: get_patient(p) in va_pats)
        flearn = vision_learner(fdls, resnet34, metrics=[error_rate, RocAucBinary()])
        flearn.fine_tune(EPOCHS)
        # predictions → image AUC and patient AUC for this fold
        mlflow.log_metrics({'fold_image_auc': img_auc, 'fold_patient_auc': pat_auc}, step=fold)

    oof = pd.concat(oof)
    mlflow.log_metrics({'cv_image_auc_mean': ..., 'cv_image_auc_std': ...,
                        'cv_patient_auc_mean': ..., 'cv_patient_auc_std': ...,
                        'oof_patient_auc': oof_pat_auc})
    mlflow.log_artifact('oof_predictions.csv')
```

- `StratifiedGroupKFold`: `groups=patient` keeps each patient in one fold; `y=tumor_type` balances tumor types across folds.
- Each fold trains a **new model from the pretrained weights**, so no fold's model has seen its validation patients.
- **Out-of-fold (OOF) predictions** combine all folds: every image is predicted by the one model that never saw that patient.

> [!IMPORTANT]
> **🟧 MLflow:** One run holds the whole cross-validation. Per-fold metrics are logged with `step=fold` (so they plot like a curve across folds), summary metrics (mean, std, OOF AUC) are logged once, and the full out-of-fold predictions table is saved as an artifact for later analysis.

**Findings**
- Image AUC 0.895 ± 0.077, patient AUC 0.959 ± 0.068, out-of-fold patient AUC across all 70 patients 0.944 (2 epochs).
- Fold-to-fold image AUC ranged from 0.79 to 0.99, so the earlier single split was unrepresentative.
- Out-of-fold by tumor type: ductal carcinoma (38 patients) was classified best (93%); rarer types were weaker (mucinous 68%, adenosis 59%). Fibroadenoma errors came from 2 of its 7 patients.

---

## 8. Epochs comparison (2 vs 4) 🟧 MLflow

**Purpose:** The first CV run changed two things at once (split method and 4 → 2 epochs). Re-running CV with 4 epochs isolates the effect of training length.

**How it relates to deep learning:** Training longer can improve a network or make it overfit. Comparing runs that differ in **one setting only** is how you attribute a change in performance to its cause.

```python
def run_cv(epochs, mag=MAG, n_folds=5, arch=resnet34):
    """Patient-level, tumor-type-stratified CV. Logs everything to one MLflow run."""
    run_name = f'cv{n_folds}_{arch.__name__}_{mag}_ep{epochs}'
    with mlflow.start_run(run_name=run_name) as run:
        ...
    return oof, summary

oof_ep4, summary_ep4 = run_cv(epochs=4)
comparison = pd.DataFrame([summary_ep2, summary_ep4], index=['2 epochs', '4 epochs'])
```

- The cross-validation code is wrapped in a reusable function, so each experiment is one call and earlier results aren't overwritten. The same function can compare magnifications: `run_cv(epochs=4, mag='40X')`.

> [!IMPORTANT]
> **🟧 MLflow:** The run name encodes the settings (`cv5_resnet34_200X_ep4`), and the function returns each run's ID. Comparing experiments becomes a comparison of MLflow runs rather than of variables that get overwritten in the notebook.

**Findings**
| | Image AUC | Patient AUC | OOF patient AUC |
|---|---|---|---|
| 2 epochs | 0.895 ± 0.077 | 0.959 ± 0.068 | 0.944 |
| 4 epochs | 0.895 ± 0.074 | 0.962 ± 0.038 | 0.954 |

- Training length barely matters; the differences are smaller than the fold-to-fold variation.
- The improvement over the single split came from the **evaluation** (testing all patients), not from training length.
- Rising validation loss with stable AUC suggests the model became more overconfident, not worse at ranking.
- **Final configuration: ResNet34, 200X, 4 epochs** (best OOF AUC, most stable across folds).

---

## 9. Feature export (traceability) 🟧 MLflow

**Purpose:** Train the final model and save, for every image, the network's internal feature vector together with its patient, tumor type, label, predictions, and run ID.

**How it relates to deep learning:** A CNN turns an image into a **feature vector** (here, 512 numbers) just before its final decision layer. Exporting these features makes the model's internal representation inspectable: you can see which images it considers similar, and trace any prediction back to what the model "saw."

```python
final_run = mlflow.start_run(run_name=f'final_{ARCH}_{MAG}_ep{FINAL_EPOCHS}')
mlflow.log_params({..., 'purpose': 'final model + feature export',
                   'cv_reference_run': summary_ep4['run_id']})
final_learn = vision_learner(final_dls, resnet34, metrics=error_rate)
final_learn.fine_tune(FINAL_EPOCHS)

feat_model = nn.Sequential(final_learn.model[0], final_learn.model[1][:-1]).eval()
test_dl = final_dls.test_dl(files_m)
with torch.no_grad():
    feats = torch.cat([feat_model(xb).cpu() for (xb,) in test_dl]).numpy()

export = pd.DataFrame({'image_id': ..., 'patient': ..., 'tumor_type': ..., 'label': ...,
                       'magnification': MAG, 'run_id': FINAL_RUN_ID,
                       'prob_malignant_final': ...})
export = export.merge(oof_ep4[['path', 'prob_malignant']] ..., on='path')   # honest OOF prediction
export = pd.concat([export, feat_cols], axis=1)                             # 512 feature columns

export.to_parquet('features_200X.parquet', index=False)
mlflow.log_artifact('features_200X.parquet')
mlflow.log_figure(fig, 'features_pca.png')
final_learn.export('final_model.pkl')
mlflow.log_artifact('final_model.pkl')
mlflow.end_run()
```

- `final_learn.model[0]` is the ResNet **body**; `final_learn.model[1][:-1]` is the **head without its last layer**. Together they output the 512-dimensional features just before the B/M decision.
- `test_dl` applies the same resizing **without random augmentation**, in the same order as the file list, so rows line up.
- Two prediction columns: `prob_malignant_final` (from the final model, which saw these images, so optimistic) and `prob_malignant_oof` (from the CV model that never saw that patient; the honest one).
- **Parquet** stores the 500+ numeric columns compactly with their types.
- **PCA** reduces the 512 features to 2 so the classes can be plotted.

> [!IMPORTANT]
> **🟧 MLflow: this is the core traceability step.** One run holds everything needed to audit the model:
> - the **settings**, plus `cv_reference_run`, which links the final model to the CV run that justified its configuration;
> - the **features table** (`features_200X.parquet`): every image's ID, patient, label, predictions, run ID, and 512 features;
> - the **PCA plot** of the feature space;
> - the **trained model itself** (`final_model.pkl`).

**Findings**
- Malignant images form a tight cluster in feature space, while benign images spread widely: benign tissue is far more heterogeneous to the model.
- The classes overlap in a narrow band, where most errors are expected.
- Caveat: the final model was trained on these images, so the separation shown is optimistic.

---

## 10. Traceability demo 🟧 MLflow

**Purpose:** Show the export in use: take the most confident misclassification and trace it back to its source.

**How it relates to deep learning:** Looking up an error's **nearest neighbors in feature space** shows what the network thinks the image resembles, which helps explain *why* it was misclassified rather than only *that* it was.

```python
export['pred_oof'] = (export.prob_malignant_oof > 0.5).astype(int)
wrong = export[export.pred_oof != export.label]
case = wrong.loc[(wrong.prob_malignant_oof - 0.5).abs().idxmax()]

F = export.filter(like='feat_').values
F = F / np.linalg.norm(F, axis=1, keepdims=True)
sims = F @ F[case.name]
sims[(export.patient == case.patient).values] = -np.inf
nn_idx = np.argsort(-sims)[:6]

with mlflow.start_run(run_id=FINAL_RUN_ID):
    mlflow.log_text(report, f"trace_{case.image_id.replace('.png', '')}.txt")
```

- `wrong` holds every image the CV models got wrong, using the honest out-of-fold predictions.
- `(prob - 0.5).abs().idxmax()` picks the mistake farthest from 0.5, meaning the most confident one.
- **Cosine similarity** measures how alike two feature vectors are, from 1 (same direction) down to −1. Normalizing first makes the dot product equal cosine similarity.
- The same patient's images are **excluded**, since they would obviously look similar; the question is which *other* tissue the model confuses this with.

> [!IMPORTANT]
> **🟧 MLflow:** The investigation itself (the case details and its nearest neighbors) is saved as a text artifact on the final model's run, so the trace is stored next to the model that made the prediction.

**Findings**
- Out-of-fold, 352 of 2,013 images (17.5%) are misclassified.
- **Most confident error:** `SOB_M_PC-14-9146-200-011.png`, a papillary carcinoma from patient 14-9146. The CV model predicted p(malignant) = 0.00; even the final model, trained on it, is unsure (0.47).
- Its nearest neighbors from other patients are mostly benign (3 fibroadenoma, 1 phyllodes tumor), plus 2 papillary carcinomas.
- Interpretation: the glandular, papillary architecture of this carcinoma resembles duct-rich benign lesions in feature space, and papillary carcinoma is rare in the data (6 patients).

---

## 11. Results and conclusions

| Metric (5-fold patient-level CV, 4 epochs) | Value |
|---|---|
| Image-level AUC | 0.895 ± 0.074 |
| Patient-level AUC | 0.962 ± 0.038 |
| **Out-of-fold patient AUC (all 70 patients)** | **0.954** |
| Out-of-fold misclassified images | 352 / 2,013 (17.5%) |

**Key findings**
1. **Evaluation design mattered more than training length.** A single patient split gave image AUC 0.76; cross-validation across all patients gave 0.90.
2. **Errors are driven by patients and tumor-type imbalance.** Ductal carcinoma, the most common type, is classified best; rarer types are weaker.
3. **Benign tissue is more heterogeneous** than malignant tissue in the model's feature space.
4. **Every prediction is traceable** to its patient, features, and MLflow run.

**Limitations**
- One magnification (200X) and one dataset from a single lab; no external validation.
- Small cohort: 70 patients, only ~18 benign.
- No stain normalization; decision threshold fixed at 0.5.
- Tumor-type results come from the 2-epoch CV run.

**Next steps**
- Compare magnifications (40X, 100X, 400X) with `run_cv`.
- Stain normalization and tumor-type–aware loss weighting.
- Validate on an external histopathology dataset.
- Apply the same pipeline (patient-level CV, MLflow, feature export) to mammograms (CBIS-DDSM).

---

## Reproducing
1. Open the notebook on Kaggle, attach the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis), and turn on **GPU** and **Internet**.
2. Run all cells (~30 minutes on a T4 GPU).
3. Outputs in `/kaggle/working`: `mlflow.db`, `mlruns/`, `oof_*.csv`, `features_200X.parquet`, `final_model.pkl`.
4. To browse runs locally, download the outputs and run `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Reference
Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). A Dataset for Breast Cancer Histopathological Image Classification. *IEEE Transactions on Biomedical Engineering*, 63(7), 1455–1462.
