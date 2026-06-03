# ReHeartNet — ECG Reconstruction from PPG

Reconstructs Lead II ECG from single-channel PPG using a **Densely-Connected Bidirectional LSTM (DC-BiLSTM)** on the BIDMC dataset, evaluated with a clinical protocol.

Built on **Lee et al. (2026)**, extended with:
- **Huber loss** replacing MSE — robust to QRS transients
- **CLEF perceptual loss** — frozen ECG foundation model (Nokia Bell Labs) enforces clinical feature consistency during training
- **8-fold group cross-validation** — subject-disjoint, ~7 test subjects per fold
- **Optuna hyperparameter search** — tunes lr, λ, δ, batch size, hidden size
- **Clinical evaluation metrics** — PRD, BCE (via CLEF features), EMD, KS test on RR intervals

---

## Full Pipeline — What to Run and When

> Run every step in order. Each step depends on the previous one.

```
Step 0  One-time setup (conda env + CLEF checkpoint)
Step 1  Hyperparameter tuning     scripts/tune_hyperparams.py   --fast ~15-20 min  |  default ~45-60 min
Step 2  Main 8-fold CV            run_cv.py                     ~16-32 h on GPU
Step 3  ReHeartNet comparison     scripts/compare_reheartnet.py ~same as Step 2    (optional, run in parallel)
Step 4  Architecture ablation     run_cv.py --all-models        ~3× Step 2         (optional)
```

---

### Step 0 — One-time Setup

**a) Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate reheartnet

# Clone, patch (Windows UTF-8 fix), and install CLEF:
python scripts/setup_clef.py
```

> **GPU users** — after activation, reinstall PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> # Replace cu118 with cu121 or cu124 for CUDA 12.x  (check: nvidia-smi)
> ```

**b) Download the CLEF pretrained checkpoint**

```bash
mkdir -p models/clef

# Small (5.5 MB) — default, sufficient for training:
curl -L "https://zenodo.org/records/17572734/files/clef_small.ckpt?download=1" \
     -o models/clef/clef_small.ckpt

# Medium (368 MB) — better features if GPU memory allows:
curl -L "https://zenodo.org/records/17572734/files/clef_medium.ckpt?download=1" \
     -o models/clef/clef_medium.ckpt
```

**Requires:** BIDMC dataset in `data/BIDMC/` (53 `.dat`/`.hea` file pairs from PhysioNet).

---

### Step 1 — Hyperparameter Tuning

Runs Optuna on a small proxy subset and saves best parameters to `results/best_hyperparams.json`.
If skipped, `run_cv.py` uses the defaults in `core/config.py`.

```bash
# ~15-20 min (recommended first run)
python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --fast

# ~45-60 min (better coverage)
python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt

# ~4-8 h (full search)
python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --full

# Resume a previous study
python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --resume
```

**What Optuna searches** (all 5 hyperparameters):

| Hyperparameter | Search space | Scale | Default |
|---|---|---|---|
| Learning rate `lr` | [1e-4, 1e-2] | log-uniform | 1e-3 |
| CLEF loss weight `lambda_clinical` | [1e-3, 1.0] | log-uniform | 0.1 |
| Huber delta `huber_delta` | [0.1, 2.0] | uniform | 1.0 |
| Batch size | {32, 64, 128} | categorical | 64 |
| Hidden size `H` | {32, 64, 128} | categorical | 64 |

**Speed modes:**

| Flag | Subjects | Epochs/trial | Trials | Est. time (CPU) |
|------|----------|-------------|--------|-----------------|
| `--fast` | 5 | 5 | 10 | ~15-20 min |
| *(default)* | 6 | 10 | 20 | ~45-60 min |
| `--full` | 14 | 20 | 50 | ~4-8 h |

**Outputs** → `results/`

| File | Description |
|------|-------------|
| `best_hyperparams.json` | Best lr, λ, δ, batch size, hidden size |
| `optuna_study.db` | SQLite study (resumable) |
| `figures/optuna_history.png` | Trial convergence |
| `figures/optuna_importances.png` | Parameter importances |

---

### Step 2 — Main 8-Fold CV *(proposed model)*

Trains and evaluates **ReHeartNet + CLEF** across all 8 folds.
Reads `results/best_hyperparams.json` automatically if Step 1 was run.

```bash
# Standard run
python run_cv.py --clef-path models/clef/clef_small.ckpt

# Without wandb
python run_cv.py --clef-path models/clef/clef_small.ckpt --no-wandb

# Resume after crash (restart from fold 3)
python run_cv.py --clef-path models/clef/clef_small.ckpt --resume-fold 3

# Sanity check: 2 folds, 2 epochs each (~5 min)
python run_cv.py --clef-path models/clef/clef_small.ckpt --dry-run --no-wandb
```

**Outputs** → `results/`

| File | Description |
|---|---|
| `summary_reheartnet.json` | Mean ± 95% CI for all metrics, 8 folds |
| `fold_assignments.json` | Reproducible train/test subject splits |
| `figures/loss_curves_reheartnet_fold*.png` | Train / val loss per fold |
| `figures/reconstruction_reheartnet_fold*.png` | Real vs reconstructed ECG |
| `figures/rr_dist_reheartnet_fold*.png` | RR interval KDE comparison |
| `figures/metric_bar_reheartnet_*.png` | Per-fold bar charts with 95% CI |
| `figures/metric_boxplots_reheartnet.png` | Box plots across all 8 folds |
| `checkpoints/reheartnet_fold_XX_best.pt` | Best model per fold |

---

### Step 3 — ReHeartNet Comparison *(vs original paper)*

Runs three variants of the same DC-BiLSTM architecture with the same training
protocol as Lee et al. — only the loss function changes:

| Variant | Windows | Loss | Purpose |
|---------|---------|------|---------|
| `reheartnet_original` | 4 s, no overlap, FIR bandpass | MSE | Faithful replica of Lee et al. (2026) |
| `reheartnet_huber` | 4 s, no overlap, FIR bandpass | Huber | Effect of Huber loss |
| `reheartnet_clef` | 10 s, 50% overlap, z-score only | Huber + CLEF | Our full method |

> All variants use batch=1, lr=1e-2, ×0.75 linear decay every 50 epochs, 1000 epochs (paper protocol).
> The only thing Optuna tunes is `huber_delta`, `lambda_clinical`, and `hidden_size` — not the training dynamics.
> `reheartnet_clef` uses 10 s windows because the CLEF encoder requires 10 s input (see BCE note below).

```bash
python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt

# Resume / dry-run
python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt --resume
python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt --dry-run
```

**Outputs** → `results/comparison_reheartnet/`

| File | Description |
|------|-------------|
| `comparison.json` | Side-by-side mean ± CI, all three variants |
| `figures/comparison_panel.png` | **Main comparison figure** (2×3 panel, all 6 metrics) |
| `figures/cmp_{metric}.png` | Per-metric bar chart (green outline = best) |
| `summary_reheartnet_{original,huber,clef}.json` | Per-variant summaries |

> `Δ_Huber` = original → Huber gap = benefit of Huber loss  
> `Δ_CLEF`  = Huber → CLEF gap = benefit of CLEF clinical perceptual regularisation

---

### Step 4 — Architecture Ablation *(optional)*

```bash
python run_cv.py --clef-path models/clef/clef_small.ckpt --all-models --no-wandb
```

Runs: `linear` → `lstm` → `bilstm` → `reheartnet` (same Huber+CLEF loss for all).

---

## Output Directory Map

```
results/
├── best_hyperparams.json            ← Step 1: Optuna best params
├── optuna_study.db                  ← Step 1: resumable study
├── fold_assignments.json            ← Step 2: CV splits (fixed seed)
├── summary_reheartnet.json          ← Step 2: MAIN RESULT
├── figures/
│   ├── optuna_history.png           ← Step 1
│   ├── optuna_importances.png       ← Step 1
│   ├── loss_curves_reheartnet_*.png ← Step 2 (per fold)
│   ├── reconstruction_reheartnet_*.png
│   ├── rr_dist_reheartnet_*.png
│   ├── metric_bar_reheartnet_*.png
│   ├── metric_boxplots_reheartnet.png
│   └── comparison_*.png             ← Step 4 (ablation)
└── comparison_reheartnet/           ← Step 3
    ├── comparison.json
    ├── figures/comparison_panel.png ← KEY FIGURE for paper
    └── figures/cmp_*.png

checkpoints/                         ← ignored by git
├── reheartnet_fold_XX_best.pt       ← Step 2
├── reheartnet_{original,mse,huber,clef}_fold_XX_best.pt  ← Step 3
```

---

## Project Structure

```
Final_Project/
├── core/
│   ├── config.py                   # All hyperparameters and paths
│   ├── data_loader.py              # BIDMCDataset, build_group_fold, build_test_dataset, get_cv_splits
│   ├── train.py                    # train_fold() — loss_type: mse/huber/clef; lr_schedule: plateau/linear_decay
│   ├── evaluate.py                 # evaluate_fold() — all 6 clinical metrics
│   ├── losses/
│   │   └── composite_loss.py       # ClinicalCompositeLoss + load_clef_encoder()
│   ├── models/
│   │   ├── reheartnet.py           # DC-BiLSTM (5 blocks, dense connections)
│   │   ├── blocks.py               # DCBiLSTMBlock
│   │   ├── baselines.py            # LinearReg, SimpleLSTM, PlainBiLSTM + get_model()
│   │   └── ptbxl_classifier.py     # CLEFClassifier (BCE) + SurrogateECGClassifier
│   ├── metrics/
│   │   └── clinical_metrics.py     # PRD, BCE, EMD, KS, beat-timing MAE
│   └── visualization/
│       └── plots.py                # All figure generation
├── src/
│   └── preprocessing.py            # WFDB loading, z-score, FIR bandpass, windowing, phase align
├── scripts/
│   ├── tune_hyperparams.py         # Step 1 — Optuna search (--fast / default / --full)
│   ├── compare_reheartnet.py       # Step 3 — 4-way comparison (original/MSE/Huber/CLEF)
│   ├── sanity_check.py             # Quick end-to-end test (~2 min)
│   └── setup_clef.py               # Clone + patch + install CLEF
├── run_cv.py                       # Steps 2 & 4 — main CV entry point
├── environment.yml                 # Conda environment (Python 3.12)
├── requirements.txt                # pip dependencies
├── data/BIDMC/                     # 53 subjects (not committed)
├── models/clef/                    # CLEF checkpoint (not committed)
└── Final Report/                   # LaTeX paper
```

---

## Model Architecture

**ReHeartNet** — Densely-Connected Bidirectional LSTM (Lee et al. 2026):

> Input length varies by model variant: original/Huber use **(B, 500, 1)** [4 s @ 125 Hz];
> CLEF variant uses **(B, 1250, 1)** [10 s @ 125 Hz]. The BiLSTM is sequence-length agnostic.

```
PPG input  (B, L, 1)         [L=500 for 4 s models; L=1250 for CLEF model]
    ↓
BiLSTM Block 1  (hidden=H)   → 2H-dim output
    ↓  dense: [input(1) + B1(2H)] = (1+2H)-dim input to next
BiLSTM Block 2               → 2H-dim output
    ↓  dense: [input(1) + B1(2H) + B2(2H)] = (1+4H)-dim input
BiLSTM Block 3 ... Block 5
    ↓  concatenate all features: (1 + 5×2H)-dim = 641-dim  [at H=64]
Linear head  →  1 scalar per timestep
    ↓
ECG output  (B, L, 1)        [reconstructed Lead II, same length as input]
```

> **Note on hidden size H:** Lee et al. defer exact hyperparameters to supplementary.
> We default to `H=64` and tune it via Optuna `{32, 64, 128}`.

**Published BIDMC results** from Lee et al. (2026) Table I — LOSO evaluation:

| Model | RMSE (mV) | Pearson r | Beat MAE (s) |
|-------|-----------|-----------|--------------|
| CardioGAN | 0.8696 | 0.0421 | 1.3908 |
| BiLSTM (stacked) | 0.1253 | 0.6263 | 0.5758 |
| Transformer | 0.1365 | 0.4927 | 5.0301 |
| **ReHeartNet (original)** | **0.1070** | **0.7273** | **0.3123** |

Our target: improve on ReHeartNet by replacing MSE with Huber + CLEF perceptual loss.

---

## Training Loss

**Original ReHeartNet (Lee et al. 2026):**
```
L = MSELoss(pred, true)
```

**Ours:**
```
L = HuberLoss(pred, true, δ) + λ × ||Φ(true) − Φ(pred)||²
```

| Term | Role | Tuned by |
|------|------|----------|
| **Huber(δ)** | Point-wise fidelity, robust to QRS transients | Optuna: δ ∈ [0.1, 2.0] |
| **λ ‖Φ−Φ̂‖²** | Clinical feature consistency via frozen CLEF encoder | Optuna: λ ∈ [1e-3, 1.0] |

CLEF preprocessing (inside the loss only — separate from main pipeline):
resample 125→500 Hz · bandpass 0.67–40 Hz · per-window z-score.

---

## Evaluation Metrics

All metrics computed on held-out test subjects per fold. Reported as **mean ± 95% CI**.

| Metric | Description | Direction | Implementation |
|--------|-------------|-----------|----------------|
| **PRD** | % Root Mean Square Difference | ↓ lower | `compute_prd()` |
| **Pearson r** | Waveform correlation | ↑ higher | `compute_pearson_r()` |
| **BCE** | Consistency in CLEF clinical feature space | ↓ lower | `CLEFClassifier` + `compute_bce()` |
| **EMD** | Wasserstein-1 on RR interval distributions | ↓ lower | `compute_emd()` |
| **KS stat** | KS test D-statistic on RR interval CDFs | ↓ lower | `compute_ks()` |
| **Beat MAE** | Mean absolute R-peak timing error (s) | ↓ lower | `compute_beat_timing_mae()` |

**BCE note:** Uses `CLEFClassifier` — the frozen CLEF encoder maps each ECG window to 256-dim
clinical feature scores (sigmoid-activated). BCE between real and reconstructed scores measures
diagnostic consistency in CLEF's clinically-supervised feature space. No PTB-XL data needed.

`CLEFClassifier` requires **10 s input** (it resamples to 5000 samples at 500 Hz internally).
For 4 s models (original/MSE/Huber), BCE is evaluated on a **separate 10 s test DataLoader**
built from the same held-out subjects — the BiLSTM is sequence-length agnostic so the trained
model runs on 10 s PPG windows at eval time. PRD, Pearson r, EMD, KS, and Beat MAE are still
evaluated on 4 s windows consistent with training.

R-peaks detected with `neurokit2.ecg_peaks()` (Pan-Tompkins). Windows with <3 peaks skipped.

---

## Cross-Validation Design

**8-fold group CV** — `KFold(n_splits=8, shuffle=True, random_state=42)` at subject level:

- ~46 training subjects, ~7 test subjects per fold (~840 windows at 4 s / ~665 windows at 10 s)
- Fresh `ReHeartNet` per fold — no leakage between folds
- Phase alignment (PPG→ECG cross-correlation) applied to **training windows only**
- Fold assignments saved to `results/fold_assignments.json` (reproducible)

---

## Configuration

[core/config.py](core/config.py) — parameters confirmed in Lee et al. vs. our defaults:

```python
# ── Confirmed in Lee et al. (2026) ────────────────────────────────────────
FS              = 125       # BIDMC native sampling rate (Hz)
NUM_BLOCKS      = 5         # 5 stacked DC-BiLSTM blocks

# ── Not in paper (supplementary only) — our defaults, all tuned by Optuna ─
SEQ_LEN         = 1250      # 10 s × 125 Hz
HIDDEN_SIZE     = 64        # BiLSTM hidden units per direction  [Optuna: 32/64/128]
BATCH_SIZE      = 64        #                                    [Optuna: 32/64/128]
LEARNING_RATE   = 1e-3      # Adam initial lr                    [Optuna: 1e-4–1e-2]
EPOCHS          = 100       # max epochs (early stopping patience=15)

# ── Our contributions (not in original paper) ─────────────────────────────
LAMBDA_CLINICAL = 0.1       # CLEF perceptual loss weight        [Optuna: 1e-3–1.0]
HUBER_DELTA     = 1.0       # Huber loss δ                       [Optuna: 0.1–2.0]
CLEF_MODEL_SIZE = "small"   # "small" (256-dim) | "medium" (1024-dim)
CLEF_CHECKPOINT_DIR = "models/clef"
```

---

## Data

**BIDMC** (Beth Israel Deaconess Medical Center, PhysioNet) — 53 ICU subjects,
simultaneous ECG + PPG at 125 Hz, ~8 minutes each.

**Preprocessing pipeline** (`src/preprocessing.py`):
1. Load Lead II (ECG) and PLETH (PPG) by **channel name** — order varies per subject
2. Z-score normalise over the full recording
3. Optionally apply FIR bandpass filter: ECG 0.5–55 Hz, PPG 0.5–10 Hz (Lee et al. original only)
4. Segment into fixed-length windows:
   - **Original / our MSE / our Huber**: 4 s (500 samples), no overlap, FIR bandpass → **~120 windows/subject**
   - **Our CLEF model**: 10 s (1250 samples), 50% overlap, z-score only → **~95 windows/subject**
5. Phase-align PPG to ECG via cross-correlation (training windows only)

---

## References

- **ReHeartNet**: Lee et al. (2026) — *ReHeartNet: Reconstruct ECG From PPG by Using Dense Connected Deep Learning Model*, IEEE Open Journal of Engineering in Medicine and Biology, [doi:10.1109/OJEMB.2026.3670010](https://doi.org/10.1109/OJEMB.2026.3670010)
- **CLEF**: Nokia Bell Labs — [GitHub](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) · [Zenodo weights](https://zenodo.org/records/17572734) · arXiv:2512.02180
- **BIDMC dataset**: Pimentel et al. (2016), PhysioNet, [doi:10.13026/C2208R](https://doi.org/10.13026/C2208R)
- **Optuna**: Akiba et al. (2019), KDD, [doi:10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)
- **NeuroKit2**: Makowski et al. (2021), *Behavior Research Methods*
