# Code Walkthrough (baseline_icpr_2026 + ICPR_2026)

Date: 2026-01-14

This document walks through the two codebases found in this workspace:

- `baseline_icpr_2026/`: a **multi-frame CRNN** baseline for Low-Resolution License Plate Recognition (LRLPR).
- `ICPR_2026/`: a toolbox of **degradation modelling** and **handcrafted enhancement** utilities intended to support robust LRLPR.

---

## 1) baseline_icpr_2026

### 1.1 Goal

Implement a practical baseline for the competition task: predict the license plate text for each **track** using **only LR frames**.

Key ideas:

- Each sample is a **track**: 5 consecutive frames.
- Training can optionally use HR frames (training set only) by **degrading HR** to mimic LR.
- Recognition model is **CRNN + CTC** (no explicit character segmentation required).

### 1.2 Repo structure

- `config.py`: hyperparameters + character set.
- `dataset.py`: track scanning, reproducible train/val split, frame loading.
- `transforms.py`: augmentations for training + synthetic degradations.
- `models/crnn.py`: CNN backbone + attention fusion + BiLSTM + classifier.
- `models/fusion.py`: attention-based temporal fusion across frames.
- `utils.py`: seeding + greedy CTC decoding.
- `train.py`: training loop with AMP, OneCycleLR; saves best model.

### 1.3 Configuration (`config.py`)

Main config fields:

- `DATA_ROOT`: root containing `track_*` folders (recursively discovered).
- `IMG_HEIGHT`, `IMG_WIDTH`: images are resized to `(32, 128)`.
- `CHARS`: `"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"`
- CTC mapping:
  - blank index = 0
  - characters are indexed from 1..N
  - `NUM_CLASSES = len(CHARS) + 1`

Notes:

- The character set is constrained; any unseen characters are dropped during label encoding.

### 1.4 Data pipeline (`dataset.py`)

#### Track discovery

`AdvancedMultiFrameDataset` recursively searches for `**/track_*` directories.

A track is usable if:

- `annotations.json` exists
- At least one LR frame exists: `lr-*.png|jpg`
- An annotation contains `plate_text` (fallbacks: `license_plate` or `text`)

#### Train/val split

- Split is performed at the **track** level.
- Split IDs are persisted to JSON: `Config.VAL_SPLIT_FILE`.
- Determinism uses `Config.SEED`.

This ensures your validation set does not change across reruns (as long as data root is consistent).

#### Multi-frame loading

- Each sample always returns **exactly 5 frames**.
- If fewer than 5 available, last frame is repeated.
- If more than 5, first 5 are used.

#### HR usage during training

When `mode == 'train'`, there is a 50% chance to:

- use HR frames (if present)
- apply synthetic degradation (`get_degradation_transforms`) to HR

This is a simple but effective way to exploit HR-only training information.

#### Collate

CTC requires concatenated targets:

- images: `[B, 5, C, H, W]`
- targets: concatenated 1D tensor of all labels in batch
- target_lengths: length of each label sequence

### 1.5 Augmentations (`transforms.py`)

Two transform pipelines:

- `get_train_transforms()`
  - resize to `(32, 128)`
  - affine + perspective jitter
  - brightness/contrast, HSV
  - coarse dropout
  - normalize + tensor

- `get_degradation_transforms()`
  - blur (Gaussian/Motion/Defocus)
  - noise (Gaussian/ISO/Multiplicative)
  - JPEG compression artifacts
  - downscale

Important: the degradation pipeline is **generic** (not learned from the dataset).

### 1.6 Model (`models/crnn.py` + `models/fusion.py`)

#### High-level architecture

```
Input  : x  [B, 5, 3, 32, 128]
Frames : reshape -> [B*5, 3, 32, 128]
CNN    : -> [B*5, 512, 1, W]
Fusion : attention-weighted sum across 5 frames -> [B, 512, W]
RNN    : BiLSTM over time(W) -> [B, W, 2*hidden]
Head   : Linear -> [B, W, num_classes]
CTC    : loss over variable-length label
```

#### CNN backbone

A VGG-like stack that aggressively reduces height to 1 while preserving width as time dimension.

#### AttentionFusion

`AttentionFusion` computes per-frame scores and performs a softmax-weighted sum.

- input: `[B*5, C, 1, W]`
- scores: `[B, 5, 1, W]`
- output: `[B, C, W]`

This fuses the five frames into one per-track feature sequence.

### 1.7 Training (`train.py`)

The training script does:

- seed fixing (`seed_everything`)
- dataset creation for train/val
- `CTCLoss(blank=0, zero_infinity=True)`
- optimizer: `AdamW`
- scheduler: `OneCycleLR`
- mixed precision training (enabled only on CUDA)

Artifacts:

- `best_model.pth` saved into `--output-dir` (or current directory if not specified)
- `metrics.csv` written per epoch
- `run_config.json` records CLI overrides + run metadata
- `val_tracks.json` persisted for reproducible split

Decoding:

- `decode_predictions()` implements greedy CTC decode:
  - collapse repeats
  - remove blanks

### 1.8 How to run baseline (recommended)

1) Extract dataset so the baseline sees `baseline_icpr_2026/data/train/...`

2) Run training:

```bash
cd baseline_icpr_2026
python train.py \
  --data-root data/train \
  --output-dir ..\\runs\\sprint1\\baseline_rerun \
  --epochs 50 \
  --batch-size 64 \
  --split-ratio 0.8
```

---

## 2) ICPR_2026

### 2.1 Goal

This repo is not a full recognition model. It focuses on supporting components that can improve LRLPR pipelines:

- **Degradation modelling**: estimate a realistic transformation mapping HR → LR, learned from paired HR/LR frames.
- **Handcrafted enhancement**: simple upscalers / filters to create a stronger input for OCR/recognition.

The intention is to support strategies highlighted in the competition brief: *super-resolution, temporal modeling, robust OCR*.

### 2.2 Repo structure

Key folders:

- `src/degradation_modelling/`
  - `run.py`: CLI entrypoint
  - `runner.py`: main pipeline
  - `config.py`: search spaces + settings
  - `degradations.py`: operators and parameter dataclasses
  - `losses.py`: LR matching loss
  - `search.py`: per-pair greedy search + LR synthesis
  - `global_search.py`: fit ONE global set of params over many pairs

- `src/handcrafted_modelling/`
  - `base.py`: method registry (upscalers + enhancement filters)
  - `model.py`: `LicensePlateEnhancer` (wrapper)
  - `export_hr_plus.py`: CLI to generate HR_plus outputs at x4 (added for Sprint 1)
  - `method.py`: currently empty

- `src/utility/visualizer.py`: tiny helper for showing images (had a missing import; now fixed).

### 2.3 Degradation modelling pipeline

The degradation modeller assumes you have paired images in each track:

- `hr-001.jpg` ... `hr-005.jpg`
- `lr-001.jpg` ... `lr-005.jpg`

It learns parameters of a synthetic degradation process that makes HR look like LR.

#### Operators (in order)

From `degradations.py`, the process is:

1) lighting adjustment (alpha/beta/gamma)
2) motion blur (length, angle)
3) gaussian blur (sigma)
4) scale-down (scale + interpolation)
5) gaussian noise (std)
6) final **bicubic resize** to match LR spatial size exactly

#### Loss

From `losses.py`:

- pixel L1 loss on normalized image
- gradient-magnitude L1 loss (Sobel), optionally grayscale

Total:

$\text{loss} = w_{l1} \cdot \text{L1} + w_{grad} \cdot \text{L1}(\nabla)$

Defaults:

- `l1 = 1.0`
- `grad_l1 = 0.25`

#### Search

Two modes:

- **Global fit** (default): fit ONE params set on many (HR,LR) pairs, then apply.
- **Per-frame fit**: fit separately for each pair (slower, usually not needed).

The optimizer is **greedy coordinate search** over discrete search spaces (`config.py`).

#### Outputs

For each LR frame file `lr-XYZ.jpg`, output is written next to it as:

- `lr-XYZ-downsample.png`

And parameters are persisted to:

- per-track: `degradation_modelling_params.json`
- global: `degradation_modelling_global_params.json` (stored under `data_root`)

### 2.4 How to run degradation modelling

Example (Windows-style paths):

```bash
cd ICPR_2026
python -m src.degradation_modelling.run \
  --data-root ..\\baseline_icpr_2026\\data\\train \
  --global-fit \
  --max-pairs-for-fit 5000 \
  --fit-seed 42
```

This is useful for:

- learning **domain-specific** degradation parameters
- generating synthetic LR for augmentation
- understanding the LR formation characteristics of each scenario/layout

### 2.5 Handcrafted modelling

`src/handcrafted_modelling/base.py` defines a method registry mapping strings to callable upscaler/enhancer classes, e.g.:

- `nearest`, `bilinear`, `bicubic`, `lanczos`, `lanczos2`
- `denoise_clahe_sharpen`
- `edge_enhance`
- `freq_enhance`

Notes:

- Some methods are placeholders (e.g. `bilateral_adaptive`, `multiscale` currently behave like Lanczos).
- `LicensePlateEnhancer.enhance()` now returns the enhanced image (was previously missing a return).

### 2.6 How to generate HR_plus outputs (x4)

The new CLI `src/handcrafted_modelling/export_hr_plus.py` will:

- walk all `track_*` directories
- read LR frames
- apply the selected method at scale=4
- save PNG outputs into a separate `output_root`
- write a manifest CSV you can analyze later

Example:

```bash
cd ICPR_2026
python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ..\\baseline_icpr_2026\\data\\train \
  --output-root ..\\runs\\sprint1\\hr_plus \
  --method denoise_clahe_sharpen \
  --scale 4
```

---

## 3) Practical integration points

Even though the repos target different layers of the pipeline, they can be combined:

- Use `ICPR_2026` degradation modelling to generate **realistic synthetic LR** from HR, then train `baseline_icpr_2026` on those instead of generic albumentations degradations.
- Use `ICPR_2026` HR_plus outputs as a preprocessing step before recognition/OCR.

The comparison document (next file) formalizes the tradeoffs and integration options.
