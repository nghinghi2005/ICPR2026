# Repo Comparison: baseline_icpr_2026 vs ICPR_2026

Date: 2026-01-14

This document compares the two repos in the workspace and provides a detailed analysis of how they differ, what each does well, and how they can be combined into a stronger competition pipeline.

---

## 1) Executive summary

- `baseline_icpr_2026` is a **complete recognition baseline**: it trains a multi-frame CRNN and outputs plate text predictions.
- `ICPR_2026` is a **pipeline toolbox**: it focuses on **domain-specific degradation modelling** (HR→LR) and **handcrafted enhancement** (HR_plus), not on end-to-end recognition.

If your goal is to submit a working solution quickly: start from `baseline_icpr_2026`.

If your goal is to improve performance beyond generic augmentation: incorporate `ICPR_2026`’s degradation modelling and/or HR_plus preprocessing.

---

## 2) Goals & scope

### baseline_icpr_2026

- Primary output: **plate text**.
- Learns a discriminative model directly for recognition.
- Uses HR frames only indirectly (degrade HR half the time during training).

### ICPR_2026

- Primary output: **processed images** (e.g. downsampled LR-fake, enhanced HR_plus), and **fitted degradation parameters**.
- Helps build better data augmentation and preprocessing stages.
- Intended to be paired with a downstream recognizer (baseline CRNN, OCR engine, or an ensemble).

---

## 3) Data assumptions & compatibility

### Shared assumptions

Both repos assume the competition dataset layout:

- Each `track_*` contains:
  - `lr-001`..`lr-005` (required)
  - `hr-001`..`hr-005` (training-only)
  - `annotations.json` containing `plate_text`

### Differences

- `baseline_icpr_2026` loads either LR or HR-with-degradation **per sample**.
- `ICPR_2026/degradation_modelling` requires paired HR/LR frames (same indices) to fit parameters.

Practical note:

- In the training dataset (your `data.zip`), most tracks appear to have both LR and HR frames, so the two repos are directly compatible for experimentation.

---

## 4) Core methodology differences

### 4.1 Recognition model vs image-process model

- Baseline uses a CRNN with a temporal fusion module and CTC loss.
- ICPR_2026 models the *image formation* process and provides handcrafted enhancement, but has no recognizer.

Implication:

- Baseline improvements tend to come from better training (architecture, decoding, loss) or data.
- ICPR improvements come from better upstream images / more realistic augmentation.

### 4.2 Synthetic degradation strategy

#### baseline_icpr_2026

- Uses `albumentations` “generic” degradation transforms.
- Pros:
  - easy, fast, broad augmentation
  - no need for HR/LR pairing
- Cons:
  - can be mismatched to real LR domain
  - random degradations may harm learning if unrealistic

#### ICPR_2026

- Fits a **domain-specific** degradation pipeline using paired HR/LR.
- Pros:
  - grounded in actual data
  - yields interpretable params (lighting, blur, scale-down, noise)
- Cons:
  - search is discrete and greedy (may miss best params)
  - requires paired HR/LR and is compute-heavy for large fits

---

## 5) Engineering maturity & reliability

### baseline_icpr_2026

- Training script is straightforward.
- After adjustments, supports:
  - output directory
  - recorded metrics
  - CUDA AMP only when available

Potential pain points:

- No dedicated inference/submission script.
- CTC decoding is greedy only (no beam search).

### ICPR_2026

- Degradation modelling is modular and clean:
  - `dataclasses`, explicit config, optional tqdm
  - avoids importing `cv2` in config

Potential pain points:

- `handcrafted_modelling/method.py` is empty (likely WIP).
- Some handcrafted methods are placeholders.
- `utility/visualizer.py` needed a missing import (fixed).

---

## 6) Where each repo is strong

### baseline_icpr_2026 strengths

- End-to-end recognizer + training loop.
- Multi-frame input with attention fusion.
- Easy to rerun and iterate.

### ICPR_2026 strengths

- Explicit modelling of domain degradations.
- Produces reusable artifacts:
  - global params JSON
  - per-track params JSON
  - generated LR-fake images
- Provides a rapid HR_plus x4 generator for preprocessing experiments.

---

## 7) High-value integration options (recommended)

### Option A (Sprint-1 friendly): LR → HR_plus (x4) → baseline recognizer

1) Generate HR_plus images for LR frames (x4) using a chosen method.
2) Feed HR_plus images into recognizer (requires adapting baseline dataset to load HR_plus paths).

Pros:

- fast to test multiple enhancement strategies
- fits Sprint 1 deliverable (“HR_plus output”)

Cons:

- no guarantee the enhancement helps recognition
- may amplify noise/artifacts

### Option B: Replace generic degradation with fitted degradation (HR → LR_fitted)

1) Fit global degradation params via `ICPR_2026/src/degradation_modelling`.
2) Apply those params to HR to generate realistic LR-fake.
3) Train baseline using a mix:
   - real LR
   - fitted LR-fake

Pros:

- augmentation becomes “in-domain”
- directly uses HR information in a principled way

Cons:

- needs some data plumbing changes in baseline dataset

### Option C: Hybrid ensemble

- Baseline CRNN prediction
- OCR engine prediction (e.g., PaddleOCR/Tesseract) on HR_plus
- Combine via confidence-weighted voting

Pros:

- robust “systems” approach

Cons:

- more engineering + calibration work

---

## 8) What to improve next (after Sprint 1)

Baseline-side:

- Add inference + submission writer (track_id,text;confidence).
- Consider beam search CTC decoding.
- Add per-layout conditioning (Brazilian vs Mercosur).

ICPR-side:

- Improve degradation search space and/or switch to continuous optimization.
- Replace placeholder handcrafted methods with real ones.
- Evaluate HR_plus methods by downstream recognition gain, not only image quality metrics.

---

## 9) Concrete “Sprint 1” recommendation

Given your sprint goal (baseline rerun + HR_plus x4):

1) Rerun baseline training with fixed outputs/metrics saved.
2) Generate HR_plus x4 outputs for at least two methods:
   - `lanczos2`
   - `denoise_clahe_sharpen`
3) Record:
   - training curves (`metrics.csv`)
   - `best_model.pth`
   - HR_plus manifest CSV
4) Use the output analysis to decide Sprint 2 direction.

The next doc explains the competition and your dataset in detail.
