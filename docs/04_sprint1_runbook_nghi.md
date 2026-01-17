# Sprint 1 Runbook (Nghi): Baseline rerun + HR_plus (x4) outputs

Date: 2026-01-14

This runbook follows the Sprint-1 note in `sprint1_plan`:

- Nghi: **baseline rerun + HR_plus output (x4 on resolution)**

The goal is not only to generate outputs, but to **record artifacts** so we can compare methods and plan the next sprint based on evidence.

---

## 0) Sprint 1 deliverables

### Deliverable A — Baseline rerun (reproducible)

Artifacts to produce:

- `best_model.pth`
- `metrics.csv` (train_loss/val_loss/val_acc per epoch)
- `run_config.json` (data_root, hyperparams, split_ratio, seed)
- `val_tracks.json` (exact val split IDs)

### Deliverable B — HR_plus (x4) image outputs

Artifacts to produce:

- a directory of generated images (HR_plus) for LR frames
- a CSV manifest describing what was generated + simple PSNR-to-HR proxy

---

## 1) Workspace conventions (recommended)

Create a run folder per sprint date:

```
runs/
  sprint1_2026-01-14/
    baseline_rerun/
    hr_plus/
      x4/lanczos2/...
      x4/denoise_clahe_sharpen/...
```

Reason:

- You want results to be comparable without overwriting.

---

## 2) Prepare data (one-time)

### 2.1 Extract `data.zip`

The baseline expects `baseline_icpr_2026/data/train/...` by default.

Run from the workspace root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_data.ps1
```

This should create:

- `baseline_icpr_2026/data/train/Scenario-A/...`
- `baseline_icpr_2026/data/train/Scenario-B/...`

### 2.2 Quick sanity check

Confirm you can see track folders:

- `baseline_icpr_2026/data/train/Scenario-B/Mercosur/track_XXXXX/`

Each track should contain `lr-001..lr-005` and `annotations.json`.

---

## 3) Deliverable A: Baseline rerun

### 3.1 Why rerun baseline

This gives a reproducible “starting point” and baseline curves. It also establishes:

- your train/val split (fixed)
- your best achievable exact-match rate without HR_plus

### 3.2 Run baseline training (recorded)

Run from the workspace root (or from `baseline_icpr_2026/`):

```powershell
cd baseline_icpr_2026
python train.py \
  --data-root data/train \
  --output-dir ..\\runs\\sprint1_2026-01-14\\baseline_rerun \
  --epochs 50 \
  --batch-size 64 \
  --split-ratio 0.8
```

Notes:

- `val_tracks.json` is stored inside the output dir so reruns are comparable.
- If you don’t have a GPU, training will run without AMP.

### 3.3 What to record

In `runs/sprint1_2026-01-14/baseline_rerun/`, verify you have:

- `run_config.json`
- `metrics.csv`
- `val_tracks.json`
- `best_model.pth`

### 3.4 Quick analysis checklist (baseline)

- Plot `val_acc` vs epoch
- Confirm `val_acc` plateaus (or diverges)
- If overfitting:
  - increase augmentation
  - reduce epochs
  - adjust LR schedule

---

## 4) Deliverable B: HR_plus (x4) outputs

### 4.1 What HR_plus means here

“HR_plus (x4)” means you generate an enhanced/upscaled image from each LR frame.

For Sprint 1 we use **handcrafted** methods (fast to run, easy to compare). Later sprints can try diffusion/UNet SR.

### 4.2 Generate HR_plus x4 images

Two suggested methods to compare:

1) `lanczos2` (strong interpolation baseline)
2) `denoise_clahe_sharpen` (denoise + contrast enhancement + sharpening)

Run:

```powershell
cd ICPR_2026
python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ..\\baseline_icpr_2026\\data\\train \
  --output-root ..\\runs\\sprint1_2026-01-14\\hr_plus \
  --method lanczos2 \
  --scale 4

python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ..\\baseline_icpr_2026\\data\\train \
  --output-root ..\\runs\\sprint1_2026-01-14\\hr_plus \
  --method denoise_clahe_sharpen \
  --scale 4
```

Outputs:

- images under `runs/.../hr_plus/x4/<method>/...`
- manifest CSV per method:
  - `manifest_hr_plus_x4_<method>.csv`

The manifest includes a proxy image-quality metric:

- PSNR between generated output and HR resized to output size (only where HR exists).

### 4.3 What to record

Keep:

- the manifest CSV
- the JSON run metadata
- the output images

These are the raw materials for “output analysis” that informs Sprint 2.

---

## 5) Output analysis (how to decide next improvements)

### 5.1 Image-level analysis

From the manifest CSV:

- Compute PSNR distribution per method.
- Slice by:
  - Scenario A vs Scenario B
  - Brazilian vs Mercosur

Goal:

- identify where a method helps/hurts.

### 5.2 Recognition-level analysis (most important)

Image quality improvements do not always improve OCR/recognition.

For Sprint 1 you can do one of these:

- (Fast) Pick ~100 tracks and run a chosen OCR engine on LR vs HR_plus; compare exact-match.
- (More consistent) Adapt baseline dataset to load HR_plus instead of LR and measure validation exact-match.

If you want, I can implement the “HR_plus-as-input” dataset switch in `baseline_icpr_2026` so the baseline can be evaluated on HR_plus directly.

### 5.3 Record decisions

Write a short note per method:

- what it visually changes (blur removal? contrast? ringing?)
- where it fails (over-sharpen, halos)
- whether it helps exact-match accuracy

This note becomes the Sprint 2 design input.

---

## 6) Sprint 1 checklist

- [ ] Data extracted into `baseline_icpr_2026/data/train`
- [ ] Baseline rerun completed; artifacts saved under `runs/.../baseline_rerun/`
- [ ] HR_plus x4 generated for at least 2 methods; manifests saved
- [ ] Quick analysis completed (plots + per-scenario breakdown)
- [ ] Next-sprint hypothesis written down
