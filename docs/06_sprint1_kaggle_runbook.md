# Sprint 1 Runbook (Kaggle Notebooks)

This guide shows how to run **both repositories** in this workspace on **Kaggle** and complete Sprint 1 deliverables:

- **Deliverable A**: baseline rerun (reproducible training run + recorded artifacts)
- **Deliverable B**: HR_plus (x4) outputs + manifest CSV

It is written to be copy/paste friendly in a Kaggle Notebook.

---

## 0) Kaggle-specific constraints and conventions

Kaggle storage conventions:

- Read-only inputs: `/kaggle/input/...`
- Writable work dir: `/kaggle/working/...`
- Final artifacts: anything left under `/kaggle/working` can be saved by using **Save Version** (Kaggle will snapshot notebook outputs).

GPU:

- Notebook settings → Accelerator = **GPU** (recommended)

Internet:

- If you need to `pip install` packages not already present, enable Internet in Notebook settings.

---

## 1) Get the code into Kaggle

### Option A (recommended): Add this repo as a Kaggle Dataset

1) Create a Kaggle Dataset from your repo folder (including both `baseline_icpr_2026/` and `ICPR_2026/`).
2) Attach it to your Kaggle Notebook.
3) It will appear at something like:

- `/kaggle/input/icpr2026-repo/`

Then copy it into the writable working directory:

```bash
!ls -la /kaggle/input

REPO_IN="/kaggle/input/icpr2026-repo"   # change to your dataset folder name
REPO_OUT="/kaggle/working/ICPR2026"

!mkdir -p "$REPO_OUT"
!cp -R "$REPO_IN"/* "$REPO_OUT"/
!ls -la "$REPO_OUT"
```

### Option B: `git clone` (requires Internet)

```bash
%cd /kaggle/working
!git clone <YOUR_REPO_URL> ICPR2026
%cd /kaggle/working/ICPR2026
```

---

## 2) Provide the dataset (`data.zip`) to Kaggle

You need `data.zip` as a Kaggle Dataset.

Recommended:

1) Create a Kaggle Dataset containing `data.zip`.
2) Attach it to the notebook.
3) You will see it under `/kaggle/input/<your-data-dataset>/data.zip`.

Example:

```bash
DATASET_IN="/kaggle/input/icpr2026-data"  # change this
!ls -la "$DATASET_IN"
```

---

## 3) Install dependencies

Kaggle often has many ML packages preinstalled, but to keep things reproducible, install the baseline requirements.

```bash
%cd /kaggle/working/ICPR2026

pip -q install -r baseline_icpr_2026/requirements.txt

# Headless OpenCV avoids GUI bindings issues
pip -q install -U opencv-python-headless
```

Sanity check:

```python
import torch, cv2, albumentations
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cv2:', cv2.__version__)
```

---

## 4) Extract the dataset into the expected layout

The baseline expects:

- `baseline_icpr_2026/data/train/...`

Use Kaggle paths.

```bash
%cd /kaggle/working/ICPR2026

DATA_ZIP="/kaggle/input/icpr2026-data/data.zip"   # change this
OUT_DIR="baseline_icpr_2026/data"

!mkdir -p "$OUT_DIR"

# Extract only once
if [ ! -d "$OUT_DIR/train" ]; then
  unzip -q "$DATA_ZIP" -d "$OUT_DIR"
fi

!ls -la baseline_icpr_2026/data/train | head
```

You should see `Scenario-A` and `Scenario-B`.

---

## 5) Create a Sprint-1 run folder (Kaggle working)

Recommended convention:

- `/kaggle/working/runs/sprint1_2026-01-14/`

```bash
RUNS_ROOT="/kaggle/working/runs/sprint1_2026-01-14"
!mkdir -p "$RUNS_ROOT/baseline_rerun"
!mkdir -p "$RUNS_ROOT/hr_plus"
!ls -la "$RUNS_ROOT"
```

---

## 6) Deliverable A — Baseline rerun (recorded)

Run:

```bash
%cd /kaggle/working/ICPR2026/baseline_icpr_2026

python train.py \
  --data-root data/train \
  --output-dir "/kaggle/working/runs/sprint1_2026-01-14/baseline_rerun" \
  --epochs 50 \
  --batch-size 64 \
  --split-ratio 0.8
```

Notes:

- If you hit out-of-memory, reduce `--batch-size`.
- If dataloader is slow, try `--num-workers 2` (or 0 if you see worker issues).

Verify artifacts:

```bash
!ls -la "/kaggle/working/runs/sprint1_2026-01-14/baseline_rerun"
```

You should see:

- `best_model.pth`
- `metrics.csv`
- `run_config.json`
- `val_tracks.json`

---

## 7) Deliverable B — HR_plus (x4) outputs

Generate HR_plus x4 from LR frames.

Run two methods:

```bash
%cd /kaggle/working/ICPR2026/ICPR_2026

python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ../baseline_icpr_2026/data/train \
  --output-root "/kaggle/working/runs/sprint1_2026-01-14/hr_plus" \
  --method lanczos2 \
  --scale 4

python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ../baseline_icpr_2026/data/train \
  --output-root "/kaggle/working/runs/sprint1_2026-01-14/hr_plus" \
  --method denoise_clahe_sharpen \
  --scale 4
```

Expected outputs:

- images under `.../hr_plus/x4/<method>/...`
- `manifest_hr_plus_x4_<method>.csv`
- `run_hr_plus_x4_<method>.json`

---

## 8) Quick analysis (recommended)

### 8.1 Plot baseline curves

```python
import pandas as pd
import matplotlib.pyplot as plt

metrics_path = "/kaggle/working/runs/sprint1_2026-01-14/baseline_rerun/metrics.csv"
df = pd.read_csv(metrics_path)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(df["epoch"], df["train_loss"], label="train_loss")
ax[0].plot(df["epoch"], df["val_loss"], label="val_loss")
ax[0].set_title("Loss")
ax[0].legend()

ax[1].plot(df["epoch"], df["val_acc"], label="val_acc")
ax[1].set_title("Val exact-match (%)")
ax[1].legend()

plt.show()
```

### 8.2 Compare HR_plus methods by PSNR proxy

```python
import pandas as pd

root = "/kaggle/working/runs/sprint1_2026-01-14/hr_plus"

for method in ["lanczos2", "denoise_clahe_sharpen"]:
    p = f"{root}/manifest_hr_plus_x4_{method}.csv"
    df = pd.read_csv(p)
    df["psnr_to_hr_resized_db"] = pd.to_numeric(df["psnr_to_hr_resized_db"], errors="coerce")
    print("\n===", method, "===")
    print(df["psnr_to_hr_resized_db"].describe())

    df["scenario"] = df["track_dir"].str.extract(r"(Scenario-[AB])", expand=False)
    print(df.groupby(["scenario", "plate_layout"])['psnr_to_hr_resized_db'].mean())
```

---

## 9) Save artifacts from Kaggle

Kaggle will snapshot `/kaggle/working` when you **Save Version**.

To keep the output clean:

- Ensure everything you need is under `/kaggle/working/runs/sprint1_2026-01-14/`

Optionally, zip the run folder:

```bash
%cd /kaggle/working
!zip -r sprint1_2026-01-14_runs.zip runs/sprint1_2026-01-14
!ls -la sprint1_2026-01-14_runs.zip
```

Then you can download `sprint1_2026-01-14_runs.zip` from the Kaggle output panel.

---

## 10) Checklist (Sprint 1)

- [ ] Dataset extracted into `baseline_icpr_2026/data/train`
- [ ] Baseline rerun completed; artifacts under `/kaggle/working/runs/.../baseline_rerun/`
- [ ] HR_plus x4 generated for at least 2 methods; manifests saved
- [ ] Quick plots / breakdown run
- [ ] Next-sprint hypothesis written down

---

## Appendix: troubleshooting

### A) `pip install` fails due to no Internet

- Enable Internet in Notebook settings, or
- Use Kaggle’s preinstalled packages if they already satisfy versions.

### B) OOM on GPU

- Reduce `--batch-size`
- Reduce `--epochs` for quick debugging

### C) Validation split changes

The baseline stores `val_tracks.json` inside the run output directory. Reuse the same `--output-dir` to keep the exact same split.
