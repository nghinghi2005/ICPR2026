# Sprint 1 Runbook (Google Colab)

This guide shows how to run **both repositories** in this workspace on **Google Colab** and complete Sprint 1 deliverables:

- **Deliverable A**: baseline rerun (reproducible training run + recorded artifacts)
- **Deliverable B**: HR_plus (x4) outputs + manifest CSV

It is written to be copy/paste friendly in a Colab notebook.

---

## 0) What you will produce (Sprint 1 artifacts)

After you finish this runbook, you should have:

### A) Baseline rerun artifacts

In a run folder you choose (recommended on Google Drive):

- `best_model.pth`
- `metrics.csv` (train_loss / val_loss / val_acc per epoch)
- `run_config.json`
- `val_tracks.json` (the exact val split track IDs)

### B) HR_plus artifacts

In a run folder you choose (recommended on Google Drive):

- generated images under `hr_plus/x4/<method>/...`
- `manifest_hr_plus_x4_<method>.csv`
- `run_hr_plus_x4_<method>.json`

---

## 1) Colab setup (runtime + Drive)

### 1.1 Select GPU runtime

In Colab:

- `Runtime` → `Change runtime type` → `Hardware accelerator` = **GPU**

Training will work on CPU, but it will be much slower.

### 1.2 Mount Google Drive

Use Drive to persist outputs and avoid losing artifacts when the Colab session ends.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Choose a working folder on Drive, e.g.:

- `/content/drive/MyDrive/ICPR2026/`

---

## 2) Get the code into Colab

You have two options.

### Option A (recommended): upload / clone the repo

If this repo is on GitHub (or internal Git), clone it:

```bash
%cd /content
# Replace with your repo URL
git clone <YOUR_REPO_URL> ICPR2026
%cd /content/ICPR2026
```

### Option B: upload a zip of the workspace

- Upload a zip containing the workspace (both `baseline_icpr_2026/` and `ICPR_2026/`).
- Unzip into `/content/ICPR2026`.

---

## 3) Provide the dataset to Colab

You need `data.zip` (the competition training dataset archive).

### Option A (recommended): put `data.zip` on Drive

Upload `data.zip` to:

- `/content/drive/MyDrive/ICPR2026/data.zip`

### Option B: upload `data.zip` to the Colab session

This is not persistent (you must re-upload if the session resets).

---

## 4) Install dependencies

Colab usually includes PyTorch, but versions can vary. This project needs:

- `torch`, `torchvision`
- `albumentations`, `opencv-python`, `Pillow`, `numpy`, `tqdm`

Run:

```bash
%cd /content/ICPR2026

# Baseline dependencies
pip -q install -r baseline_icpr_2026/requirements.txt

# Colab note: headless OpenCV often avoids GUI issues
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

## 5) Extract the dataset into the expected layout

The baseline expects:

- `baseline_icpr_2026/data/train/...`

In Colab (Linux), don’t use the PowerShell script. Use `unzip` instead.

### 5.1 Choose paths

Set:

- `REPO_ROOT = /content/ICPR2026`
- `DATA_ZIP = /content/drive/MyDrive/ICPR2026/data.zip` (or wherever you placed it)

### 5.2 Extract

```bash
%cd /content/ICPR2026

# Adjust this if your data.zip is in a different location
DATA_ZIP="/content/drive/MyDrive/ICPR2026/data.zip"
OUT_DIR="baseline_icpr_2026/data"

mkdir -p "$OUT_DIR"

# Extract only if train/ doesn't exist yet
if [ ! -d "$OUT_DIR/train" ]; then
  unzip -q "$DATA_ZIP" -d "$OUT_DIR"
fi

# Quick check
ls -la baseline_icpr_2026/data/train | head
```

You should see `Scenario-A` and `Scenario-B`.

---

## 6) Create a Sprint-1 run folder (on Drive)

Recommended convention (Drive):

- `/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/`

Create:

```bash
RUNS_ROOT="/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14"
mkdir -p "$RUNS_ROOT/baseline_rerun"
mkdir -p "$RUNS_ROOT/hr_plus"
```

---

## 7) Deliverable A — Baseline rerun (recorded)

### 7.1 Run training

Run from `baseline_icpr_2026/` so imports are consistent.

```bash
%cd /content/ICPR2026/baseline_icpr_2026

python train.py \
  --data-root data/train \
  --output-dir "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/baseline_rerun" \
  --epochs 50 \
  --batch-size 64 \
  --split-ratio 0.8
```

Tips:

- If you hit out-of-memory, reduce `--batch-size` (e.g. 32 or 16).
- If dataloader is slow, set `--num-workers 2` (or 0 if you see hangs).

### 7.2 Verify artifacts

```bash
ls -la "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/baseline_rerun"
```

You should see:

- `best_model.pth`
- `metrics.csv`
- `run_config.json`
- `val_tracks.json`

---

## 8) Deliverable B — HR_plus (x4) image outputs

You will generate HR_plus from LR frames using handcrafted methods.

Two recommended methods:

1) `lanczos2`
2) `denoise_clahe_sharpen`

### 8.1 Run HR_plus export

```bash
%cd /content/ICPR2026/ICPR_2026

python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ../baseline_icpr_2026/data/train \
  --output-root "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/hr_plus" \
  --method lanczos2 \
  --scale 4

python -m src.handcrafted_modelling.export_hr_plus \
  --data-root ../baseline_icpr_2026/data/train \
  --output-root "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/hr_plus" \
  --method denoise_clahe_sharpen \
  --scale 4
```

Expected outputs:

- images under `.../hr_plus/x4/<method>/Scenario-*/.../track_*/lr-*_x4.png`
- `manifest_hr_plus_x4_<method>.csv`
- `run_hr_plus_x4_<method>.json`

---

## 9) Quick analysis (recommended for Sprint 1)

### 9.1 Plot baseline training curves

```python
import pandas as pd
import matplotlib.pyplot as plt

metrics_path = "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/baseline_rerun/metrics.csv"
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

### 9.2 Compare HR_plus methods by PSNR proxy

```python
import pandas as pd

root = "/content/drive/MyDrive/ICPR2026/runs/sprint1_2026-01-14/hr_plus"

for method in ["lanczos2", "denoise_clahe_sharpen"]:
    p = f"{root}/manifest_hr_plus_x4_{method}.csv"
    df = pd.read_csv(p)
    # psnr is empty for rows without HR pairing
    df["psnr_to_hr_resized_db"] = pd.to_numeric(df["psnr_to_hr_resized_db"], errors="coerce")
    print("\n===", method, "===")
    print(df["psnr_to_hr_resized_db"].describe())

    # Optional: scenario/layout slice from track path + plate_layout column
    df["scenario"] = df["track_dir"].str.extract(r"(Scenario-[AB])", expand=False)
    print(df.groupby(["scenario", "plate_layout"])['psnr_to_hr_resized_db'].mean())
```

---

## 10) Checklist (Sprint 1)

- [ ] Dataset extracted into `baseline_icpr_2026/data/train` on Colab runtime
- [ ] Baseline rerun completed; artifacts saved on Drive
- [ ] HR_plus x4 generated for at least 2 methods; manifests saved on Drive
- [ ] Quick plots / breakdown run
- [ ] Notes recorded for Sprint 2 hypothesis

---

## Appendix: troubleshooting

### A) `cv2` import fails

Run:

```bash
pip -q install -U opencv-python-headless
```

### B) Training is too slow

- Use GPU runtime
- Reduce epochs for a smoke test: `--epochs 3`
- Reduce batch size if OOM

### C) Validation split changes unexpectedly

The baseline persists the split in the output dir as `val_tracks.json`. Ensure you reuse the same `--output-dir` when you want identical splits.
