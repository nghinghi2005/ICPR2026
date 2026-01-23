# Sprint 2 – Kaggle Notebook (SR → OCR-CTC)

This folder contains a Kaggle-ready, self-contained pipeline:

1) Train/export a lightweight Super-Resolution model to generate **HR+** frames from LR inputs.
2) Train a Multi-Frame CRNN using **CTC loss**.
3) Decode with **Brazil/Mercosur template constraints** (regex + confusion-map correction) to reduce common OCR confusions (O/0, I/1, B/8, etc.).

## Kaggle storage

- Read-only dataset: `/kaggle/input/...`
- Safe writable outputs: `/kaggle/working/sprint2_outputs/...`
  - checkpoints
  - exported HR+ images
  - metrics + sample predictions

To persist artifacts across sessions, use Kaggle "Save & Commit" so `/kaggle/working` outputs are saved with the notebook version.

## Entry point

Open and run: `sprint2/sprint2_kaggle_pipeline.ipynb`
