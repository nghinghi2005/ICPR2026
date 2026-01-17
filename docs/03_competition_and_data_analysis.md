# ICPR 2026 LRLPR Competition + Dataset Analysis

Date: 2026-01-14

This document summarizes the competition brief (`ICPR_2026_LRLPR_Competition.pdf`) and analyzes the training data provided in this workspace (`data.zip`).

---

## 1) Task definition (what you must predict)

- Input per sample: a **track** of **5 low-resolution (LR) images** of the same license plate.
- Output per sample: **license plate text**.
- Constraint: during competition evaluation, **only LR images are available**.

HR images:

- HR images are provided **only in training**.
- They exist to support super-resolution/enhancement strategies.

---

## 2) Evaluation metric and ranking

### Primary metric: Recognition Rate (Exact Match)

A track counts as correct only if the predicted text matches the ground truth exactly (any character mismatch => incorrect).

$$
\text{RecognitionRate} = \frac{\#\text{CorrectTracks}}{\#\text{TotalTracks}}
$$

### Tie-breaker: Confidence Gap

If two teams achieve the same Recognition Rate, ranking is broken by Confidence Gap:

$$
\text{ConfidenceGap} = \mu(\text{confidence | correct}) - \mu(\text{confidence | incorrect})
$$

The team with **higher Confidence Gap** ranks higher.

Implication:

- Your system should output a calibrated confidence score per track.
- Confidence must separate correct vs incorrect predictions, not just be high.

---

## 3) Submission format

You submit a `.zip` containing a single `.txt` file.

Each line:

```
track_00001,ABC1234;0.9876
```

Fields:

- `track_id,plate_text;confidence`

Participants may aggregate across the five LR frames using:

- majority voting
- confidence-based selection
- temporal modeling

Submission limits (public test set):

- 5 submissions/day
- 25 total over competition period

---

## 4) Dataset description (from competition brief)

### Track structure

- Each track contains 5 consecutive LR images.
- Training tracks also include 5 consecutive HR images.

### Scenarios

- Scenario A (10,000 tracks): more controlled conditions.
- Scenario B (10,000 tracks): broader environmental conditions.

Test sets:

- Public Test Set (1,000 tracks): subset of Scenario B.
- Blind Test Set (3,000+ tracks): Scenario B.

Notes:

- Scenario B has *different camera orientation* and more varied conditions.

### Dataset creation (high-level)

- plates detected (YOLOv11)
- tracked (BoT-SORT)
- LR frames: vehicle farther from camera
- HR frames: vehicle closer to camera
- text labeled semi-automatically using a strong recognition model on HR

---

## 5) Rules and constraints (practical)

- External datasets are allowed, but must be documented.
- If you use external data and rank high, you may need to provide results trained only on the provided training set.
- Test data must not be used for training in any way.
- Organizers may request code from top-ranked teams for verification.

---

## 6) Local data in this workspace: `data.zip`

### 6.1 What’s inside

This workspace contains:

- `data.zip` with a `train/` tree only (no `test/` or `val/` directory inside the zip).

A typical track folder:

```
train/Scenario-B/Brazilian/track_10001/
  lr-001.jpg ... lr-005.jpg
  hr-001.jpg ... hr-005.jpg
  annotations.json
```

### 6.2 Annotation schema

A sample `annotations.json` observed in the zip:

```json
{
  "plate_layout": "Brazilian",
  "plate_text": "BAI8068",
  "corners": {}
}
```

Fields seen:

- `plate_text`: the ground-truth string.
- `plate_layout`: layout category.
- `corners`: present but can be empty. (The brief notes Scenario B may not have corner annotations.)

### 6.3 Scenarios and layouts present

By scanning the zip contents:

- Scenarios present: `Scenario-A`, `Scenario-B`
- Layouts present: `Brazilian`, `Mercosur`

### 6.4 Track counts (local)

Counts in `data.zip`:

- Scenario-A Brazilian: 5,000 tracks
- Scenario-A Mercosur: 5,000 tracks
- Scenario-B Brazilian: 2,000 tracks
- Scenario-B Mercosur: 8,000 tracks

Total training tracks: 20,000.

### 6.5 Practical implications for modelling

- Layout is a strong conditioning signal (different fonts/spacing/structure). You can:
  - train separate recognizers per layout, or
  - train a single model with layout token/conditioning.

- Scenario shift is real:
  - Scenario A is more controlled.
  - Scenario B is closer to test conditions.

A common strategy:

- start training on all data
- validate and tune using a Scenario-B-heavy validation split

---

## 7) How to use this dataset with the repos

### 7.1 Extract the zip (recommended layout)

To match `baseline_icpr_2026/config.py` default (`DATA_ROOT = "data/train"`), extract the zip into:

- `baseline_icpr_2026/data/`

so you end up with:

- `baseline_icpr_2026/data/train/Scenario-A/...`

A PowerShell helper script is provided in this workspace:

- `scripts/prepare_data.ps1`

### 7.2 Baseline training

- `baseline_icpr_2026/train.py` scans recursively for `track_*`.
- It will load 5 LR frames per track (and sometimes HR+degradation during training).

### 7.3 Degradation modelling (ICPR_2026)

- `ICPR_2026/src/degradation_modelling` fits degradation params using paired HR/LR.
- This can produce domain-aligned synthetic LR for augmentation.

---

## 8) What’s missing locally (as of now)

- No `test/` set is present in `data.zip`.
- The public/blind test sets described in the competition brief are not included here.

So for Sprint 1, you will validate using a held-out split from training data.

---

## 9) Recommended Sprint-1 evaluation protocol

Since you don’t have official test data locally:

1) Create a **fixed validation split** (persist track IDs).
2) Report Recognition Rate (exact match) on that split.
3) Report per-scenario and per-layout breakdown:
   - A vs B
   - Brazilian vs Mercosur
4) Keep artifacts in a run directory so you can compare runs.

The Sprint 1 runbook explains how to do this consistently.
