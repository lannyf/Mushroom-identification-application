# Mushroom Segmentation Pipeline Runbook

This document describes the complete offline pseudo-label → fine-tuning pipeline built for mushroom segmentation.

---

## 1. Prerequisites (Completed)

| Component | Status | Location |
|-----------|--------|----------|
| SAM 2 repo + install | ✅ | `sam2/` (editable install in `.venv`) |
| SAM 2 checkpoint | ✅ | `artifacts/sam2.1_hiera_tiny.pt` |
| YOLOv8n-seg weights | ✅ | `artifacts/yolov8n-seg.pt` |
| Background images | ✅ | `data/raw/background/` (48 images) |
| Manual annotations (12 eval) | ✅ | `data/Mushroom segmentation.coco-segmentation/train/` |
| Evaluation images | ✅ | `data/raw/evaluation_images/` (60 images) |

**Important:** All SAM 2 scripts **must** be run from a directory that does NOT contain a `sam2/` sub-folder (e.g. `cd /tmp` before running).

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: SAM 2 Pilot (gate check)                           │
│  → scripts/sam2_pilot.py                                    │
│  → 20 images, must achieve >80% usable masks                │
└─────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: SAM 2 Batch Generation (training images)           │
│  → scripts/generate_sam2_masks.py                           │
│  → Outputs: data/SegMaskSAM2/ + manifest.jsonl              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: SAM 2 Batch Generation (evaluation images)         │
│  → scripts/generate_sam2_eval_masks.py                      │
│  → Outputs: data/SegMaskSAM2_eval/ + manifest.jsonl         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Prepare YOLOv8-seg Dataset                         │
│  → scripts/prepare_yolo_seg_dataset.py                      │
│  → Outputs: data/segmentation/dataset.yaml                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Train YOLOv8n-seg (Google Colab recommended)       │
│  → scripts/train_yolov8_seg.py                              │
│  → Outputs: artifacts/yolov8_seg_ft.pt                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Evaluate & Promote                                 │
│  → scripts/evaluate_segmentation.py                         │
│  → Update models/mushroom_segmenter.py                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Commands

### 3.1 SAM 2 Pilot (Already Run — PASSED 90%)

```bash
cd /tmp
python /path/to/project/scripts/sam2_pilot.py \
  --project-root /path/to/project \
  --n-images 20 \
  --seed 42
```

**Result:** 18/20 usable (90%) — **GATE PASSED**
- Report: `artifacts/sam2_pilot/sam2_pilot_report.json`
- Strategy: center_point=16, bbox_fallback=2, failed=2

---

### 3.2 SAM 2 Batch Generation — Training Images

```bash
cd /tmp
python /path/to/project/scripts/generate_sam2_masks.py \
  --project-root /path/to/project \
  --images-dir data/raw/images \
  --output-dir data/SegMaskSAM2
```

**Status:** Currently running in background on 352 images.
- Resume supported (reads existing `data/SegMaskSAM2/manifest.jsonl`)
- Expected time: ~25–50 minutes on CPU
- Failed images logged in manifest for manual review

---

### 3.3 SAM 2 Batch Generation — Evaluation Images

Run this **after** training images are done (or in parallel on another machine):

```bash
# Holdout set (30 images)
cd /tmp
python /path/to/project/scripts/generate_sam2_eval_masks.py \
  --project-root /path/to/project \
  --eval-list data/raw/eval_holdout_30.txt \
  --output-dir data/SegMaskSAM2_eval_holdout

# Secondary set (30 images)
cd /tmp
python /path/to/project/scripts/generate_sam2_eval_masks.py \
  --project-root /path/to/project \
  --eval-list data/raw/eval_secondary_30.txt \
  --output-dir data/SegMaskSAM2_eval_secondary
```

---

### 3.4 COCO → YOLO Conversion (Already Run)

The 12 manually annotated evaluation images were converted:

```bash
python scripts/convert_coco_to_yolo.py \
  --coco-json "data/Mushroom segmentation.coco-segmentation/train/_annotations.coco.json" \
  --images-dir "data/Mushroom segmentation.coco-segmentation/train" \
  --output-dir data/segmentation/eval_annotations/yolo \
  --rdp-epsilon 2.0
```

**Result:** 12 .txt files with 33 polygon instances total.

---

### 3.5 Prepare YOLOv8-seg Dataset

Run this **after** Step 3.2 completes:

```bash
python scripts/prepare_yolo_seg_dataset.py \
  --images-dir data/raw/images \
  --masks-dir data/SegMaskSAM2 \
  --background-dir data/raw/background \
  --output-dir data/segmentation \
  --train-ratio 0.80 \
  --seed 42 \
  --bg-train 30 \
  --bg-val 10
```

**Output:**
```
data/segmentation/
  dataset.yaml
  images/train/      (~282 training + 30 background)
  images/val/        (~71 training + 10 background)
  labels/train/      (.txt polygon files)
  labels/val/
```

---

### 3.6 Train YOLOv8n-seg

**Primary path (recommended): Google Colab**

1. Zip `data/segmentation/` and upload to Google Drive.
2. In Colab:
```python
!pip install ultralytics
!python scripts/train_yolov8_seg.py \
  --dataset /content/drive/MyDrive/segmentation/dataset.yaml \
  --device 0
```
3. Download `artifacts/yolov8_seg_ft.pt` back to the project.

**Local CPU path (10–24 hours):**
```bash
python scripts/train_yolov8_seg.py \
  --dataset data/segmentation/dataset.yaml \
  --device cpu
```

**Hyperparameters used:**
- `epochs=100`, `imgsz=640`, `batch=8`
- `patience=20`, `close_mosaic=10`
- `lr0=0.001`
- Strong HSV augmentation

---

### 3.7 Evaluate & Promote

```bash
# Evaluate fine-tuned model against holdout ground truth
python scripts/evaluate_segmentation.py \
  --model artifacts/yolov8_seg_ft.pt \
  --images-dir data/raw/evaluation_images \
  --masks-dir data/SegMaskSAM2_eval_holdout \
  --compare-generic \
  --output artifacts/segmentation_evaluation.json
```

**Promotion gate:**
- Improve on ≥2 of 3 metrics (IoU / Precision / Recall)
- Recall must not drop >0.05 absolute or 10% relative
- Full `pytest tests/` must pass
- CPU inference within 150% of generic model speed

---

## 4. Evaluation Split

| Set | Count | Ground Truth Source |
|-----|-------|---------------------|
| Holdout | 30 | 12 manual COCO + 18 SAM 2 reviewed |
| Secondary | 30 | 30 SAM 2 reviewed |

Files:
- `data/raw/eval_holdout_30.txt`
- `data/raw/eval_secondary_30.txt`

---

## 5. Updated Runtime Code

`models/mushroom_segmenter.py` has been updated:
- `get_segmenter()` now accepts `preferred_path` and `fallback_path`
- Defaults to `artifacts/yolov8_seg_ft.pt` if it exists, else `artifacts/yolov8n-seg.pt`
- Added post-processing heuristics:
  - Center-bias tiebreaker
  - Skin-colour rejection (>30% skin pixels rejected)
  - Aspect ratio guard (w/h > 4 or < 0.25 rejected)

---

## 6. Files Created

| File | Purpose |
|------|---------|
| `scripts/convert_coco_to_yolo.py` | COCO → YOLO polygon converter |
| `scripts/sam2_pilot.py` | 20-image quality gate |
| `scripts/generate_sam2_masks.py` | Batch SAM 2 mask generation (training) |
| `scripts/generate_sam2_eval_masks.py` | Batch SAM 2 mask generation (eval) |
| `scripts/prepare_yolo_seg_dataset.py` | YOLO dataset preparation |
| `scripts/train_yolov8_seg.py` | YOLOv8-seg training script |
| `scripts/evaluate_segmentation.py` | Segmentation evaluation & promotion gate |
| `scripts/SEGMENTATION_RUNBOOK.md` | This document |
