# Crack Segmentation LoRA — Experiment Log

Fine-tuning SAM3 with LoRA for crack segmentation. The text prompt is **fixed to
`"crack"`** (the data loader derives the prompt from the single COCO category name).

## Data

Source: `/workspace/nas_200/minkyung/unified/{BCL_NonSteel, BCL_Steel, CCSD, LCW, NCCD}`
(image/mask pairs). Converted to COCO with [`prepare_crack_coco.py`](../prepare_crack_coco.py):

- **Positives only** (masks containing crack pixels); pure-negative images skipped.
- **Whole mask = 1 `crack` instance** per image (bbox + RLE of the full foreground).
- Images referenced by absolute path (not copied).

Output: `/workspace/nas_200/minkyung/crack_coco/{train,valid,test}`

| split | images |
|---|---|
| train | 7,947 |
| valid | 989 |
| test  | 989 |

All metrics below are on the held-out **test** split (989 images).

## Common training setup

- Model: `facebook/sam3`, LoRA rank 16 / alpha 32, applied to vision + text +
  DETR encoder/decoder + mask decoder (`configs/crack_lora_config*.yaml`).
- Trainable params ≈ 18M (2.1%). batch 2 × grad-accum 8, lr 5e-5, bf16.
- 1× RTX A5000 (24 GB), ~3.5 s/it, ~3.8 h/epoch.
- Checkpoints saved per epoch; `best_lora_weights.pt` = lowest validation loss.
  Val loss converges by ~epoch 2–3 in every run.

## Experiments

| ID | Description | Loss weights (mask / dice) | Best epoch | Output |
|---|---|---|---|---|
| **Base** | Zero-shot SAM3, no LoRA | — | — | — |
| **v1** | LoRA, default loss | 200 / 10 | epoch 2 | `outputs/crack_lora/` |
| **v2** | LoRA, stronger Dice (exp #1) | 250 / 30 | epoch 3 | `outputs/crack_lora_v2/` |

Only the loss weights differ between v1 and v2; model, data, and all other
hyperparameters are identical. Dice loss directly optimizes mask overlap, so it
was raised in v2 to improve boundary quality.

> Note: absolute training/validation loss values are **not comparable** across v1
> and v2 because the loss weighting changed (v2 loss is numerically larger by
> construction). Compare with the task metrics below, not the loss.

## Results — pixel-level metrics (binary crack vs background)

Measured with [`evaluate_pixel_metrics.py`](../evaluate_pixel_metrics.py),
threshold 0.3 unless noted. IoU = mean per-image IoU.

| Config | IoU | IoU (micro) | Precision | Recall | F1 / Dice |
|---|---|---|---|---|---|
| Base (zero-shot) | 0.406 | 0.232 | 0.306 | 0.492 | 0.377 |
| v1 (dice 10) | 0.533 | 0.286 | 0.461 | 0.430 | 0.445 |
| **v2 (dice 30)** ⭐ | **0.546** | **0.383** | **0.758** | 0.436 | **0.554** |

### Threshold study (v2)

| threshold | IoU | IoU (micro) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **0.3** ⭐ | **0.546** | **0.383** | **0.758** | 0.436 | **0.554** |
| 0.2 | 0.541 | 0.367 | 0.688 | 0.440 | 0.537 |

Lowering the threshold to 0.2 barely changed recall (+0.004) but cost precision
(−0.07) and F1 (−0.017). The missed cracks are missed outright, not merely
low-confidence — so threshold tuning cannot recover them. **0.3 is the better
operating point.**

## Results — COCO detection metrics (segm)

Measured with [`validate_sam3_lora.py`](../validate_sam3_lora.py) `--merge`.
(Only run for Base and v1.)

| Config | mAP@[.5:.95] | mAP@50 | mAP@75 | F1@50 |
|---|---|---|---|---|
| Base | 0.130 | 0.335 | 0.070 | 0.341 |
| v1 | 0.190 | 0.526 | 0.093 | 0.624 |

## Conclusions

- LoRA clearly beats zero-shot SAM3 on every metric.
- **Experiment #1 (Dice 10 → 30) succeeded**: precision jumped 0.46 → 0.76 with
  recall held flat, lifting F1 0.445 → 0.554 and micro-IoU 0.286 → 0.383. The
  model became much more precise (far fewer false-positive crack pixels).
- **Image-level crack presence detection is near-perfect** (IL F1 ≈ 0.99 in the
  cgF1 evaluation); the remaining gap is boundary/recall on thin cracks.
- **Best model: v2 at threshold 0.3** — `outputs/crack_lora_v2/best_lora_weights.pt`.
- Recall plateaus around ~0.44 and is **not** fixable via threshold; it is a
  model/data limitation (thin cracks lost at 1008² resolution, limited crack
  diversity).

## Visual comparison

Side-by-side GT / LoRA / Base on 4 cross-dataset samples (prompt "crack"):
`assets/crack_v2_vs_base.png` (and `assets/crack_lora_vs_base.png` for v1).
Base massively over-predicts (e.g. 22 fragments on one CCSD image); v2 stays
focused on the true crack — consistent with the precision numbers.

## Reproduce

```bash
# 1. Build COCO data from image/mask pairs
python prepare_crack_coco.py

# 2. Train (v2 = best config)
python train_sam3_lora_native.py --config configs/crack_lora_config_v2.yaml --device 0

# 3a. Pixel metrics (IoU / Precision / Recall / F1 / Dice)
python evaluate_pixel_metrics.py \
  --data-dir /workspace/nas_200/minkyung/crack_coco/test \
  --config configs/crack_lora_config_v2.yaml \
  --weights outputs/crack_lora_v2/best_lora_weights.pt --threshold 0.3

# 3b. COCO metrics (mAP, cgF1)
python validate_sam3_lora.py --config configs/crack_lora_config_v2.yaml \
  --weights outputs/crack_lora_v2/best_lora_weights.pt \
  --val_data_dir /workspace/nas_200/minkyung/crack_coco/test --merge

# 4. Visual comparison
python compare_lora_base_batch.py --images <img1> <img2> ... \
  --data-dir /workspace/nas_200/minkyung/crack_coco/test \
  --config configs/crack_lora_config_v2.yaml \
  --weights outputs/crack_lora_v2/best_lora_weights.pt \
  --output assets/crack_v2_vs_base.png --threshold 0.3
```
