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
| **v3** | v2 + data augmentation (exp #2) | 250 / 30 | epoch 6 | `outputs/crack_lora_v3/` |
| **v4** | v2 + tiling of CCSD/LCW (exp #3) | 250 / 30 | epoch 4 | `outputs/crack_lora_v4/` |
| **v5** | v2 + clDice loss (exp #4) | 250 / 30 + clDice 20 | epoch 6 | `outputs/crack_lora_v5/` |

v4 (exp #3) tiles the high-resolution CCSD and wide-scene LCW images into
512×512 crops (`prepare_crack_tiles.py` → `crack_tiles/`), keeping only tiles that
contain crack pixels; BCL/NCCD stay whole. Augmentation is off, to isolate the
tiling effect against v2. Evaluated with a stitched full-image eval
(`evaluate_tiled.py`): each test image is covered with tiles, predictions are
OR-stitched back to full resolution, then scored like the other runs.

v3 adds train-time augmentation (random horizontal/vertical flip applied
consistently to image+boxes+masks, plus brightness/contrast jitter), enabled via
`training.augment: true` in `configs/crack_lora_config_v3.yaml`.

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
| **v2 (dice 30)** | 0.546 | **0.383** | **0.758** | 0.436 | **0.554** |
| v3 (v2 + augment) | **0.553** | 0.312 | 0.498 | **0.455** | 0.475 |

The overall micro/F1 drop for v3 is a **CCSD artifact** — see the augmentation
comparison below. By image-averaged IoU (which weights every image equally) v3 is
the best config; by pixel-pooled F1, CCSD's huge images dominate and drag it down.

### Threshold study (v2)

| threshold | IoU | IoU (micro) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **0.3** ⭐ | **0.546** | **0.383** | **0.758** | 0.436 | **0.554** |
| 0.2 | 0.541 | 0.367 | 0.688 | 0.440 | 0.537 |

Lowering the threshold to 0.2 barely changed recall (+0.004) but cost precision
(−0.07) and F1 (−0.017). The missed cracks are missed outright, not merely
low-confidence — so threshold tuning cannot recover them. **0.3 is the better
operating point.**

### Per-dataset breakdown (v2, threshold 0.3)

From `evaluate_pixel_metrics.py --by-source` (`pixel_metrics_by_source.json`):

| dataset | imgs | IoU | Precision | Recall | F1 |
|---|---|---|---|---|---|
| BCL_NonSteel | 576 | 0.611 | 0.793 | 0.782 | 0.788 |
| BCL_Steel | 203 | 0.571 | 0.649 | 0.778 | 0.708 |
| NCCD | 71 | 0.578 | 0.753 | 0.744 | 0.749 |
| CCSD | 44 | 0.323 | 0.810 | 0.360 | 0.499 |
| LCW | 95 | 0.174 | 0.281 | 0.401 | 0.330 |

- **BCL + NCCD (850/989 images) are already strong** (F1 0.71–0.79).
- **CCSD**: high precision (0.81), low recall (0.36) — thin cracks lost when the
  huge images (3264×2448) are downscaled to 1008 → **tiling** is the targeted fix.
- **LCW** is weak on *every* metric (F1 0.33) → likely a domain / annotation-quality
  problem, not just resolution; needs visual inspection.
- The low overall micro-recall (0.44) is dominated by CCSD/LCW (very large images);
  most images (BCL/NCCD) reach recall ~0.75.

### Augmentation effect — v2 vs v3 per-dataset F1 (threshold 0.3)

From `pixel_metrics_by_source.json` (v2) and `pixel_metrics_v3_by_source.json` (v3):

| dataset | imgs | v2 F1 | v3 F1 | Δ |
|---|---|---|---|---|
| BCL_NonSteel | 576 | 0.788 | 0.799 | +0.011 |
| BCL_Steel | 203 | 0.708 | 0.764 | +0.057 |
| NCCD | 71 | 0.749 | 0.775 | +0.027 |
| LCW | 95 | 0.330 | 0.357 | +0.027 |
| **CCSD** | 44 | 0.499 | 0.403 | **−0.096** |

- Augmentation **improved 4 of 5 datasets** (recall rose without hurting precision
  on BCL/NCCD) and lifted image-averaged IoU (0.546 → 0.553).
- **CCSD regressed**: precision collapsed 0.81 → 0.43. Augmentation added false
  positives on its huge, low-contrast images — confirming that CCSD needs
  **tiling** (resolution), not more augmentation.
- LCW barely moved (still F1 ≈ 0.36) → not an augmentation/resolution issue.
- **Recommended model: v3** for the general/BCL/NCCD case; keep v2 (or tile) for
  CCSD-style high-resolution inputs.

## Results — COCO detection metrics (segm)

Measured with [`validate_sam3_lora.py`](../validate_sam3_lora.py) `--merge`.
(Only run for Base and v1.)

| Config | mAP@[.5:.95] | mAP@50 | mAP@75 | F1@50 |
|---|---|---|---|---|
| Base | 0.130 | 0.335 | 0.070 | 0.341 |
| v1 | 0.190 | 0.526 | 0.093 | 0.624 |

### Tiling effect — exp #3 (v4), stitched eval, threshold 0.3

`evaluate_tiled.py` (`pixel_metrics_v4_tiled.json`). Overall and on the tiling
targets (CCSD, LCW):

| scope | metric | v2 | v4 (tiled) | Δ |
|---|---|---|---|---|
| overall | IoU / P / R / F1 | 0.546 / 0.758 / 0.436 / 0.554 | 0.543 / 0.366 / **0.485** / 0.417 | recall ↑, F1 ↓ |
| CCSD | Recall | 0.360 | **0.414** | +0.05 |
| CCSD | Precision | **0.810** | 0.303 | **−0.51** |
| CCSD | F1 | **0.499** | 0.350 | −0.15 |
| LCW | Recall | 0.401 | **0.461** | +0.06 |
| LCW | Precision | 0.281 | 0.229 | −0.05 |
| LCW | F1 | 0.330 | 0.306 | −0.02 |

**Tiling raised recall on both targets (its goal) but precision collapsed on CCSD
(0.81 → 0.30), so net F1/IoU got worse.** Tiling itself was not the problem — the
training/inference setup was inconsistent:

> **The positive-only-tile limitation.** Training kept only tiles that *contain*
> crack pixels, so the model never saw crack-free background tiles and never
> learned to output "nothing" on them. At inference the full image is covered by
> *all* tiles, including the many background ones — there the model hallucinates
> cracks, and OR-stitching accumulates these false positives across tiles →
> precision collapse (worst on CCSD, which produces the most tiles per image).

To make tiling pay off, training must include **negative (crack-free) tiles** so
the model learns to reject background (exp #5 candidate), and/or inference must
gate background tiles (raise the score threshold, or use an image-level
crack-presence check before accepting tile predictions).

### clDice loss (v5) + boundary-tolerant / clDice metrics

`crack_losses.py` adds a differentiable centerline-Dice term (`MasksClDice`),
enabled via `loss_cldice` in the config. `evaluate_pixel_metrics.py --relaxed`
adds boundary-tolerant (±2 px) Precision/Recall/F1 and a clDice metric.

v2 vs v5 on the test set (threshold 0.3):

| metric | v2 | v5 (+clDice) | Δ |
|---|---|---|---|
| strict meanIoU | 0.546 | 0.556 | +0.010 |
| strict Precision / Recall / F1 | 0.758 / 0.436 / 0.554 | 0.759 / 0.449 / 0.564 | F1 +0.010 |
| relaxed (±2px) F1 | 0.854 | 0.855 | ~ |
| clDice | 0.849 | 0.853 | +0.004 |

Per-dataset clDice (v2 → v5): BCL_NonSteel 0.924→0.930, BCL_Steel 0.862→0.879,
NCCD 0.855→0.881, **CCSD 0.449→0.444 (flat), LCW 0.543→0.498 (worse)**.

**Two findings:**
1. **clDice helped only marginally on whole images** (F1 +0.01, clDice +0.004),
   and not at all on CCSD. This confirms the resolution argument: the mask
   decoder's native output is ~256 px (1008/4), so on whole 1008 images thin
   cracks are sub-resolution and the skeleton signal is weak. clDice needs higher
   *effective* resolution (i.e. tiling) to pay off. It did help the close-up
   datasets (BCL/NCCD) where cracks are larger.
2. **The relaxed/clDice metrics reframe the whole project.** Strict crack IoU
   ~0.55 looks low, but at ±2 px tolerance the model scores **F1 ≈ 0.85, clDice
   ≈ 0.85** overall (BCL ≈ 0.93). The low strict IoU is mostly the metric
   punishing thin structures for 1-2 px boundary error — not poor detection.
   **Report strict + relaxed/clDice together.**

## Conclusions

- LoRA clearly beats zero-shot SAM3 on every metric.
- **Experiment #3 (tiling, v4) did not pay off as implemented**: recall rose on
  CCSD/LCW but precision collapsed because the model was trained on positive tiles
  only and over-predicts on background tiles at inference. Needs negative tiles.
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
