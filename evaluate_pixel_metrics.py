#!/usr/bin/env python3
"""
Pixel-level crack-segmentation metrics (IoU / Precision / Recall / F1 / Dice)
for the LoRA model and the base SAM3 model, on a COCO test set.

Treats crack segmentation as a binary (crack vs background) problem: all
predicted instance masks above the score threshold are unioned into one
crack map and compared pixel-wise against the union of GT masks.

Reuses the proven inference/GT helpers from compare_lora_base_batch.py.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from compare_lora_base_batch import (
    load_lora_model, load_base_model, predict, load_ground_truth,
)


def union_masks(masks, shape):
    out = np.zeros(shape, dtype=bool)
    if masks is None:
        return out
    for m in masks:
        mm = np.asarray(m).astype(bool)
        if mm.shape != shape:  # safety: align to GT resolution
            from PIL import Image as PILImage
            mm = np.array(PILImage.fromarray(mm).resize(
                (shape[1], shape[0]), PILImage.NEAREST)).astype(bool)
        out |= mm
    return out


def evaluate(model, images, data_dir, threshold, device, label):
    tp = fp = fn = 0
    ious = []
    for im in tqdm(images, desc=label):
        path = im["file_name"]
        H, W = im["height"], im["width"]
        gt_masks, prompt = load_ground_truth(path, data_dir)
        prompt = prompt or "crack"
        gt = union_masks(gt_masks, (H, W))
        _, pred_masks, _ = predict(model, path, prompt, 1008, threshold, device)
        pred = union_masks(pred_masks, (H, W))

        inter = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        ious.append(inter / union if union > 0 else 1.0)
        tp += inter
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(gt, ~pred).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    micro_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "label": label, "images": len(images), "threshold": threshold,
        "mean_per_image_IoU": float(np.mean(ious)),
        "micro_IoU": micro_iou, "Precision": prec, "Recall": rec,
        "F1": f1, "Dice": dice,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default="configs/crack_lora_config.yaml")
    ap.add_argument("--weights", default="outputs/crack_lora/best_lora_weights.pt")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--num-samples", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = json.load(open(Path(args.data_dir) / "_annotations.coco.json"))
    images = coco["images"]
    if args.num_samples:
        images = images[: args.num_samples]

    results = []
    print("\n=== Evaluating LoRA model ===")
    lora = load_lora_model(args.config, args.weights, device)
    results.append(evaluate(lora, images, args.data_dir, args.threshold, device, "LoRA"))
    del lora
    torch.cuda.empty_cache()

    print("\n=== Evaluating Base model ===")
    base = load_base_model(device)
    results.append(evaluate(base, images, args.data_dir, args.threshold, device, "Base"))

    print("\n" + "=" * 72)
    print(f"PIXEL-LEVEL CRACK SEGMENTATION METRICS (test={results[0]['images']}, thr={args.threshold})")
    print("=" * 72)
    hdr = f"{'model':6s} {'meanIoU':>8s} {'microIoU':>9s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Dice':>7s}"
    print(hdr)
    for r in results:
        print(f"{r['label']:6s} {r['mean_per_image_IoU']:8.4f} {r['micro_IoU']:9.4f} "
              f"{r['Precision']:7.4f} {r['Recall']:7.4f} {r['F1']:7.4f} {r['Dice']:7.4f}")
    print("=" * 72)
    with open("pixel_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved pixel_metrics.json")


if __name__ == "__main__":
    main()
