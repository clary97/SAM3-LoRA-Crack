#!/usr/bin/env python3
"""
Batch inference of a trained crack-LoRA model over a folder of images that have
NO ground-truth masks (e.g. field/drone data). Produces:
  - prediction overlays (image + red predicted crack mask) for a sample per folder
  - a per-folder summary: #images, mean detections, % images with >=1 detection,
    mean predicted crack-pixel fraction

No GT -> this is QUALITATIVE only (no IoU/Precision/Recall/F1).

Whole-image inference (the model resizes input to 1008), matching how v2/v5 were
trained/evaluated. Huge source images are downscaled to --disp px first (the
model's 1008 bottleneck makes the pre-downscale loss negligible) to keep the
mask upsample cheap.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from compare_lora_base_batch import load_lora_model, predict

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def overlay(pil_img, mask, out_path):
    """Save image with the crack mask drawn in red."""
    im = np.array(pil_img.convert("RGB"))
    if mask is not None and mask.any():
        if mask.shape != im.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(
                (im.shape[1], im.shape[0]), Image.NEAREST)).astype(bool)
        im[mask] = (0.4 * im[mask] + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    Image.fromarray(im).save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Folder with images (may have subfolders)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, help="Output dir for overlays + summary")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--disp", type=int, default=2048, help="Downscale long side to this before inference/overlay")
    ap.add_argument("--save-per-folder", type=int, default=20, help="How many overlays to save per subfolder")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lora_model(args.config, args.weights, device)
    root = Path(args.data_dir)
    out_root = Path(args.out)

    # group images by immediate subfolder (or root)
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] or [root]
    tmp = "/tmp/_infer_disp.jpg"
    summary = {}

    for sub in subdirs:
        imgs = sorted([p for p in sub.iterdir() if p.suffix in IMG_EXTS])
        if not imgs:
            continue
        odir = out_root / sub.name
        odir.mkdir(parents=True, exist_ok=True)
        n_det_total = n_with = 0
        frac_sum = 0.0
        saved = 0
        for p in tqdm(imgs, desc=sub.name):
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"skip {p.name}: {e}"); continue
            # downscale long side to --disp
            w, h = im.size
            scale = args.disp / max(w, h)
            if scale < 1:
                im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            im.save(tmp, quality=95)
            _, masks, count = predict(model, tmp, "crack", 1008, args.threshold, device)
            union = None
            if masks is not None and len(masks):
                union = np.zeros(masks.shape[-2:], dtype=bool)
                for m in masks:
                    union |= m.astype(bool)
            frac = float(union.mean()) if union is not None else 0.0
            n_det_total += count
            n_with += 1 if count > 0 else 0
            frac_sum += frac
            if saved < args.save_per_folder:
                overlay(im, union, odir / f"{p.stem}_pred.jpg")
                saved += 1
        n = len(imgs)
        summary[sub.name] = {
            "images": n,
            "mean_detections": round(n_det_total / n, 3),
            "pct_images_with_detection": round(100 * n_with / n, 1),
            "mean_crack_pixel_pct": round(100 * frac_sum / n, 4),
            "overlays_saved": saved,
        }
        print(f"[{sub.name}] {summary[sub.name]}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY (no GT -> qualitative) ===")
    for k, v in summary.items():
        print(f"{k:14s} imgs={v['images']:4d}  meanDet={v['mean_detections']:6.2f}  "
              f"%withDet={v['pct_images_with_detection']:5.1f}  meanCrack%={v['mean_crack_pixel_pct']:.3f}")
    print(f"Saved overlays + {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
