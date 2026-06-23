#!/usr/bin/env python3
"""
Stitched evaluation for a tile-trained model, comparable to evaluate_pixel_metrics.

For each FULL test image (crack_coco/test):
  - CCSD / LCW  -> cover the image with overlapping tiles, run inference per tile,
    OR-stitch the predicted tile masks back onto a full-resolution canvas.
  - BCL / NCCD  -> run inference on the whole image (small enough already).
Then compute pixel IoU / Precision / Recall / F1 / Dice against the full GT mask,
overall and per source dataset — same definitions as evaluate_pixel_metrics.py.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
import pycocotools.mask as mask_utils
from tqdm import tqdm

from compare_lora_base_batch import load_lora_model, predict
from evaluate_pixel_metrics import source_of, _metrics, union_masks, relaxed_and_cldice

ImageFile.LOAD_TRUNCATED_IMAGES = True
TILE_SOURCES = {"CCSD", "LCW"}


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def gt_mask(anns, H, W):
    m = np.zeros((H, W), dtype=bool)
    for a in anns:
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            m |= mask_utils.decode(seg).astype(bool)
        elif isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, H, W)
            m |= mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    return m


def predict_full(model, path, threshold, device, tile, stride, tmp):
    """Return a full-resolution binary prediction for one image."""
    if source_of(path) not in TILE_SOURCES:
        pil, masks, _ = predict(model, path, "crack", 1008, threshold, device)
        return union_masks(masks, (pil.size[1], pil.size[0]))

    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    canvas = np.zeros((h, w), dtype=bool)
    for y in tile_positions(h, tile, stride):
        for x in tile_positions(w, tile, stride):
            crop = img[y:y + tile, x:x + tile]
            Image.fromarray(crop).save(tmp, quality=95)
            _, masks, _ = predict(model, tmp, "crack", 1008, threshold, device)
            canvas[y:y + crop.shape[0], x:x + crop.shape[1]] |= union_masks(masks, crop.shape[:2])
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="FULL-image test dir (crack_coco/test)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--relaxed", action="store_true",
                    help="Also report boundary-tolerant P/R/F1 and clDice")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--out", default="pixel_metrics_v4_tiled.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = json.load(open(Path(args.data_dir) / "_annotations.coco.json"))
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    images = coco["images"][: args.num_samples] if args.num_samples else coco["images"]

    model = load_lora_model(args.config, args.weights, device)
    tmp = "/tmp/_eval_tile.jpg"
    tp = fp = fn = 0
    ious = []
    rel = ([], [], [])
    grp = {}
    for im in tqdm(images, desc="stitched"):
        H, W = im["height"], im["width"]
        gt = gt_mask(ann_by_img.get(im["id"], []), H, W)
        pred = predict_full(model, im["file_name"], args.threshold, device, args.tile, args.stride, tmp)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST)).astype(bool)
        inter = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        i_fp = int(np.logical_and(pred, ~gt).sum())
        i_fn = int(np.logical_and(gt, ~pred).sum())
        ious.append(inter / union if union > 0 else 1.0)
        tp += inter; fp += i_fp; fn += i_fn
        r = relaxed_and_cldice(gt, pred, args.tol) if args.relaxed else None
        if r:
            rel[0].append(r[0]); rel[1].append(r[1]); rel[2].append(r[2])
        s = source_of(im["file_name"])
        g = grp.setdefault(s, [0, 0, 0, [], [], [], []])
        g[0] += inter; g[1] += i_fp; g[2] += i_fn; g[3].append(ious[-1])
        if r:
            g[4].append(r[0]); g[5].append(r[1]); g[6].append(r[2])

    overall = {"label": "v6_tiled", "threshold": args.threshold,
               **_metrics(tp, fp, fn, ious, len(images), (rel if args.relaxed else None))}
    overall["by_source"] = {s: _metrics(g[0], g[1], g[2], g[3], len(g[3]),
                                        ((g[4], g[5], g[6]) if args.relaxed else None))
                            for s, g in sorted(grp.items(), key=lambda kv: -len(kv[1][3]))}

    def row(label, m):
        base = (f"{label:13s} {m['images']:5d} {m['mean_per_image_IoU']:8.4f} "
                f"{m['micro_IoU']:9.4f} {m['Precision']:7.4f} {m['Recall']:7.4f} "
                f"{m['F1']:7.4f} {m['Dice']:7.4f}")
        if args.relaxed and "relaxed_F1" in m:
            base += (f"  | {m['relaxed_Precision']:7.4f} {m['relaxed_Recall']:7.4f} "
                     f"{m['relaxed_F1']:7.4f} {m['clDice']:7.4f}")
        return base

    print("\n" + "=" * 96)
    print(f"STITCHED TILED EVAL (test={len(images)}, thr={args.threshold}, tile={args.tile}/{args.stride})")
    print("=" * 96)
    hdr = f"{'group':13s} {'imgs':>5s} {'meanIoU':>8s} {'microIoU':>9s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Dice':>7s}"
    if args.relaxed:
        hdr += "  | " + f"{'rPrec':>7s} {'rRecall':>7s} {'rF1':>7s} {'clDice':>7s}"
    print(hdr)
    print(row(overall["label"], overall))
    for s, m in overall["by_source"].items():
        print(row(f"  {s}", m))
    print("=" * 96)
    json.dump([overall], open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
