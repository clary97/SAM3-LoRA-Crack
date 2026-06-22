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


def source_of(file_name):
    """Map a COCO file_name (absolute source path) to its dataset name."""
    fn = file_name
    if "Non-steel" in fn or "Non-Steel" in fn:
        return "BCL_NonSteel"
    if "Bridge Crack Library" in fn and ("Steel" in fn or "steel" in fn):
        return "BCL_Steel"
    if "concreteCrackSegmentationDataset" in fn:
        return "CCSD"
    if "/LCW" in fn or "LCW " in fn:
        return "LCW"
    if "NCCD" in fn:
        return "NCCD"
    return "OTHER"


def _metrics(tp, fp, fn, ious, n, rel=None):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {
        "images": n,
        "mean_per_image_IoU": float(np.mean(ious)) if ious else 0.0,
        "micro_IoU": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "Precision": prec, "Recall": rec, "F1": f1,
        "Dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }
    if rel is not None:  # rel = (rprec_list, rrec_list, cldice_list)
        rp = float(np.mean(rel[0])) if rel[0] else 0.0
        rr = float(np.mean(rel[1])) if rel[1] else 0.0
        out["relaxed_Precision"] = rp
        out["relaxed_Recall"] = rr
        out["relaxed_F1"] = 2 * rp * rr / (rp + rr) if (rp + rr) else 0.0
        out["clDice"] = float(np.mean(rel[2])) if rel[2] else 0.0
    return out


def relaxed_and_cldice(gt, pred, tol=2):
    """Boundary-tolerant precision/recall (±tol px) and clDice for one image."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    gp, pp = gt.sum(), pred.sum()
    gt_d = ndi.binary_dilation(gt, iterations=tol) if tol > 0 else gt
    pred_d = ndi.binary_dilation(pred, iterations=tol) if tol > 0 else pred
    rprec = (pred & gt_d).sum() / pp if pp > 0 else (1.0 if gp == 0 else 0.0)
    rrec = (gt & pred_d).sum() / gp if gp > 0 else (1.0 if pp == 0 else 0.0)
    sp = skeletonize(pred) if pp > 0 else pred
    st = skeletonize(gt) if gp > 0 else gt
    tprec = (sp & gt_d).sum() / sp.sum() if sp.sum() > 0 else (1.0 if gp == 0 else 0.0)
    tsens = (st & pred_d).sum() / st.sum() if st.sum() > 0 else (1.0 if pp == 0 else 0.0)
    cldice = 2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) > 0 else 0.0
    return float(rprec), float(rrec), float(cldice)


def evaluate(model, images, data_dir, threshold, device, label, by_source=False,
             relaxed=False, tol=2):
    tp = fp = fn = 0
    ious = []
    rel = ([], [], [])              # rprec, rrec, cldice (overall)
    grp = {}                        # source -> [tp, fp, fn, [ious], rprec, rrec, cldice]
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
        i_iou = inter / union if union > 0 else 1.0
        i_fp = int(np.logical_and(pred, ~gt).sum())
        i_fn = int(np.logical_and(gt, ~pred).sum())
        ious.append(i_iou); tp += inter; fp += i_fp; fn += i_fn

        r = relaxed_and_cldice(gt, pred, tol) if relaxed else None
        if r:
            rel[0].append(r[0]); rel[1].append(r[1]); rel[2].append(r[2])

        if by_source:
            s = source_of(path)
            g = grp.setdefault(s, [0, 0, 0, [], [], [], []])
            g[0] += inter; g[1] += i_fp; g[2] += i_fn; g[3].append(i_iou)
            if r:
                g[4].append(r[0]); g[5].append(r[1]); g[6].append(r[2])

    out = {"label": label, "threshold": threshold,
           **_metrics(tp, fp, fn, ious, len(images), (rel if relaxed else None))}
    if by_source:
        out["by_source"] = {
            s: _metrics(g[0], g[1], g[2], g[3], len(g[3]),
                        ((g[4], g[5], g[6]) if relaxed else None))
            for s, g in sorted(grp.items(), key=lambda kv: -len(kv[1][3]))
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default="configs/crack_lora_config.yaml")
    ap.add_argument("--weights", default="outputs/crack_lora/best_lora_weights.pt")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--by-source", action="store_true",
                    help="Also report metrics broken down per source dataset")
    ap.add_argument("--skip-base", action="store_true",
                    help="Evaluate only the LoRA model (skip the base SAM3 baseline)")
    ap.add_argument("--relaxed", action="store_true",
                    help="Also report boundary-tolerant Precision/Recall/F1 and clDice")
    ap.add_argument("--tol", type=int, default=2, help="Tolerance in px for --relaxed")
    ap.add_argument("--out", default="pixel_metrics.json", help="Output JSON path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = json.load(open(Path(args.data_dir) / "_annotations.coco.json"))
    images = coco["images"]
    if args.num_samples:
        images = images[: args.num_samples]

    def row(label, m):
        base = (f"{label:13s} {m['images']:5d} {m['mean_per_image_IoU']:8.4f} "
                f"{m['micro_IoU']:9.4f} {m['Precision']:7.4f} {m['Recall']:7.4f} "
                f"{m['F1']:7.4f} {m['Dice']:7.4f}")
        if args.relaxed and "relaxed_F1" in m:
            base += (f"  | {m['relaxed_Precision']:7.4f} {m['relaxed_Recall']:7.4f} "
                     f"{m['relaxed_F1']:7.4f} {m['clDice']:7.4f}")
        return base

    results = []
    print("\n=== Evaluating LoRA model ===")
    lora = load_lora_model(args.config, args.weights, device)
    results.append(evaluate(lora, images, args.data_dir, args.threshold, device, "LoRA",
                            args.by_source, args.relaxed, args.tol))
    del lora
    torch.cuda.empty_cache()

    if not args.skip_base:
        print("\n=== Evaluating Base model ===")
        base = load_base_model(device)
        results.append(evaluate(base, images, args.data_dir, args.threshold, device, "Base",
                                args.by_source, args.relaxed, args.tol))

    print("\n" + "=" * 96)
    print(f"PIXEL-LEVEL CRACK SEGMENTATION METRICS (test={len(images)}, thr={args.threshold})")
    if args.relaxed:
        print(f"strict: IoU/Prec/Recall/F1/Dice   |   relaxed (+-{args.tol}px): rPrec rRecall rF1 clDice")
    print("=" * 96)
    hdr = f"{'group':13s} {'imgs':>5s} {'meanIoU':>8s} {'microIoU':>9s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Dice':>7s}"
    if args.relaxed:
        hdr += "  | " + f"{'rPrec':>7s} {'rRecall':>7s} {'rF1':>7s} {'clDice':>7s}"
    print(hdr)
    for r in results:
        print(row(r["label"], r))
        if "by_source" in r:
            for s, m in r["by_source"].items():
                print(row(f"  {s}", m))
    print("=" * 96)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
