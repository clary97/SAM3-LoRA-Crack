#!/usr/bin/env python3
"""
Convert the unified crack image/mask dataset into the COCO format expected by
COCOSegmentDataset in train_sam3_lora_native.py.

Design decisions (confirmed with user):
  - Positives only: images whose mask contains crack pixels are used; pure-negative
    images (all-black masks) are skipped.
  - Whole mask = 1 instance: each image gets a single "crack" annotation covering
    the entire foreground of its mask (bbox = tight box around all crack pixels,
    segmentation = RLE of the full binary mask).
  - Prompt is fixed to "crack" because the data loader derives query_text from the
    category name (categories = [{id:1, name:"crack"}]).
  - Images are NOT copied. COCO `file_name` stores the ABSOLUTE path to each source
    image; the loader (split_dir / file_name) opens it directly via pathlib.

Output layout (only JSON files are written; images stay in place):
    <out>/train/_annotations.coco.json
    <out>/valid/_annotations.coco.json
    <out>/test/_annotations.coco.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from tqdm import tqdm

# Source sub-folders under the unified root. Each has images/ and masks/.
SOURCES = ["BCL_NonSteel", "BCL_Steel", "CCSD", "LCW", "NCCD"]

# Image extensions to probe when matching a mask stem to its image.
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_image(images_dir: Path, stem: str):
    """Return the image path matching `stem`, trying common extensions."""
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def mask_to_annotation(mask_path: Path, image_size):
    """Load a binary mask and return (bbox_xywh, area, rle) or None if empty.

    `image_size` is (width, height) of the paired image. If the mask resolution
    differs, it is resized (nearest) to match so the RLE lines up with the image.
    """
    m = np.array(Image.open(mask_path).convert("L"))
    img_w, img_h = image_size
    if m.shape[0] != img_h or m.shape[1] != img_w:
        m = np.array(
            Image.fromarray(m).resize((img_w, img_h), Image.NEAREST)
        )

    fg = m > 127
    if not fg.any():
        return None  # negative image -> skip

    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]  # COCO xywh
    area = int(fg.sum())

    # Encode the full binary mask as RLE. pycocotools needs Fortran-order uint8.
    rle = mask_utils.encode(np.asfortranarray(fg.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
    return bbox, area, rle


def build_records(unified_root: Path):
    """Scan all sources and return a list of per-source record lists.

    Each record: {"file_name": abs_image_path, "width", "height", "bbox",
                  "area", "rle", "source"}.
    """
    by_source = {}
    for src in SOURCES:
        src_dir = unified_root / src
        images_dir = src_dir / "images"
        masks_dir = src_dir / "masks"
        if not masks_dir.is_dir():
            print(f"[skip] {src}: no masks dir at {masks_dir}")
            continue

        records = []
        mask_files = sorted(masks_dir.glob("*.png"))
        n_neg = 0
        n_noimg = 0
        for mask_path in tqdm(mask_files, desc=f"{src:12s}"):
            stem = mask_path.stem
            img_path = find_image(images_dir, stem)
            if img_path is None:
                n_noimg += 1
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            ann = mask_to_annotation(mask_path, (w, h))
            if ann is None:
                n_neg += 1
                continue
            bbox, area, rle = ann
            records.append({
                "file_name": str(img_path.resolve()),
                "width": w,
                "height": h,
                "bbox": bbox,
                "area": area,
                "rle": rle,
                "source": src,
            })
        print(f"[{src}] positives={len(records)} negatives_skipped={n_neg} "
              f"missing_image={n_noimg}")
        by_source[src] = records
    return by_source


def split_records(by_source, val_frac, test_frac, seed):
    """Stratified split per source so every dataset appears in each split."""
    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "test": []}
    for src, records in by_source.items():
        recs = records[:]
        rng.shuffle(recs)
        n = len(recs)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        splits["test"].extend(recs[:n_test])
        splits["valid"].extend(recs[n_test:n_test + n_val])
        splits["train"].extend(recs[n_test + n_val:])
    return splits


def write_coco(records, out_path: Path):
    """Write one COCO json (single category 'crack', one annotation per image)."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}],
    }
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
        })
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": rec["bbox"],
            "area": rec["area"],
            "segmentation": rec["rle"],
            "iscrowd": 0,
        })
        ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"  wrote {out_path}  (images={len(coco['images'])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unified-root", default="/workspace/nas_200/minkyung/unified",
                    help="Root containing BCL_NonSteel/, BCL_Steel/, CCSD/, LCW/, NCCD/")
    ap.add_argument("--out", default="/workspace/nas_200/minkyung/crack_coco",
                    help="Output dir for train/valid/test COCO jsons")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    unified_root = Path(args.unified_root)
    out = Path(args.out)

    by_source = build_records(unified_root)
    total = sum(len(v) for v in by_source.values())
    print(f"\nTotal positive image/mask pairs: {total}")

    splits = split_records(by_source, args.val_frac, args.test_frac, args.seed)
    for name in ["train", "valid", "test"]:
        write_coco(splits[name], out / name / "_annotations.coco.json")

    print("\nDone. Point the training config at:")
    print(f"  training.data_dir: \"{out}\"")


if __name__ == "__main__":
    main()
