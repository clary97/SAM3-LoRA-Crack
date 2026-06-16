#!/usr/bin/env python3
"""
Build a TILED training set for the high-resolution / wide-scene datasets
(CCSD, LCW) where thin cracks are lost when whole images are downscaled to 1008.

Strategy (preserves the existing crack_coco train/valid split — no leakage):
  - Read crack_coco/{train,valid}/_annotations.coco.json.
  - For each image, decode its full-resolution mask from the COCO RLE.
  - CCSD / LCW images  -> cut into overlapping tiles; keep tiles that contain
    crack pixels (capped per image). Each kept tile is saved to disk and gets a
    "crack" annotation (whole tile-mask = 1 instance, RLE).
  - BCL_NonSteel / BCL_Steel / NCCD images -> kept whole (entry copied as-is,
    still referencing the original image path).

Only train/valid are tiled; the test split stays full-image (crack_coco/test)
so the tiled model can be compared fairly against v1/v2/v3 via a stitched eval.

Output: <out>/{train,valid}/_annotations.coco.json  (+ saved tile images)
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import pycocotools.mask as mask_utils
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

TILE_SOURCES = {"CCSD", "LCW"}   # only these get tiled


def source_of(fn):
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


def decode_mask(ann, h, w):
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    return np.zeros((h, w), dtype=bool)


def rle_of(mask_bool):
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)  # ensure the right/bottom edge is covered
    return pos


def make_tiles(img, mask, tile, stride, min_pixels, max_tiles):
    """Return list of (img_crop, mask_crop, crack_px) for positive tiles."""
    H, W = mask.shape
    out = []
    for y in tile_positions(H, tile, stride):
        for x in tile_positions(W, tile, stride):
            m = mask[y:y + tile, x:x + tile]
            px = int(m.sum())
            if px >= min_pixels:
                out.append((img[y:y + tile, x:x + tile], m, px))
    out.sort(key=lambda t: -t[2])      # densest cracks first
    return out[:max_tiles]


def process_split(split, coco_root, out_root, tile, stride, min_pixels, max_tiles):
    coco = json.load(open(coco_root / split / "_annotations.coco.json"))
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    img_dir = out_root / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    out = {"images": [], "annotations": [],
           "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}]}
    iid = aid = 1
    n_tiled = n_whole = n_tiles = 0

    for im in tqdm(coco["images"], desc=f"{split:5s}"):
        fn = im["file_name"]
        H, W = im["height"], im["width"]
        src = source_of(fn)
        anns = ann_by_img.get(im["id"], [])

        if src not in TILE_SOURCES:
            # keep whole image: copy entry + its annotations (re-id)
            out["images"].append({"id": iid, "file_name": fn, "width": W, "height": H})
            for a in anns:
                out["annotations"].append({
                    "id": aid, "image_id": iid, "category_id": 1,
                    "bbox": a["bbox"], "area": a["area"],
                    "segmentation": a["segmentation"], "iscrowd": 0})
                aid += 1
            iid += 1
            n_whole += 1
            continue

        # tile this image
        full_mask = np.zeros((H, W), dtype=bool)
        for a in anns:
            full_mask |= decode_mask(a, H, W)
        img = np.array(Image.open(fn).convert("RGB"))
        if img.shape[:2] != (H, W):  # safety: align mask to image
            full_mask = np.array(Image.fromarray(full_mask).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)).astype(bool)
            H, W = img.shape[:2]

        stem = Path(fn).stem
        for k, (ic, mc, _) in enumerate(make_tiles(img, full_mask, tile, stride, min_pixels, max_tiles)):
            # include the unique image id (iid) so stems shared across
            # sub-folders (e.g. LCW Train/Test both have "260") can't collide
            tile_path = img_dir / f"{src}_{stem}_{iid}_t{k}.jpg"
            Image.fromarray(ic).save(tile_path, quality=95)
            ys, xs = np.where(mc)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            out["images"].append({"id": iid, "file_name": str(tile_path.resolve()),
                                  "width": mc.shape[1], "height": mc.shape[0]})
            out["annotations"].append({
                "id": aid, "image_id": iid, "category_id": 1,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area": int(mc.sum()),
                "segmentation": rle_of(mc), "iscrowd": 0})
            iid += 1; aid += 1; n_tiles += 1
        n_tiled += 1

    out_file = out_root / split / "_annotations.coco.json"
    json.dump(out, open(out_file, "w"))
    print(f"[{split}] whole_images={n_whole}  tiled_images={n_tiled} -> tiles={n_tiles} "
          f"| total entries={len(out['images'])}  ({out_file})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-root", default="/workspace/nas_200/minkyung/crack_coco")
    ap.add_argument("--out", default="/workspace/nas_200/minkyung/crack_tiles")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--max-tiles", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    args = ap.parse_args()

    coco_root, out_root = Path(args.coco_root), Path(args.out)
    for split in args.splits:
        process_split(split, coco_root, out_root, args.tile, args.stride,
                      args.min_pixels, args.max_tiles)
    print("\nDone. Set training.data_dir to:", out_root)
    print("Note: evaluate on the FULL-image test set (crack_coco/test) via a stitched eval.")


if __name__ == "__main__":
    main()
