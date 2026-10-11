"""Dataset preparation for the SAM 2 interactive segmentation project.

COCO 2017 val2017 -> validated per-object records (image, binary mask, metadata)
+ reproducible image-level 60/20/20 split saved to splits.json.

Rules (from the project plan):
  - exclude iscrowd = 1
  - keep all 80 categories, no minimum object-size filter
  - masks decoded with COCO.annToMask()
  - reject only technically invalid samples: missing image, wrong mask shape, empty mask
  - keep image_id and annotation_id for traceability

Usage:
  python dataset.py --coco-root data/coco --out-dir data/processed
Expected layout:
  data/coco/val2017/*.jpg
  data/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO

SPLIT_NAMES = ("development", "validation", "test")
SPLIT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_SEED = 42


@dataclass
class ObjectRecord:
    image_id: int
    annotation_id: int
    file_name: str
    image: np.ndarray        # H x W x 3, uint8, RGB
    mask: np.ndarray         # H x W, uint8 {0, 1}
    category_id: int
    category_name: str
    area: int                # number of mask pixels
    coco_bbox: list[float]   # original COCO bbox [x, y, w, h]
    coco_area: float         # original COCO annotation area


class COCOObjects:
    """Lazy access to COCO val2017 object instances (images are not kept in memory)."""

    def __init__(self, coco_root: str | Path, split: str = "val2017"):
        self.root = Path(coco_root)
        self.img_dir = self.root / split
        ann_file = self.root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        self.cat_names = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}

    # loading
    def _load_image(self, file_name: str) -> np.ndarray | None:
        path = self.img_dir / file_name
        if not path.is_file():
            return None
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))  # some COCO images are grayscale

    def _build_record(self, ann: dict, image: np.ndarray, img_info: dict):
        """Return (ObjectRecord, None) or (None, reject_reason)."""
        mask = self.coco.annToMask(ann).astype(np.uint8)
        if mask.shape != image.shape[:2]:
            return None, "mask_shape_mismatch"
        if mask.shape != (img_info["height"], img_info["width"]):
            return None, "mask_shape_mismatch"
        area = int(mask.sum())
        if area == 0:
            return None, "empty_mask"
        rec = ObjectRecord(
            image_id=ann["image_id"],
            annotation_id=ann["id"],
            file_name=img_info["file_name"],
            image=image,
            mask=mask,
            category_id=ann["category_id"],
            category_name=self.cat_names[ann["category_id"]],
            area=area,
            coco_bbox=list(ann["bbox"]),
            coco_area=float(ann["area"]),
        )
        return rec, None

    # iteration
    def iter_images(self, image_ids: list[int] | None = None) -> Iterator[tuple[int, list[ObjectRecord], dict]]:
        """Yield (image_id, valid records of that image, reject-reason counts).

        Grouped by image so the SAM 2 predictor can call set_image() once per image.
        """
        ids = sorted(self.coco.getImgIds()) if image_ids is None else image_ids
        for image_id in ids:
            img_info = self.coco.loadImgs(image_id)[0]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=False))
            rejects: dict[str, int] = defaultdict(int)
            if not anns:
                yield image_id, [], rejects
                continue
            image = self._load_image(img_info["file_name"])
            if image is None:
                rejects["missing_image"] += len(anns)
                yield image_id, [], rejects
                continue
            records = []
            for ann in anns:
                if ann.get("iscrowd", 0) == 1:  # safety net, already filtered by getAnnIds
                    continue
                rec, reason = self._build_record(ann, image, img_info)
                if rec is None:
                    rejects[reason] += 1
                else:
                    records.append(rec)
            yield image_id, records, rejects

    def iter_objects(self, image_ids: list[int] | None = None) -> Iterator[ObjectRecord]:
        for _, records, _ in self.iter_images(image_ids):
            yield from records


# validation + split
def validate_and_index(ds: COCOObjects) -> tuple[pd.DataFrame, dict]:
    """Decode every retained annotation once, collect an index of valid objects and reject stats."""
    rows, rejects_total = [], defaultdict(int)
    n_crowd = sum(1 for a in ds.coco.anns.values() if a.get("iscrowd", 0) == 1)
    for image_id, records, rejects in ds.iter_images():
        for k, v in rejects.items():
            rejects_total[k] += v
        for r in records:
            rows.append({
                "image_id": r.image_id,
                "annotation_id": r.annotation_id,
                "file_name": r.file_name,
                "category_id": r.category_id,
                "category_name": r.category_name,
                "area": r.area,
                "coco_area": r.coco_area,
                "coco_bbox": json.dumps(r.coco_bbox),
                "height": r.mask.shape[0],
                "width": r.mask.shape[1],
            })
    df = pd.DataFrame(rows)
    stats = {
        "images_total": len(ds.coco.getImgIds()),
        "annotations_total": len(ds.coco.anns),
        "excluded_iscrowd": n_crowd,
        "rejected": dict(rejects_total),
        "objects_valid": len(df),
        "images_with_valid_objects": int(df["image_id"].nunique()) if len(df) else 0,
        "categories": int(df["category_id"].nunique()) if len(df) else 0,
    }
    return df, stats


def make_splits(image_ids: list[int], seed: int = DEFAULT_SEED) -> dict[str, list[int]]:
    """Image-level 60/20/20 split: all objects of one image stay in the same subset."""
    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_dev = int(round(n * SPLIT_RATIOS[0]))
    n_val = int(round(n * SPLIT_RATIOS[1]))
    parts = (ids[:n_dev], ids[n_dev:n_dev + n_val], ids[n_dev + n_val:])
    return {name: sorted(p) for name, p in zip(SPLIT_NAMES, parts)}


def load_splits(path: str | Path) -> dict[str, list[int]]:
    with open(path) as f:
        data = json.load(f)
    return {k: data[k] for k in SPLIT_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = COCOObjects(args.coco_root)
    df, stats = validate_and_index(ds)

    splits = make_splits(df["image_id"].tolist(), args.seed)
    split_of = {i: name for name, ids in splits.items() for i in ids}
    df["split"] = df["image_id"].map(split_of)

    df.to_csv(out / "objects.csv", index=False)
    with open(out / "splits.json", "w") as f:
        json.dump({"seed": args.seed, "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)), **splits}, f)
    stats["split_images"] = {k: len(v) for k, v in splits.items()}
    stats["split_objects"] = df["split"].value_counts().to_dict()
    with open(out / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
