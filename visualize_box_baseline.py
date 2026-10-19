"""Draw some results of the box prompt, so we can check them with our eyes.

For each object we show: the image, the GT border (green), the predicted mask (red)
and the box prompt (yellow). In the title there is IoU. It is only for this check,
the real IoU is in the evaluation part.

How to run (after run_box_baseline.py):
    python visualize_box_baseline.py --split development --n 8
Result:
    results/box_gt_<split>_examples.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pycocotools import mask as mask_utils

from dataset import COCOObjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--split", default="development")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=8, help="how many objects to draw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = Path(args.results_dir)
    stem = f"box_gt_{args.split}"
    df = pd.read_csv(res / f"{stem}.csv")
    # read the predicted masks
    rles = {}
    with open(res / f"{stem}_masks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rles[d["annotation_id"]] = d["rle"]

    # take some random objects and load them from COCO
    sample = df.sample(min(args.n, len(df)), random_state=args.seed)
    wanted = set(sample["annotation_id"])
    ds = COCOObjects(args.coco_root)
    records = [r for r in ds.iter_objects(sorted(sample["image_id"].unique().tolist()))
               if r.annotation_id in wanted]

    cols = 4
    rows = (len(records) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, records):
        row = sample[sample["annotation_id"] == r.annotation_id].iloc[0]
        rle = dict(rles[r.annotation_id])
        rle["counts"] = rle["counts"].encode("ascii")
        pred = mask_utils.decode(rle).astype(bool)
        gt = r.mask.astype(bool)
        union = (pred | gt).sum()
        # simple IoU, only for the title
        iou = (pred & gt).sum() / union if union else 0.0

        # red = predicted mask, green line = GT border
        show = r.image.copy()
        show[pred] = (0.5 * show[pred] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(show, contours, -1, (0, 255, 0), 2)
        ax.imshow(show)
        # yellow rectangle = the box we gave to the model
        x1, y1, x2, y2 = row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
        ax.set_title(f"{r.category_name}  area={r.area}\nIoU={iou:.2f}  score={row['score']:.2f}", fontsize=9)

    fig.tight_layout()
    out = res / f"{stem}_examples.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
