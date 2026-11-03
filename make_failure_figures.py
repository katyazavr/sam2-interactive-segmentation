"""Failure-case figure (GT / point / box) for chosen annotation ids.

    python make_failure_figures.py results data/coco results/failure_analysis/fig_failure_compact.png "475292,1917299,373276,1169777" "475292=(a) ...|..."
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

RES = Path(sys.argv[1])            # repo results dir
COCO_ROOT = Path(sys.argv[2])      # staged coco folder
OUT = Path(sys.argv[3])
IDS = [int(x) for x in sys.argv[4].split(",")]
TITLES = dict(kv.split("=", 1) for kv in sys.argv[5].split("|")) if len(sys.argv) > 5 else {}

coco = COCO(str(COCO_ROOT / "annotations" / "instances_val2017.json"))
pt = pd.read_csv(RES / "evaluation_test/point_center_evaluation_objects.csv").set_index("annotation_id")
bx = pd.read_csv(RES / "evaluation_test/box_gt_evaluation_objects.csv").set_index("annotation_id")


def load_rles(name):
    d = {}
    with open(RES / name, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if int(r["annotation_id"]) in IDS:
                rle = dict(r["rle"]); rle["counts"] = rle["counts"].encode("ascii")
                d[int(r["annotation_id"])] = mask_utils.decode(rle).astype(bool)
    return d


pm, bm = load_rles("point_center_test_masks.jsonl"), load_rles("box_gt_test_masks.jsonl")


def overlay(img, mask, color, gt):
    show = img.copy()
    show[mask] = (0.45 * show[mask] + 0.55 * np.array(color)).astype(np.uint8)
    for c, w in ((gt, 2),):
        cs, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(show, cs, -1, (0, 255, 0), w)
    return show


rows = len(IDS)
fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.9 * rows), squeeze=False)
for r, aid in enumerate(IDS):
    ann = coco.anns[aid]
    img = np.array(__import__("PIL.Image", fromlist=["x"]).open(COCO_ROOT / "val2017" / f"{ann['image_id']:012d}.jpg").convert("RGB"))
    gt = coco.annToMask(ann).astype(bool)
    x1, y1, x2, y2 = bx.loc[aid, ["box_x1", "box_y1", "box_x2", "box_y2"]].astype(float)
    ys, xs = np.where(gt | pm[aid] | bm[aid])
    bx0, bx1, by0, by1 = min(xs.min(), x1), max(xs.max(), x2), min(ys.min(), y1), max(ys.max(), y2)
    side = max(bx1 - bx0, by1 - by0) * 1.5 + 30
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    X0, X1, Y0, Y1 = int(cx - side / 2), int(cx + side / 2), int(cy - side / 2), int(cy + side / 2)
    pad = max(0, -X0, -Y0, X1 - img.shape[1], Y1 - img.shape[0])
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    gt = np.pad(gt, pad); pm[aid] = np.pad(pm[aid], pad); bm[aid] = np.pad(bm[aid], pad)
    X0 += pad; X1 += pad; Y0 += pad; Y1 += pad; x1 += pad; x2 += pad; y1 += pad; y2 += pad
    px_off = pad
    cat = pt.loc[aid, "category_name"]
    panels = [
        ("Ground truth", overlay(img, gt, (0, 255, 0), gt), None),
        (f"Point  IoU={pt.loc[aid,'iou']:.2f}  score={pt.loc[aid,'score']:.2f}", overlay(img, pm[aid], (255, 0, 0), gt), "pt"),
        (f"Box  IoU={bx.loc[aid,'iou']:.2f}  score={bx.loc[aid,'score']:.2f}", overlay(img, bm[aid], (0, 90, 255), gt), "bx"),
    ]
    for c, (title, im, kind) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(im[Y0:Y1, X0:X1])
        if kind == "pt":
            ax.scatter([pt.loc[aid, "point_x"] + px_off - X0], [pt.loc[aid, "point_y"] + px_off - Y0], marker="*", s=220, c="yellow", edgecolors="black", zorder=5)
        if kind == "bx":
            ax.add_patch(plt.Rectangle((x1 - X0, y1 - Y0), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=2))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[r, 0].text(0, 1.14, f"{TITLES.get(str(aid), '')}  [{cat}, id {aid}, area {int(pt.loc[aid,'area'])} px]",
                    transform=axes[r, 0].transAxes, fontsize=10, fontweight="bold", va="bottom")
fig.subplots_adjust(left=0.01,right=0.99,top=0.97,bottom=0.01,wspace=0.03,hspace=0.28)
fig.savefig(OUT, dpi=110)
print("saved", OUT)
