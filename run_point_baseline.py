"""Run the baseline (one positive point in the center) with SAM 2.1 on one split.

How to run (from the main project folder):
    # small test on 10 images
    python run_point_baseline.py --split development --max-images 10
    # full run for the final results
    python run_point_baseline.py --split test

Files in results/:
    point_center_<split>.csv            one row for each object (ids, point, score, time)
    point_center_<split>_masks.jsonl    predicted masks in COCO RLE format (for IoU and Dice)
    point_center_<split>_run_info.json  settings and computer info
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # for Mac: if mps can not do something, use CPU

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from dataset import COCOObjects, load_splits
from sam2_model import (CHECKPOINT_NAME, DEFAULT_CHECKPOINT, MODEL_CONFIG, Sam2Runner,
                        center_point, load_predictor)

CONDITION = "point_center"


def encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def run_info(args, runner: Sam2Runner, n_images: int, n_objects: int, seconds: float) -> dict:
    import sam2
    dev = runner.device
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    return {
        "condition": CONDITION,
        "split": args.split,
        "max_images": args.max_images,
        "images": n_images,
        "objects": n_objects,
        "checkpoint": CHECKPOINT_NAME,
        "model_config": MODEL_CONFIG,
        "multimask_output": True,
        "mask_selection": "highest predicted score",
        "precision": args.precision,
        "device": str(dev),
        "gpu": gpu,
        "warmup_runs": args.warmup,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sam2_path": str(Path(sam2.__file__).parent),
        "platform": platform.platform(),
        "total_seconds": round(seconds, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--split", default="development", choices=["development", "validation", "test"])
    ap.add_argument("--max-images", type=int, default=None, help="take only the first N images (for a small test)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{CONDITION}_{args.split}"

    image_ids = load_splits(args.splits)[args.split]
    if args.max_images:
        image_ids = image_ids[:args.max_images]

    ds = COCOObjects(args.coco_root)
    runner = Sam2Runner(load_predictor(args.checkpoint, args.device), precision=args.precision)
    print(f"device={runner.device} precision={args.precision} images={len(image_ids)}")

    rows, warmed_up = [], False
    t_start = time.perf_counter()
    with open(out / f"{stem}_masks.jsonl", "w") as f_masks:
        for i, (image_id, records, _) in enumerate(ds.iter_images(image_ids), 1):
            if not records:
                continue
            image = records[0].image
            if not warmed_up:
                runner.warmup(image, args.warmup)
                warmed_up = True
            encoder_ms = runner.set_image(image)          # one time for each image
            for r in records:
                x, y = center_point(r.mask)
                res = runner.predict_point((x, y))
                rows.append({
                    "image_id": r.image_id,
                    "annotation_id": r.annotation_id,
                    "category_name": r.category_name,
                    "area": r.area,
                    "split": args.split,
                    "condition": CONDITION,
                    "point_x": x,
                    "point_y": y,
                    "mask_index": res.mask_index,
                    "score": round(res.score, 5),
                    "pred_area": int(res.mask.sum()),
                    "encoder_ms": round(encoder_ms, 3),   # the same for all objects in one image
                    "prompt_ms": round(res.prompt_ms, 3),
                })
                f_masks.write(json.dumps({"annotation_id": r.annotation_id,
                                          "condition": CONDITION,
                                          "rle": encode_rle(res.mask)}) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(image_ids)} images, {len(rows)} objects")
    seconds = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{stem}.csv", index=False)
    info = run_info(args, runner, df["image_id"].nunique() if len(df) else 0, len(df), seconds)
    with open(out / f"{stem}_run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    if len(df):
        enc = df.drop_duplicates("image_id")["encoder_ms"]
        print(f"objects: {len(df)}  time: {seconds:.1f} s")
        print(f"encoder_ms  mean {enc.mean():.1f}  median {enc.median():.1f}")
        print(f"prompt_ms   mean {df['prompt_ms'].mean():.1f}  median {df['prompt_ms'].median():.1f}")
        print(f"mask_index counts: {df['mask_index'].value_counts().sort_index().to_dict()}")
    print(f"saved to {out}/{stem}.*")


if __name__ == "__main__":
    main()
