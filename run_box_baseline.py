"""Run SAM 2.1 with a COCO ground-truth bounding-box prompt.

Example (from project root):
    python run_box_baseline.py --split development --max-images 2
    python run_box_baseline.py --split development --max-images 50

The box is supplied from the COCO annotation (oracle prompt), not detected by
another model. Compare with point_center using the same image IDs and objects.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
from sam2_model import CHECKPOINT_NAME, DEFAULT_CHECKPOINT, MODEL_CONFIG, Sam2Runner, load_predictor

CONDITION = "box_gt"


def coco_box_xyxy(coco_bbox, image_shape):
    """Convert COCO [x,y,width,height] to clipped SAM2 [x1,y1,x2,y2]."""
    x, y, w, h = map(float, coco_bbox)
    image_h, image_w = image_shape[:2]
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid COCO box: {coco_bbox}")
    box = np.array([
        np.clip(x, 0, image_w),
        np.clip(y, 0, image_h),
        np.clip(x + w, 0, image_w),
        np.clip(y + h, 0, image_h),
    ], dtype=np.float32)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"COCO box outside image: {coco_bbox}")
    return box


def encode_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def predict_box(runner, box):
    """Predict 3 masks from one box; select highest-scoring mask and time prompt."""
    with torch.inference_mode():
        with runner._autocast():
            runner._sync()
            start = time.perf_counter()
            masks, scores, _ = runner.predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=True
            )
            runner._sync()
    prompt_ms = (time.perf_counter() - start) * 1000
    best = int(np.argmax(scores))
    return (masks[best] > 0).astype(np.uint8), float(scores[best]), best, prompt_ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco-root", default="data/coco")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--split", default="development", choices=["development", "validation", "test"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if args.max_images is not None and args.max_images < 1:
        ap.error("--max-images must be positive")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{CONDITION}_{args.split}"
    image_ids = load_splits(args.splits)[args.split]
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]

    ds = COCOObjects(args.coco_root)
    runner = Sam2Runner(load_predictor(args.checkpoint, args.device), precision=args.precision)
    print(f"device={runner.device} precision={args.precision} images={len(image_ids)}")

    rows = []
    warmed_up = False
    started = time.perf_counter()
    with (out / f"{stem}_masks.jsonl").open("w", encoding="utf-8") as masks_file:
        for i, (image_id, records, _) in enumerate(ds.iter_images(image_ids), 1):
            if not records:
                continue
            image = records[0].image
            if not warmed_up:
                runner.warmup(image, args.warmup)
                warmed_up = True
            encoder_ms = runner.set_image(image)
            for rec in records:
                box = coco_box_xyxy(rec.coco_bbox, image.shape)
                mask, score, mask_index, prompt_ms = predict_box(runner, box)
                rows.append({
                    "image_id": rec.image_id,
                    "annotation_id": rec.annotation_id,
                    "category_name": rec.category_name,
                    "area": rec.area,
                    "split": args.split,
                    "condition": CONDITION,
                    "box_x1": float(box[0]),
                    "box_y1": float(box[1]),
                    "box_x2": float(box[2]),
                    "box_y2": float(box[3]),
                    "mask_index": mask_index,
                    "score": round(score, 5),
                    "pred_area": int(mask.sum()),
                    "encoder_ms": round(encoder_ms, 3),
                    "prompt_ms": round(prompt_ms, 3),
                })
                masks_file.write(json.dumps({
                    "annotation_id": rec.annotation_id,
                    "condition": CONDITION,
                    "rle": encode_rle(mask),
                }) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(image_ids)} images, {len(rows)} objects")

    seconds = time.perf_counter() - started
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{stem}.csv", index=False)
    import sam2
    info = {
        "condition": CONDITION,
        "prompt_source": "COCO ground-truth annotation bounding box (oracle)",
        "split": args.split,
        "max_images": args.max_images,
        "images": int(df.image_id.nunique()) if len(df) else 0,
        "objects": len(df),
        "checkpoint": CHECKPOINT_NAME,
        "model_config": MODEL_CONFIG,
        "multimask_output": True,
        "mask_selection": "highest predicted score",
        "precision": args.precision,
        "device": str(runner.device),
        "warmup_runs": args.warmup,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sam2_path": str(Path(sam2.__file__).parent),
        "platform": platform.platform(),
        "total_seconds": round(seconds, 1),
    }
    (out / f"{stem}_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if len(df):
        enc = df.drop_duplicates("image_id")["encoder_ms"]
        print(f"objects: {len(df)}  time: {seconds:.1f} s")
        print(f"encoder_ms  mean {enc.mean():.1f}  median {enc.median():.1f}")
        print(f"prompt_ms   mean {df['prompt_ms'].mean():.1f}  median {df['prompt_ms'].median():.1f}")
        print(f"mask_index counts: {df['mask_index'].value_counts().sort_index().to_dict()}")
    print(f"saved to {out}/{stem}.*")


if __name__ == "__main__":
    main()
