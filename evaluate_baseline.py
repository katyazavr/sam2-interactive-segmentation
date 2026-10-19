"""Evaluate SAM 2 predictions (point_center or box_gt) against COCO ground-truth masks.

Run after run_point_baseline.py / run_box_baseline.py. No model or checkpoint needed.

    python evaluate_baseline.py --split development --condition point_center
    python evaluate_baseline.py --split development --condition box_gt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def iou_dice(pred, gt):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: prediction {pred.shape}, ground truth {gt.shape}")
    intersection = np.count_nonzero(pred & gt)
    pred_count, gt_count = np.count_nonzero(pred), np.count_nonzero(gt)
    union = pred_count + gt_count - intersection
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / (pred_count + gt_count) if pred_count + gt_count else 1.0
    return float(iou), float(dice)


def evaluate(csv_path, masks_path, coco_root, out_dir, condition='point_center'):
    from pycocotools import mask as mask_utils
    from dataset import COCOObjects
    baseline = pd.read_csv(csv_path)
    required = {'image_id', 'annotation_id', 'encoder_ms', 'prompt_ms'}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {sorted(missing)}")
    if baseline.annotation_id.duplicated().any():
        raise ValueError("Duplicate annotation_id in baseline CSV")
    baseline = baseline.set_index('annotation_id', drop=False)
    rles = {}
    with open(masks_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            aid = int(record['annotation_id'])
            if aid in rles:
                raise ValueError(f"Duplicate mask for annotation_id={aid}")
            rles[aid] = record['rle']
    missing_masks = set(baseline.index) - set(rles)
    extra_masks = set(rles) - set(baseline.index)
    if missing_masks or extra_masks:
        raise ValueError(f"CSV/RLE mismatch: {len(missing_masks)} missing masks, {len(extra_masks)} extra masks")

    ds = COCOObjects(coco_root)
    rows = []
    image_ids = baseline.image_id.drop_duplicates().astype(int).tolist()
    seen = set()
    for image_id, records, _ in ds.iter_images(image_ids):
        for rec in records:
            aid = rec.annotation_id
            if aid not in rles:
                continue
            pred = mask_utils.decode(rles[aid])
            iou, dice = iou_dice(pred, rec.mask)
            row = baseline.loc[aid].to_dict()
            if int(row['image_id']) != int(image_id):
                raise ValueError(f"Image ID mismatch for annotation {aid}")
            row['iou'] = iou
            row['dice'] = dice
            rows.append(row)
            seen.add(aid)
    if seen != set(rles):
        raise ValueError(f"Could not locate {len(set(rles) - seen)} GT objects in COCO data")
    if not rows:
        raise ValueError('No objects evaluated')
    results = pd.DataFrame(rows).sort_values(['image_id', 'annotation_id'])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f'{condition}_evaluation_objects.csv', index=False)

    # Encoder is measured once per image, Never average repeated encoder values per object.
    enc = results.drop_duplicates('image_id')['encoder_ms'].astype(float)
    prompt = results['prompt_ms'].astype(float)
    summary = {
        'n_images': int(results.image_id.nunique()),
        'n_objects': int(len(results)),
        'mean_iou': float(results.iou.mean()),
        'median_iou': float(results.iou.median()),
        'std_iou': float(results.iou.std(ddof=0)),
        'mean_dice': float(results.dice.mean()),
        'median_dice': float(results.dice.median()),
        'std_dice': float(results.dice.std(ddof=0)),
        'mean_encoder_ms_per_image': float(enc.mean()),
        'median_encoder_ms_per_image': float(enc.median()),
        'mean_prompt_ms_per_object': float(prompt.mean()),
        'median_prompt_ms_per_object': float(prompt.median()),
    }
    (out / f'{condition}_evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([summary]).to_csv(out / f'{condition}_evaluation_summary.csv', index=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation files to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='development', choices=['development', 'validation', 'test'])
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--coco-root', default='data/coco')
    parser.add_argument('--condition', default='point_center', choices=['point_center', 'box_gt'])
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    stem = f'{args.condition}_{args.split}'
    results_dir = Path(args.results_dir)
    evaluate(results_dir / f'{stem}.csv', results_dir / f'{stem}_masks.jsonl',
             args.coco_root, args.out_dir or str(results_dir / f'evaluation_{args.split}'),
             condition=args.condition)


if __name__ == '__main__':
    main()