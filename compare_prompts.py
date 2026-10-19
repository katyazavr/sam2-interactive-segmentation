"""Compare box prompt with the point baseline on the same objects.

Run after evaluate_baseline.py was run for both conditions:
    python evaluate_baseline.py --split development --condition point_center
    python evaluate_baseline.py --split development --condition box_gt
    python compare_prompts.py --split development

Output in results/evaluation_<split>/:
    comparison_<split>.csv    the comparison table (Metric / Point Prompt / Box Prompt)
    comparison_<split>.json   same numbers + paired differences (box - point)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POINT, BOX = "point_center", "box_gt"


def load(eval_dir, condition):
    path = Path(eval_dir) / f"{condition}_evaluation_objects.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run evaluate_baseline.py --condition {condition} first.")
    return pd.read_csv(path)


def check_same_objects(point, box):
    """Both conditions must cover exactly the same objects (paired comparison)."""
    for name, df in (("point", point), ("box", box)):
        if df.annotation_id.duplicated().any():
            raise ValueError(f"Duplicate annotation_id in {name} results")
    if set(point.annotation_id) != set(box.annotation_id):
        only_p = len(set(point.annotation_id) - set(box.annotation_id))
        only_b = len(set(box.annotation_id) - set(point.annotation_id))
        raise ValueError(f"Different objects: {only_p} only in point, {only_b} only in box")
    if set(point.image_id) != set(box.image_id):
        raise ValueError("Different images in point and box results")


def bootstrap_ci(values, n_boot=10000, seed=42):
    """95% bootstrap confidence interval of the mean of paired differences."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def latency(df):
    enc = df.drop_duplicates("image_id")["encoder_ms"].astype(float)   # once per image
    prompt = df["prompt_ms"].astype(float)                              # once per object
    return enc, prompt


def compare(point, box):
    check_same_objects(point, box)
    m = point.merge(box, on="annotation_id", suffixes=("_point", "_box"))
    d_iou = (m.iou_box - m.iou_point).to_numpy()
    d_dice = (m.dice_box - m.dice_point).to_numpy()
    paired = {
        "n_objects": int(len(m)),
        "n_images": int(point.image_id.nunique()),
        "mean_delta_iou": float(d_iou.mean()),
        "median_delta_iou": float(np.median(d_iou)),
        "delta_iou_ci95": bootstrap_ci(d_iou),
        "mean_delta_dice": float(d_dice.mean()),
        "median_delta_dice": float(np.median(d_dice)),
        "delta_dice_ci95": bootstrap_ci(d_dice),
        "box_better_share": float((d_iou > 1e-9).mean()),
        "equal_share": float((np.abs(d_iou) <= 1e-9).mean()),
        "point_better_share": float((d_iou < -1e-9).mean()),
    }
    p_enc, p_prompt = latency(point)
    b_enc, b_prompt = latency(box)
    rows = [
        ("Mean IoU", point.iou.mean(), box.iou.mean()),
        ("Median IoU", point.iou.median(), box.iou.median()),
        ("Std IoU", point.iou.std(ddof=0), box.iou.std(ddof=0)),
        ("Mean Dice", point.dice.mean(), box.dice.mean()),
        ("Median Dice", point.dice.median(), box.dice.median()),
        ("Std Dice", point.dice.std(ddof=0), box.dice.std(ddof=0)),
        ("Encoder latency, mean (ms)", p_enc.mean(), b_enc.mean()),
        ("Encoder latency, median (ms)", p_enc.median(), b_enc.median()),
        ("Prompt latency, mean (ms)", p_prompt.mean(), b_prompt.mean()),
        ("Prompt latency, median (ms)", p_prompt.median(), b_prompt.median()),
        ("Prompt latency, max (ms)", p_prompt.max(), b_prompt.max()),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "Point Prompt", "Box Prompt"])
    table["Difference (box - point)"] = table["Box Prompt"] - table["Point Prompt"]
    return table, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development", choices=["development", "validation", "test"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--eval-dir", default=None, help="default: <results-dir>/evaluation_<split>")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir or Path(args.results_dir) / f"evaluation_{args.split}")

    table, paired = compare(load(eval_dir, POINT), load(eval_dir, BOX))
    table.round(4).to_csv(eval_dir / f"comparison_{args.split}.csv", index=False)
    out = {"table": table.round(4).to_dict(orient="records"), "paired": paired}
    (eval_dir / f"comparison_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(table.round(4).to_string(index=False))
    print(json.dumps(paired, indent=2))
    print(f"Saved to {eval_dir}")


if __name__ == "__main__":
    main()