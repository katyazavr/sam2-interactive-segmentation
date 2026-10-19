# Baseline Evaluation (Arina Agafonova)

This evaluation uses the outputs of `run_point_baseline.py` and `run_box_baseline.py`, with no additional model inference.

## Commands

1. Generate predictions (requires COCO images, annotations and SAM 2.1 checkpoint):
   `python run_point_baseline.py --split development --max-images 50`
   `python run_box_baseline.py --split development --max-images 50`
2. Evaluate each condition:
   `python evaluate_baseline.py --split development --condition point_center`
   `python evaluate_baseline.py --split development --condition box_gt`
3. Compare box with point and build the report table and summary:
   `python compare_prompts.py --split development`
4. Tests: `python -m unittest test_evaluation test_compare -v`

For final test results use `--split test` in all commands (without `--max-images`).

## Inputs

- `results/point_center_<split>.csv` and `results/box_gt_<split>.csv` containing annotation IDs and encoder/prompt timings.
- `results/point_center_<split>_masks.jsonl` and `results/box_gt_<split>_masks.jsonl` containing predicted COCO RLE masks.
- `data/coco/annotations/instances_val2017.json` and `data/coco/val2017/` for ground truth.

## Outputs

All files are written to `results/evaluation_<split>/`.

- `<condition>_evaluation_objects.csv`: one row per object with IoU, Dice, prompt and encoder latency, metadata (`<condition>` is `point_center` or `box_gt`).
- `<condition>_evaluation_summary.csv` and `.json`: summary for one condition.
- `comparison_<split>.csv` and `.json`: point vs box table and paired differences (ΔIoU, ΔDice, bootstrap 95% CI, share of objects where each prompt is better).
- `comparison_summary.md`: report table and draft summary text

IoU = intersection / union; Dice = 2*intersection / (predicted area + GT area). Encoder timing is aggregated once per image (not once per object), while prompt timing is aggregated per object. This is a pretrained model; no training or fine-tuning is performed.

**Results are not populated until the real baseline has run.** Do not report test metrics from a 10-image development smoke test as final results. The `box_gt` condition uses the COCO ground-truth bounding box as an oracle prompt.