# Baseline Evaluation (Arina Agafonova)

This evaluation uses the existing `run_point_baseline.py` outputs, with no additional model inference.

## Commands

1. Generate predictions (requires COCO images, annotations and SAM 2.1 checkpoint):
   `python run_point_baseline.py --split development --max-images 10`
2. Evaluate: `python evaluate_baseline.py --split development`
3. Tests: `python -m unittest test_evaluation -v`

For final test results use `--split test` in both commands (without `--max-images`).

## Inputs

- `results/point_center_<split>.csv` containing annotation IDs and encoder/prompt timings.
- `results/point_center_<split>_masks.jsonl` containing predicted COCO RLE masks.
- `data/coco/annotations/instances_val2017.json` and `data/coco/val2017/` for ground truth.

## Outputs

- `results/evaluation_<split>/baseline_evaluation_objects.csv`: one row per object with IoU, Dice, prompt and encoder latency, metadata.
- `results/evaluation_<split>/baseline_evaluation_summary.csv`: one-row summary.
- `results/evaluation_<split>/baseline_evaluation_summary.json`: machine-readable summary.

IoU = intersection / union; Dice = 2*intersection / (predicted area + GT area). Encoder timing is aggregated once per image (not once per object), while prompt timing is aggregated per object. This is a pretrained model; no training or fine-tuning is performed.

**Results are not populated until the real baseline has run.** Do not report test metrics from a 10-image development smoke test as final results.
