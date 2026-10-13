# Interactive Object Segmentation with SAM 2

Stage 1 (dataset): COCO 2017 val2017 → validated per-object records + reproducible image-level split.
Stage 2 (model): pretrained SAM 2.1 (hiera base+), baseline with one positive center point.

## Setup
Python 3.12+ is needed.
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download data
```bash
python scripts/download_coco.py
```
Result: `data/coco/val2017/*.jpg` and `data/coco/annotations/instances_val2017.json`.

## Prepare dataset and split
```bash
python dataset.py --coco-root data/coco --out-dir data/processed
```
Outputs in `data/processed/`:
- `objects.csv` — one row per valid object (image_id, annotation_id, category, area, split, ...)
- `splits.json` — image IDs for development / validation / test (60/20/20, seed 42)
- `dataset_stats.json` — counts and rejection reasons

## Usage from other scripts
```python
from dataset import COCOObjects, load_splits

ds = COCOObjects("data/coco")
splits = load_splits("data/processed/splits.json")
for image_id, records, _ in ds.iter_images(splits["test"]):
    for r in records:
        r.mask, r.category_name, r.area, r.annotation_id
```

## Download SAM 2 checkpoint
```bash
python scripts/download_sam2_checkpoint.py
```
Result: `checkpoints/sam2.1_hiera_base_plus.pt` (~320 MB, not in git).

## Run point baseline
Quick check on 10 images and a picture with examples:
```bash
python run_point_baseline.py --split development --max-images 10
python visualize_point_baseline.py --split development --n 8
```
Final run (on GPU):
```bash
python run_point_baseline.py --split test
```
Outputs in `results/`:
- `point_center_<split>.csv` - one row per object (image_id, annotation_id, point, score, encoder_ms, prompt_ms, ...)
- `point_center_<split>_masks.jsonl` - predicted masks as COCO RLE (for IoU / Dice)
- `point_center_<split>_run_info.json` - model, settings, device and library versions
- `point_center_<split>_examples.png` - examples from `visualize_point_baseline.py`

## Model usage from other scripts
```python
from sam2_model import load_predictor, center_point, Sam2Runner

runner = Sam2Runner(load_predictor())
encoder_ms = runner.set_image(image)            # once per image
out = runner.predict_point(center_point(mask))
out.mask, out.score, out.prompt_ms
```

## Preparation rules
- `iscrowd = 1` annotations excluded; all 80 categories kept; no minimum object size
- masks decoded with `COCO.annToMask()`
- rejected only: missing image, mask shape mismatch, empty mask
- split is by image (all objects of an image stay in one subset)

## Model settings
- SAM 2.1 `hiera_base_plus`, pretrained, no training
- point = mask pixel farthest from the object border (distance transform)
- SAM 2 returns 3 masks, the one with the highest score is kept
- `set_image()` once per image; encoder and prompt time are measured separately, after 3 warm-up runs
- Mac (`mps`) is only for checks, final latency is measured on GPU

## Data sources
- Dataset: MS COCO 2017 — https://cocodataset.org/#download
- 2017 Val images [5K/1GB]: http://images.cocodataset.org/zips/val2017.zip
- 2017 Train/Val annotations [241MB]: http://images.cocodataset.org/annotations/annotations_trainval2017.zip (`instances_val2017.json` is taken from this archive)
- License (see https://cocodataset.org/#termsofuse):
  - Annotations and website: Creative Commons Attribution 4.0 License (https://creativecommons.org/licenses/by/4.0/legalcode), owned by the COCO Consortium.
  - Images: the COCO Consortium does not own the copyright; use must follow the Flickr Terms of Use (https://www.flickr.com/creativecommons/).
  - Software (COCO API): BSD-style license, Copyright (c) 2015, COCO Consortium.
- Citation: Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C. L. "Microsoft COCO: Common Objects in Context". ECCV 2014. arXiv:1405.0312
