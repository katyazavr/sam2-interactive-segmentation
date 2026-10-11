# Interactive Object Segmentation with SAM 2

Stage 1 (dataset): COCO 2017 val2017 → validated per-object records + reproducible image-level split.

## Setup
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

## Preparation rules
- `iscrowd = 1` annotations excluded; all 80 categories kept; no minimum object size
- masks decoded with `COCO.annToMask()`
- rejected only: missing image, mask shape mismatch, empty mask
- split is by image (all objects of an image stay in one subset)

## Data sources
- Dataset: MS COCO 2017 — https://cocodataset.org/#download
- 2017 Val images [5K/1GB]: http://images.cocodataset.org/zips/val2017.zip
- 2017 Train/Val annotations [241MB]: http://images.cocodataset.org/annotations/annotations_trainval2017.zip (`instances_val2017.json` is taken from this archive)
- License (see https://cocodataset.org/#termsofuse):
  - Annotations and website: Creative Commons Attribution 4.0 License (https://creativecommons.org/licenses/by/4.0/legalcode), owned by the COCO Consortium.
  - Images: the COCO Consortium does not own the copyright; use must follow the Flickr Terms of Use (https://www.flickr.com/creativecommons/).
  - Software (COCO API): BSD-style license, Copyright (c) 2015, COCO Consortium.
- Citation: Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C. L. "Microsoft COCO: Common Objects in Context". ECCV 2014. arXiv:1405.0312
