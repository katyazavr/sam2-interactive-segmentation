# Interactive Object Segmentation with SAM 2.1

**Point vs. Box Prompts on COCO 2017**  
Innopolis University · Computer Vision Project · 2026

**Team:** Sabina Yamilova, Anna Ksel, Arina Agafonova, Ekaterina Salazkina

## Overview

This project evaluates how the type of user prompt affects the segmentation quality of a **pretrained, frozen SAM 2.1 Hiera Base+** model. We compare a **single positive point** with a **ground-truth bounding box** on the same annotated objects. The model weights, dataset, and evaluation procedure are held constant; the prompt type is the main experimental variable.

Our research question is: **Does a bounding box produce better segmentation masks than one positive click when the model stays the same?**

The pipeline loads COCO images and annotations, builds an image-level split, creates the prompts, runs SAM 2.1, evaluates predicted masks against COCO ground truth, and generates paired comparisons and figures. **No training or fine-tuning is performed.**

## Dataset and methodology

- **Source:** [MS COCO 2017 validation set](https://cocodataset.org/#download), with 80 object categories.
- **After filtering:** 4,952 images and 36,335 annotated objects.
- **Split:** 60% development / 20% validation / 20% test, by image (seed `42`). Objects from the same image remain in the same subset.
- **Final test set:** 991 images and 6,941 objects.
- **Filtering:** Exclude `iscrowd=1`, missing images, invalid mask shapes, and empty masks. There is no minimum object-area threshold. Ground-truth masks are decoded using `COCO.annToMask()`.
- **Point condition (`point_center`):** Select the mask pixel farthest from the object boundary using a distance transform; use it as one positive click.
- **Box condition (`box_gt`):** Use the COCO ground-truth bounding box. This is an *oracle* box, not a user-drawn box.
- **Model:** SAM 2.1 `hiera_base_plus`, pretrained checkpoint. For each prompt, SAM 2 generates three masks; the mask with the highest predicted score is selected.
- **Paired comparison:** Predictions for the same objects are matched using their `annotation_id`.
- **Metrics:** Intersection over Union (IoU), Dice coefficient, and inference latency (image encoder once per image; prompt prediction per object).

## Main findings (final held-out test split)

| Metric | Point prompt | Box prompt |
|---|---:|---:|
| Mean IoU | 0.6009 | 0.7619 |
| Mean Dice | 0.6997 | 0.8508 |
| IoU standard deviation | 0.2949 | 0.1794 |
| Severe failures (IoU < 0.3) | 20.8% | 2.7% |
| Mean prompt latency | ~10.1 ms | ~10.0 ms |

Using ground-truth boxes improves mean IoU by **0.1610** and mean Dice by **0.1511**, and substantially reduces severe segmentation failures. Boxes outperform points for **68.4%** of evaluated objects. The mean image-encoding time is approximately **590 ms per image** in the reported experimental setup. These timing measurements are hardware-dependent and should not be interpreted as universal benchmarks.

The improvement is particularly pronounced for larger objects; a single click often captures only part of an object. Some cases still fail with either prompt, including overlapping objects and prompts that do not sufficiently disambiguate the target.

## Repository layout

```text
.
├── README.md                        # Project overview and reproduction steps
├── EVALUATION_README.md             # Detailed evaluation inputs and outputs
├── requirements.txt                 # Python dependencies
├── dataset.py                       # COCO filtering and deterministic splits
├── sam2_model.py                    # SAM 2 predictor, point prompt, timing
├── run_point_baseline.py            # Point-prompt inference
├── run_box_baseline.py              # Box-prompt inference
├── evaluate_baseline.py             # Object-level IoU/Dice evaluation
├── compare_prompts.py               # Paired metrics and summary
├── make_comparison.py               # Comparison tables and plots
├── make_failure_figures.py          # Representative failure visualizations
├── analyze_runtime.py               # Runtime analysis
├── visualize_point_baseline.py      # Point examples
├── visualize_box_baseline.py        # Box examples
├── test_*.py                        # Unit tests
├── scripts/
│   ├── download_coco.py
│   └── download_sam2_checkpoint.py
├── data/                            # Downloaded COCO and processed split
└── results/                         # Generated predictions and evaluations
```

Downloaded COCO images, model weights, and complete inference outputs may be large and should be generated locally instead of assuming they are bundled with the repository.

## Installation and environment

**Python:** 3.12 or newer (per the project's setup instructions). A CUDA-capable GPU is recommended for full inference; the scripts also support `mps` on macOS and `cpu` for smaller checks. Device selection defaults to `auto`. The project was also developed with macOS/MPS for local checks; runtime results should be compared only under the same hardware and settings.

```bash
git clone https://github.com/katyazavr/sam2-interactive-segmentation.git
cd sam2-interactive-segmentation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` contains NumPy, pandas, Pillow, pycocotools, PyTorch, torchvision, OpenCV, matplotlib, and the official SAM 2 package installed from its GitHub repository. Some dependencies use minimum rather than exact versions, and SAM 2 is installed from a Git branch; **the current file is not a fully locked environment**. For exact reproduction, use the Python, PyTorch, platform, device, precision, and package details captured in the generated `*_run_info.json` files, and pin the SAM 2 commit / package versions used for the final run.

## Download and prepare COCO

```bash
python scripts/download_coco.py
python dataset.py --coco-root data/coco --out-dir data/processed
```

Expected downloaded paths:

- `data/coco/val2017/*.jpg`
- `data/coco/annotations/instances_val2017.json`

Generated split files:

- `data/processed/objects.csv` — one row per valid object
- `data/processed/splits.json` — image IDs in development / validation / test
- `data/processed/dataset_stats.json` — counts and rejection reasons

Download model weights:

```bash
python scripts/download_sam2_checkpoint.py
```

Expected checkpoint: `checkpoints/sam2.1_hiera_base_plus.pt` (approximately 320 MB; not committed to Git).

## Quick smoke test

Run a small number of development images first to verify the environment, data, and checkpoint:

```bash
python run_point_baseline.py --split development --max-images 10
python run_box_baseline.py --split development --max-images 10
python evaluate_baseline.py --split development --condition point_center
python evaluate_baseline.py --split development --condition box_gt
python compare_prompts.py --split development
```

**Important:** results from this ten-image check are *not* the final test-set results.

## Reproduce the final experiment

Run these commands from the repository root **after** downloading COCO, preparing the splits, and downloading the SAM 2 checkpoint:

```bash
# 1. Generate predictions for the same test images and objects
python run_point_baseline.py --split test
python run_box_baseline.py --split test

# 2. Evaluate segmentation against COCO masks
python evaluate_baseline.py --split test --condition point_center
python evaluate_baseline.py --split test --condition box_gt

# 3. Produce paired comparison and report-ready plots
python compare_prompts.py --split test
python make_comparison.py --split test
```

Optional visualizations and failure analysis:

```bash
python visualize_point_baseline.py --split test --n 8
python visualize_box_baseline.py --split test --n 8
python make_failure_figures.py --help
python analyze_runtime.py --help
```

The latter two `--help` commands display the supported options; configure them according to the available result files rather than assuming they will run without arguments.

### Main output files

The baseline scripts create files in `results/`:

- `point_center_test.csv`, `box_gt_test.csv`: prediction metadata and timings
- `point_center_test_masks.jsonl`, `box_gt_test_masks.jsonl`: predicted masks encoded as COCO RLE
- `point_center_test_run_info.json`, `box_gt_test_run_info.json`: model, device, settings, and environment information

Evaluation creates `results/evaluation_test/`:

- `point_center_evaluation_objects.csv` and `box_gt_evaluation_objects.csv`
- Per-condition summary CSV and JSON files
- `comparison_test.csv`, `comparison_test.json`, and `comparison_summary.md`
- `comparison/` with threshold, size, category, and paired-comparison tables and figures, including `fig_iou_threshold.png`

See [EVALUATION_README.md](EVALUATION_README.md) for the evaluation file formats and metric definitions.

## Tests

```bash
python -m unittest test_evaluation test_compare -v
python -m unittest test_box_baseline -v
```

The unit tests check evaluation and comparison logic; they do not replace running full inference on the held-out test set.

## Limitations and interpretation

1. **Oracle bounding boxes:** The box is derived from COCO ground truth. Results may be weaker with imprecise boxes drawn by users.
2. **Single model and split:** We test one pretrained SAM 2.1 configuration on one held-out split, without fine-tuning or independent cross-dataset validation.
3. **Qualitative failure analysis:** Suggested causes of failure are interpretations of examples, not independently controlled experiments.
4. **Hardware-dependent timing:** Encoder and prompt latency depend on device and precision. Encoder inference happens once per image, not once per object.
5. **Prompt information differs:** A box contains more information about an object's extent than one positive click; similar inference latency does not mean equal user effort.

## Demo and final submission

The repository documents the **Python evaluation pipeline**. A separate interactive demo, presentation, and technical report may accompany the course submission. Add verified links here if those artifacts are published:

- **Interactive demo / demo video:** link to be added
- **Presentation:** link to be added
- **Two-page technical summary:** link to be added

Do not interpret the Python baseline scripts as a web demo: the currently documented repository does not include a verified web-app launch command.

## Team contributions

- **Sabina Yamilova:** Experimental results, final technical report assembly, and interactive demo/presentation support.
- **Anna Ksel:** Experimental approach comparison and evaluation analysis.
- **Arina Agafonova:** Runtime and performance analysis.
- **Ekaterina Salazkina:** Representative failure-case analysis and discussion of limitations.

## Data sources, credit, and license

- Dataset: [MS COCO 2017](https://cocodataset.org/#download)
- Validation images: http://images.cocodataset.org/zips/val2017.zip
- Train/validation annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- COCO paper: Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014, [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- SAM 2: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

COCO annotation and website content is licensed under CC BY 4.0; image rights are separate and subject to the respective image rights and Flickr terms. The COCO API uses a BSD-style license. Consult the [COCO terms](https://cocodataset.org/#termsofuse) and the third-party repository licenses before redistribution. The project's own license is in [LICENSE](LICENSE).
