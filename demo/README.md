# SAM 2 — Interactive Segmentation Lab

A local, English-language website for the **SAM 2 Interactive Segmentation** computer-vision defense. It connects directly to the real project code. No cloud inference and no additional model training.

## Launch on a Mac

1. Open the existing research repository (not a new empty folder):

   `https://github.com/katyazavr/sam2-interactive-segmentation`

2. Copy the entire **`demo/`** directory from this package into the **root** of that repository, next to `dataset.py`, `sam2_model.py`, `run_point_baseline.py` and `run_box_baseline.py`.

3. Activate the **same Python environment used by the research project** and run:

   ```bash
   python -m pip install fastapi uvicorn
   python -m demo.app
   ```

4. Open **http://127.0.0.1:8000** in a browser.

The local website works in *results-viewing mode* immediately even when COCO images and SAM2 weights are absent. In that case it displays real team **aggregate result snapshots**, and clearly disables live prediction. The illustrative preview scene is **not** a model prediction.

### To activate live inference

Make sure the original research dependencies are installed and these resources are available:

- `data/coco/val2017/*.jpg` — COCO 2017 images (about 1 GB download)
- `data/coco/annotations/instances_val2017.json` — COCO annotations
- `data/processed/splits.json` — reproducible image-level split (60/20/20, seed 42)
- `checkpoints/sam2.1_hiera_base_plus.pt` — pretrained SAM2 weights (about 320 MB)

If missing, use the scripts already in the research repository:

```bash
python -m pip install -r requirements.txt
python scripts/download_coco.py
python dataset.py --coco-root data/coco --out-dir data/processed
python scripts/download_sam2_checkpoint.py
```

Restart `python -m demo.app`, then refresh the browser. **Nothing downloads automatically at startup.** Inference runs on CUDA when available, otherwise Apple MPS on a compatible Mac, then CPU.

## What the website does

- Gallery of real COCO images and annotation IDs, including curated failures.
- Point prompt from a clicked location and box prompt drawn on the image.
- **Automatic research comparison**: center point inside the COCO GT mask vs oracle COCO GT bounding box.
- Both prompts use the **same pretrained SAM2.1 Hiera Base+**, with the best-scoring of three masks.
- Point and box masks, ground truth mask, confidence score, IoU, Dice, encoder and prompt times.
- Streamed real pipeline states from the local Python backend, not simulated progress.
- Saved aggregate research results: 991 test images, 6,941 objects, IoU and Dice, paired comparison, thresholds and failures.
- If `results/*_test_masks.jsonl` and COCO images are present, the website can show **precomputed test-set masks** without loading the model. These are clearly labeled *SAVED TEST*, never passed off as live inference.
- Presentation Mode for a short defense or screen recording.

## Using it for the two-minute defense video

1. Select a clear COCO sample in the gallery.
2. Leave **Research baseline** selected and choose **Both**.
3. Select **Compare both prompts** — this runs both prompt types on the same object, sharing a single image embedding. Watch the pipeline and compare masks.
4. Talk through IoU and Dice on the right. Navigate to **Experiment results** for the full test-set improvement.
5. Select a curated failure example such as *Person #475292* or *Carrot #1917299*, then explain the limitation.

For a screen recording on macOS use **Shift + Command + 5**. Verify one inference in advance, because the first SAM2 load can take significantly longer than later predictions. The Presentation Mode offers a guided sequence.

### IMPORTANT: prompt interpretation

The original research uses a **point generated from a COCO ground-truth mask** and a **box taken from the COCO ground-truth annotation**. This is an **oracle box**, not a detector prediction. The interactive manual mode is illustrative and should not be reported as if it were the controlled test evaluation.

## Running in an alternate directory

You can run this package outside the main repo to inspect the UI and saved aggregate snapshot, or point it at an existing checkout:

```bash
SAM2_PROJECT_ROOT="$HOME/path/to/sam2-interactive-segmentation" python -m demo.app
```

If the standalone package is in a separate folder, run this command from the package root containing `demo/`.

The web server binds to `127.0.0.1` only and serves no external fonts or application scripts. To change the port:

```bash
SAM2_DEMO_PORT=8765 python -m demo.app
```

## Source files

- `demo/app.py` — FastAPI backend with lazy SAM2 initialization, COCO dataset adapter, inference stream, mask encoding, metrics, saved-prediction reader and read-only figures.
- `demo/static/index.html` — website layout.
- `demo/static/styles.css` — local light theme, responsive layout and presentation mode.
- `demo/static/app.js` — gallery, canvas interactions, coordinates, streaming prediction, comparison charts and guided presentation.
- `demo/static/preview-scene.svg` — **illustrative placeholder only**, displayed when no COCO image is selected.
- `demo/assets/final_results_snapshot.json` — snapshot of the team's existing saved final test measurements, used only if the original repo results are missing.

### Limitations

- Real prediction and per-image evaluation require local COCO data, checkpoint and project dependencies. They are **not shipped** in this zip because the model and dataset are large and licensed separately.
- The initial gallery needs local COCO images and annotations.
- Saved masks require the original repository's RLE prediction files and COCO images. When using saved results, the UI says **SAVED TEST** and does not claim to have performed new inference.
- Because the website has no data and checkpoint bundled in this package, the complete GPU/MPS inference path must be smoke-tested on the actual Mac with the existing project environment.
