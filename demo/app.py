"""SAM 2 Interactive Segmentation Lab — local defense website.

Run from the existing research repository root, after copying the demo/ folder:
    python -m demo.app

Set SAM2_PROJECT_ROOT to an explicit existing project directory if needed.
This module deliberately does not download data, modify research outputs, or invent predictions.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SAM2_PROJECT_ROOT", str(HERE.parent))).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COCO_ROOT = ROOT / "data" / "coco"
ANN_PATH = COCO_ROOT / "annotations" / "instances_val2017.json"
RESULTS = ROOT / "results"
CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
SPLITS = ROOT / "data" / "processed" / "splits.json"
STATIC = HERE / "static"
CURATED = [
    (475292, "Point misses the whole person", "Point under-segmentation"),
    (1116028, "Single-key keyboard prediction", "Point under-segmentation"),
    (1917299, "Neighboring carrots merged", "Box over-segmentation"),
    (373276, "Partially occluded chair", "Box limitation"),
    (1169777, "Small backpack confused", "Both fail"),
    (1616035, "Table annotation ambiguity", "Both fail"),
]
FIGURES = {
    "failure": RESULTS / "evaluation_test" / "failure_analysis" / "fig_failure_compact.png",
    "failure_all": RESULTS / "evaluation_test" / "failure_analysis" / "fig_failure_cases.png",
    "threshold": RESULTS / "evaluation_test" / "comparison" / "fig_iou_threshold.png",
    "delta": RESULTS / "evaluation_test" / "comparison" / "fig_delta_iou.png",
    "size": RESULTS / "evaluation_test" / "comparison" / "fig_iou_by_size.png",
    "runtime": RESULTS / "runtime_analysis" / "total_runtime.png",
}

app = FastAPI(title="SAM 2 Interactive Segmentation Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class PredictRequest(BaseModel):
    annotation_id: int = Field(gt=0)
    mode: Literal["point", "box", "compare"] = "compare"
    source: Literal["automatic", "manual"] = "automatic"
    point: list[float] | None = None
    box: list[float] | None = None
    use_saved: bool = False


class DataStore:
    """Load COCO annotations only on demand; images remain on disk."""

    def __init__(self) -> None:
        self._coco = None
        self._lock = threading.Lock()

    def coco(self):
        if self._coco is not None:
            return self._coco
        if not ANN_PATH.exists():
            raise RuntimeError("COCO annotations are missing. Run python scripts/download_coco.py in the research repository.")
        with self._lock:
            if self._coco is None:
                try:
                    from pycocotools.coco import COCO
                except ImportError as ex:
                    raise RuntimeError("pycocotools is not installed. Install repository dependencies first.") from ex
                self._coco = COCO(str(ANN_PATH))
        return self._coco

    def annotation(self, annotation_id: int):
        coco = self.coco()
        ann = coco.anns.get(annotation_id)
        if not ann or ann.get("iscrowd", 0):
            raise RuntimeError(f"Annotation {annotation_id} was not found or is a crowd annotation.")
        info = coco.imgs[ann["image_id"]]
        image_path = COCO_ROOT / "val2017" / info["file_name"]
        if not image_path.is_file():
            raise RuntimeError(f"COCO image missing: {info['file_name']}. Run python scripts/download_coco.py.")
        return coco, ann, info, image_path

    def record(self, annotation_id: int):
        import numpy as np
        from PIL import Image
        coco, ann, info, image_path = self.annotation(annotation_id)
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        gt = coco.annToMask(ann).astype(bool)
        category = coco.cats[ann["category_id"]]["name"]
        return ann, rgb, gt, category


store = DataStore()


class ModelService:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.runner = None
        self.active_image_id = None

    def load(self):
        if self.runner is not None:
            return self.runner
        if not CHECKPOINT.is_file():
            raise RuntimeError("SAM2 checkpoint missing. Run python scripts/download_sam2_checkpoint.py.")
        try:
            from sam2_model import Sam2Runner, load_predictor
        except ImportError as ex:
            raise RuntimeError(f"Cannot import research model code: {ex}. Place demo/ inside the existing repository.") from ex
        self.runner = Sam2Runner(load_predictor(str(CHECKPOINT), "auto"), precision="fp32")
        return self.runner

    def run(self, image_id: int, image, point, box, mode: str):
        from run_box_baseline import predict_box
        runner = self.load()
        cache_hit = self.active_image_id == image_id
        encoder_ms = 0.0 if cache_hit else float(runner.set_image(image))
        self.active_image_id = image_id
        outputs = {}
        if mode in ("point", "compare"):
            result = runner.predict_point(tuple(point))
            outputs["point"] = {"mask": result.mask, "score": result.score, "prompt_ms": result.prompt_ms}
        if mode in ("box", "compare"):
            mask, score, _, prompt_ms = predict_box(runner, box)
            outputs["box"] = {"mask": mask, "score": score, "prompt_ms": prompt_ms}
        return outputs, encoder_ms, cache_hit, str(runner.device)


model = ModelService()


def json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def results_snapshot():
    snapshot = json_file(HERE / "assets" / "final_results_snapshot.json") or {}
    point = json_file(RESULTS / "evaluation_test" / "point_center_evaluation_summary.json") or snapshot.get("point", {})
    box = json_file(RESULTS / "evaluation_test" / "box_gt_evaluation_summary.json") or snapshot.get("box", {})
    comparison = json_file(RESULTS / "evaluation_test" / "comparison_test.json") or snapshot.get("comparison", {})
    point_info = json_file(RESULTS / "point_center_test_run_info.json") or snapshot.get("point_info", {})
    box_info = json_file(RESULTS / "box_gt_test_run_info.json") or snapshot.get("box_info", {})
    from_repository = (RESULTS / "evaluation_test" / "comparison_test.json").is_file()
    return {
        "point": point, "box": box, "comparison": comparison,
        "point_info": point_info, "box_info": box_info,
        "thresholds": snapshot.get("thresholds", []),
        "from_repository": from_repository,
        "source_note": "Saved final test files in the research repository" if from_repository else "Bundled snapshot of the team's saved final test results (November 2026)",
    }


def mask_as_overlay(mask, rgb: tuple[int, int, int]) -> str:
    """Encode binary mask as transparent colored PNG. Mask itself is real model/COCO output."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(mask).astype(bool)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[arr, :3] = rgb
    rgba[arr, 3] = 175
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def metrics(pred, gt) -> dict:
    import numpy as np
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    denom = int(pred.sum()) + int(gt.sum())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / denom) if denom else 1.0,
        "pred_area": int(pred.sum()),
    }


def default_prompt(mask, coco_bbox, image_shape):
    from sam2_model import center_point
    from run_box_baseline import coco_box_xyxy
    return list(center_point(mask.astype("uint8"))), coco_box_xyxy(coco_bbox, image_shape).tolist()


def validate_point(pt, image):
    import numpy as np
    if pt is None or len(pt) != 2 or not all(np.isfinite(p) for p in pt):
        raise ValueError("Point must contain two finite coordinates. Click on the image first.")
    h, w = image.shape[:2]
    if not (0 <= pt[0] < w and 0 <= pt[1] < h):
        raise ValueError("Point is outside the image.")
    return [float(pt[0]), float(pt[1])]


def validate_box(box, image):
    import numpy as np
    if box is None or len(box) != 4 or not all(np.isfinite(p) for p in box):
        raise ValueError("Box must contain four finite coordinates. Drag a rectangle on the image.")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted([max(0, min(w, x1)), max(0, min(w, x2))])
    y1, y2 = sorted([max(0, min(h, y1)), max(0, min(h, y2))])
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("Draw a larger box around the object.")
    return [x1, y1, x2, y2]


@lru_cache(maxsize=64)
def saved_csv_row(path_str: str, annotation_id: int):
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        row = df.loc[df["annotation_id"] == annotation_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except (OSError, ValueError, KeyError):
        return None


@lru_cache(maxsize=32)
def saved_mask(kind: str, annotation_id: int):
    path = RESULTS / f"{kind}_test_masks.jsonl"
    if not path.is_file():
        return None
    from pycocotools import mask as coco_mask
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if int(record.get("annotation_id", -1)) == annotation_id:
                    rle = record["rle"].copy()
                    rle["counts"] = rle["counts"].encode("ascii")
                    return coco_mask.decode(rle).astype(bool)
            except (KeyError, ValueError, TypeError):
                continue
    return None


def saved_outputs(annotation_id: int, requested_mode: str):
    outputs = {}
    names = [("point", "point_center"), ("box", "box_gt")]
    for name, stem in names:
        if requested_mode not in (name, "compare"):
            continue
        mask = saved_mask(stem, annotation_id)
        if mask is None:
            raise RuntimeError(f"Saved mask for {name} not found. Use live inference or choose a test-set annotation.")
        row = saved_csv_row(str(RESULTS / f"{stem}_test.csv"), annotation_id) or {}
        outputs[name] = {"mask": mask, "score": float(row.get("score", 0)), "prompt_ms": float(row.get("prompt_ms", 0))}
    return outputs


@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    has_dataset = ANN_PATH.is_file()
    has_image_dir = (COCO_ROOT / "val2017").is_dir()
    has_point = (RESULTS / "point_center_test_masks.jsonl").is_file()
    has_box = (RESULTS / "box_gt_test_masks.jsonl").is_file()
    has_core = (ROOT / "sam2_model.py").is_file() and (ROOT / "run_box_baseline.py").is_file()
    return {
        "project_root": str(ROOT), "dataset_ready": has_dataset and has_image_dir,
        "checkpoint_ready": CHECKPOINT.is_file(), "core_ready": has_core,
        "saved_masks_ready": has_point and has_box, "saved_results_ready": (RESULTS / "evaluation_test" / "comparison_test.json").is_file(),
        "model_loaded": model.runner is not None,
        "mode": "live ready" if (has_dataset and has_image_dir and CHECKPOINT.is_file() and has_core) else "results / setup",
        "setup_commands": ["python scripts/download_coco.py", "python dataset.py --coco-root data/coco --out-dir data/processed", "python scripts/download_sam2_checkpoint.py"],
    }


@app.get("/api/results")
def get_results():
    return results_snapshot()


@app.get("/api/examples")
def examples():
    if not ANN_PATH.is_file() or not (COCO_ROOT / "val2017").is_dir():
        return {"items": [], "available": False}
    try:
        coco = store.coco()
        items = []
        used = set()
        # Start with successful, *measured* examples when the saved test CSVs exist.
        # Do not claim success using an annotation whose actual scores are unknown.
        p_csv = RESULTS / "evaluation_test" / "point_center_evaluation_objects.csv"
        b_csv = RESULTS / "evaluation_test" / "box_gt_evaluation_objects.csv"
        if p_csv.is_file() and b_csv.is_file():
            try:
                import pandas as pd
                p_rows = pd.read_csv(p_csv, usecols=["annotation_id", "iou", "area", "category_name"])
                b_rows = pd.read_csv(b_csv, usecols=["annotation_id", "iou"])
                merged = p_rows.merge(b_rows, on="annotation_id", suffixes=("_point", "_box"))
                eligible = merged[(merged["iou_point"] >= 0.85) & (merged["iou_box"] >= 0.90) & (merged["area"] >= 4000)]
                preferred = ["dog", "cat", "car", "bus", "bicycle", "person", "chair", "airplane"]
                for category in preferred:
                    group = eligible[eligible["category_name"] == category].sort_values("iou_box", ascending=False)
                    for row in group.itertuples(index=False):
                        aid = int(row.annotation_id)
                        if aid in coco.anns:
                            ann = coco.anns[aid]
                            file_name = coco.imgs[ann["image_id"]]["file_name"]
                            if not (COCO_ROOT / "val2017" / file_name).is_file():
                                continue
                            items.append({"id": aid, "title": f"Strong result · IoU point {row.iou_point:.2f}, box {row.iou_box:.2f}", "kind": "Verified success", "category": category, "image_id": ann["image_id"]})
                            used.add(aid)
                            break
                    if len(items) >= 5:
                        break
            except (ValueError, KeyError, OSError, ImportError):
                pass
        for aid, title, kind in CURATED:
            ann = coco.anns.get(aid)
            if ann and not ann.get("iscrowd", 0):
                file_name = coco.imgs[ann["image_id"]]["file_name"]
                if (COCO_ROOT / "val2017" / file_name).is_file():
                    items.append({"id": aid, "title": title, "kind": kind, "category": coco.cats[ann["category_id"]]["name"], "image_id": ann["image_id"]})
                    used.add(aid)
        # Broader gallery: deterministic selection from the test split, where possible.
        split = json_file(SPLITS) or {}
        test_ids = set(split.get("test", []))
        fallback_ids = sorted(test_ids) if test_ids else sorted(coco.getImgIds())
        for image_id in fallback_ids[:280]:
            if len(items) >= 16:
                break
            anns = coco.getAnnIds(imgIds=image_id, iscrowd=False)
            if not anns:
                continue
            candidates = [coco.anns[a] for a in anns if a not in used and coco.anns[a].get("area", 0) > 4500]
            if not candidates:
                continue
            ann = max(candidates, key=lambda a: a.get("area", 0))
            info = coco.imgs[image_id]
            if not (COCO_ROOT / "val2017" / info["file_name"]).is_file():
                continue
            aid = ann["id"]
            category = coco.cats[ann["category_id"]]["name"]
            items.append({"id": aid, "title": f"{category.title()} · COCO sample", "kind": "Dataset sample", "category": category, "image_id": image_id})
            used.add(aid)
        return {"items": items, "available": bool(items)}
    except (OSError, RuntimeError, ImportError) as ex:
        return {"items": [], "available": False, "error": str(ex)}


@app.get("/api/image/{annotation_id}")
def image_for_annotation(annotation_id: int, thumb: bool = False):
    try:
        _, _, _, image_path = store.annotation(annotation_id)
    except RuntimeError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    if not thumb:
        return FileResponse(str(image_path), media_type="image/jpeg")
    from PIL import Image
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((520, 320))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/api/example/{annotation_id}")
def get_example(annotation_id: int):
    try:
        ann, image, gt, category = store.record(annotation_id)
        point, box = default_prompt(gt, ann["bbox"], image.shape)
    except (RuntimeError, ImportError, ValueError) as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    return {
        "annotation_id": annotation_id, "image_id": ann["image_id"], "category": category,
        "width": int(image.shape[1]), "height": int(image.shape[0]),
        "area": int(gt.sum()), "point": point, "box": box,
        "ground_truth": mask_as_overlay(gt, (20, 190, 144)),
        "image_url": f"/api/image/{annotation_id}",
        "saved_available": bool(saved_csv_row(str(RESULTS / "evaluation_test" / "point_center_evaluation_objects.csv"), annotation_id))
            and bool(saved_csv_row(str(RESULTS / "evaluation_test" / "box_gt_evaluation_objects.csv"), annotation_id)),
        "annotation_source": "COCO ground truth (used only to evaluate or generate original baseline prompts)",
    }


@app.get("/api/figure/{figure_name}")
def figure(figure_name: str):
    path = FIGURES.get(figure_name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not available")
    return FileResponse(str(path), media_type="image/png")


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def run_inference(req: PredictRequest) -> Generator[str, None, None]:
    try:
        yield sse({"type": "stage", "stage": "loading", "message": "Reading COCO image and annotation"})
        ann, image, gt, category = store.record(req.annotation_id)
        yield sse({"type": "stage", "stage": "dataset", "message": "COCO image and ground truth loaded"})
        auto_point, auto_box = default_prompt(gt, ann["bbox"], image.shape)
        if req.use_saved:
            if req.source != "automatic":
                raise ValueError("Saved predictions are only valid for the automatic baseline prompts.")
            point, box = auto_point, auto_box
        elif req.source == "manual":
            point = validate_point(req.point, image) if req.mode in ("point", "compare") else auto_point
            box = validate_box(req.box, image) if req.mode in ("box", "compare") else auto_box
        else:
            point, box = auto_point, auto_box
        yield sse({"type": "stage", "stage": "prompt", "message": "Prompt coordinates validated"})
        if req.use_saved:
            yield sse({"type": "stage", "stage": "saved", "message": "Loading saved test-set predictions (not live inference)"})
            outputs = saved_outputs(req.annotation_id, req.mode)
            encoder_ms, cache_hit, device = None, False, "saved test result"
            source_label = "saved"
        else:
            yield sse({"type": "stage", "stage": "encoder", "message": "Initializing SAM2 / encoding image"})
            with model.lock:
                outputs, encoder_ms, cache_hit, device = model.run(int(ann["image_id"]), image, point, box, req.mode)
            source_label = "live"
            yield sse({"type": "stage", "stage": "decoder", "message": "SAM2 mask decoding completed"})
        colored = {"point": (58, 109, 250), "box": (152, 99, 247)}
        result = {}
        for name, item in outputs.items():
            result[name] = {
                **metrics(item["mask"], gt),
                "score": float(item["score"]), "prompt_ms": float(item["prompt_ms"]),
                "mask": mask_as_overlay(item["mask"], colored[name]),
            }
        yield sse({"type": "stage", "stage": "evaluation", "message": "IoU / Dice computed against COCO ground truth"})
        yield sse({
            "type": "result", "source": source_label, "annotation_id": req.annotation_id, "category": category,
            "point": point, "box": box, "results": result,
            "encoder_ms": encoder_ms, "encoder_cache_hit": cache_hit,
            "device": device, "message": "Live SAM2 prediction" if source_label == "live" else "Precomputed research prediction",
        })
    except Exception as ex:
        yield sse({"type": "error", "message": str(ex)})


@app.post("/api/infer-stream")
def infer_stream(req: PredictRequest):
    return StreamingResponse(run_inference(req), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAM2_DEMO_PORT", "8000"))
    print(f"SAM 2 Interactive Lab: http://127.0.0.1:{port}")
    print(f"Research project root: {ROOT}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
