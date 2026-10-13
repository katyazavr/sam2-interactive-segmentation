"""SAM 2.1 model for our project (stage 2).

We use the pretrained model, we do not train it.
Baseline: one positive point in the center of the object.

- center point = pixel in the GT mask that is far from the border.
  We find it with distance transform. The image edge is also a border.
- SAM 2 gives 3 masks, we take the mask with the best score.
- set_image() runs one time for each image. All objects in this image use the same embedding.

How to use:
    from sam2_model import load_predictor, center_point, Sam2Runner
    runner = Sam2Runner(load_predictor())
    enc_ms = runner.set_image(image)
    out = runner.predict_point(center_point(mask))
    out.mask, out.score, out.prompt_ms
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

CHECKPOINT_NAME = "sam2.1_hiera_base_plus"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{CHECKPOINT_NAME}.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"   # this config is inside the sam2 package
DEFAULT_CHECKPOINT = Path("checkpoints") / f"{CHECKPOINT_NAME}.pt"


# device and model
def pick_device(name: str = "auto") -> torch.device:
    # auto: use GPU if we have it, then Mac GPU (mps), then CPU
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_predictor(checkpoint: str | Path | None = DEFAULT_CHECKPOINT,
                   device: str = "auto"):
    """Load SAM 2.1 (hiera base+) and make SAM2ImagePredictor.

    checkpoint=None gives a model with random weights. Use it only to test the code.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dev = pick_device(device)
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run: python scripts/download_sam2_checkpoint.py")
    model = build_sam2(MODEL_CONFIG, str(checkpoint) if checkpoint else None, device=str(dev))
    return SAM2ImagePredictor(model)


# prompts
def center_point(mask: np.ndarray) -> tuple[int, int]:
    """Return (x, y) of the mask pixel that is far from the border.

    We add a frame of zeros around the mask, so the image edge is also a border.
    If there are some pixels with the same distance, argmax takes the first one,
    so we always get the same point.
    """
    padded = np.pad((mask > 0).astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


# prediction
@dataclass
class PointResult:
    mask: np.ndarray          # H x W, 0 or 1, the mask we chose
    score: float              # score of this mask
    mask_index: int           # number of the mask we chose (0, 1 or 2)
    all_scores: np.ndarray    # scores of all 3 masks
    low_res_logits: np.ndarray  # (1, 256, 256), we need it later for correction clicks
    prompt_ms: float          # time of predictor.predict() in ms


class Sam2Runner:
    def __init__(self, predictor, precision: str = "fp32"):
        self.predictor = predictor
        self.device = predictor.device
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        self.precision = precision

    def _autocast(self):
        # bf16 works only on CUDA, in other cases we use normal fp32
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _sync(self):
        # GPU works async, so we wait for it before we take the time
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def set_image(self, image: np.ndarray) -> float:
        """Encode one RGB image (uint8). Return encoder time in ms."""
        with self._autocast():
            self._sync()
            t0 = time.perf_counter()
            self.predictor.set_image(image)
            self._sync()
        return (time.perf_counter() - t0) * 1000

    @torch.inference_mode()
    def predict_point(self, point_xy: tuple[int, int]) -> PointResult:
        """One positive point. SAM 2 gives 3 masks, we take the mask with the best score."""
        coords = np.array([point_xy], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)   # 1 = positive point
        with self._autocast():
            self._sync()
            t0 = time.perf_counter()
            masks, scores, logits = self.predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=True)
            self._sync()
        prompt_ms = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(scores))
        return PointResult(
            mask=(masks[idx] > 0).astype(np.uint8),
            score=float(scores[idx]),
            mask_index=idx,
            all_scores=np.asarray(scores, dtype=np.float32),
            low_res_logits=logits[idx:idx + 1],
            prompt_ms=prompt_ms,
        )

    def warmup(self, image: np.ndarray, n: int = 3) -> None:
        """Run the model a few times without timing.
        The first runs are slow, so we do not want them in our results."""
        h, w = image.shape[:2]
        for _ in range(n):
            self.set_image(image)
            self.predict_point((w // 2, h // 2))
