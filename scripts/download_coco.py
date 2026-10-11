"""Cross-platform downloader for COCO 2017 val images and annotations (Windows/Linux/macOS).

Usage (from the repo root):
    python scripts/download_coco.py
Result:
    data/coco/val2017/*.jpg
    data/coco/annotations/instances_val2017.json
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "coco"
FILES = [
    ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip", ROOT / "val2017"),
    ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
     "annotations_trainval2017.zip", ROOT / "annotations" / "instances_val2017.json"),
]


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({min(100, done * 100 // total)}%)")
        sys.stdout.flush()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for url, zip_name, marker in FILES:
        if marker.exists():
            print(f"skip {zip_name}: {marker.name} already present")
            continue
        zip_path = ROOT / zip_name
        if not zip_path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, zip_path, progress)
            print()
        print(f"extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ROOT)
    n_img = len(list((ROOT / "val2017").glob("*.jpg")))
    ann = ROOT / "annotations" / "instances_val2017.json"
    print(f"Done: {n_img} images, annotations present: {ann.exists()}")


if __name__ == "__main__":
    main()
