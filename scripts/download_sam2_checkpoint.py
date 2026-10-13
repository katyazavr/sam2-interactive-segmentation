"""Download the SAM 2.1 hiera base+ weights (about 320 MB).

How to run (from the main project folder):
    python scripts/download_sam2_checkpoint.py
Result:
    checkpoints/sam2.1_hiera_base_plus.pt
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
OUT = Path(__file__).resolve().parent.parent / "checkpoints" / "sam2.1_hiera_base_plus.pt"


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def main():
    if OUT.exists():
        print(f"skip: {OUT} already present")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")   # if the download stops, we do not get a broken file
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp, progress)
    tmp.rename(OUT)
    print(f"\nDone: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
