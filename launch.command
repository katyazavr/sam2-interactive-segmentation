#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Starting SAM 2 Interactive Segmentation Lab..."
echo "If you are using the original repository, copy demo/ there and run python -m demo.app."
python3 -m demo.app
