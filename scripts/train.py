#!/usr/bin/env python3
"""Train the PPE detector on the Ultralytics Construction-PPE dataset.

Usage:
    python scripts/train.py [--model yolov8n.pt] [--epochs 50] [--imgsz 640]

Requires ``pip install ultralytics`` (the dataset is auto-downloaded on first
run). See src/isip/training/train.py for details.
"""

from isip.training.train import main

if __name__ == "__main__":
    main()