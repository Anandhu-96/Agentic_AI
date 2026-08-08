"""PPE detection training on the Ultralytics Construction-PPE dataset.

Ultralytics ships built-in dataset configs as YAML that are resolved by name —
passing ``data="construction-ppe.yaml"`` to ``model.train()`` makes YOLO
download and extract the dataset (1,416 images) automatically on first use.

Usage:
    python scripts/train.py                 # defaults: yolov8n.pt, 50 epochs
    python scripts/train.py --model yolov11s.pt --epochs 80 --imgsz 640
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# Built-in Ultralytics dataset config (auto-downloaded on first use).
CONSTRUCTION_PPE = "construction-ppe.yaml"

# 11 classes in the Construction-PPE dataset (as shipped in ultralytics).
CLASSES = [
    "helmet",
    "no_helmet",
    "gloves",
    "no_gloves",
    "vest",
    "no_vest",
    "boots",
    "no_boots",
    "goggles",
    "no_goggle",
    "person",
]


def train(
    model: str = "yolov8n.pt",
    data: str = CONSTRUCTION_PPE,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/ppe",
    name: str = "construction",
) -> "object":
    """Train a YOLO detector on the Construction-PPE dataset.

    Returns the trained :class:`ultralytics.YOLO` model so callers can export
    to ONNX/TensorRT or dump validation metrics.
    """
    from ultralytics import YOLO

    logger.info("starting training model=%s data=%s epochs=%d imgsz=%d batch=%d",
                model, data, epochs, imgsz, batch)
    model = YOLO(model)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
    )
    logger.info("training completed project=%s name=%s", project, name)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8/YOLOv11 on the Ultralytics Construction-PPE dataset"
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/ppe")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    train(
        model=args.model,
        data=CONSTRUCTION_PPE,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":  # pragma: no cover
    main()