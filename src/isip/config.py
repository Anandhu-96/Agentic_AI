"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    source: Any = 0
    width: int = 1280
    height: int = 720
    save_output: bool = False
    output_dir: str = "data/output"
    synthetic_camera: bool = True


class SegmentationConfig(BaseModel):
    enabled: bool = False
    model: str = "yolov8n-seg.pt"
    conf_threshold: float = 0.45


class VisionConfig(BaseModel):
    model: str = "yolov8n.pt"
    backend: str = "synthetic"  # synthetic (demo) | yolov8 | yolov8_world | yolov8_seg
    conf_threshold: float = 0.45
    iou_threshold: float = 0.5
    imgsz: int = 640  # YOLO inference resolution (internal; boxes map back to frame)
    classes: List[str] = Field(default_factory=lambda: ["person", "helmet"])
    machine_classes: List[str] = Field(
        default_factory=lambda: ["heavy machinery", "industrial machine", "construction equipment", "engine"]
    )
    ppe_required: List[str] = Field(default_factory=lambda: ["helmet", "vest"])
    ppe_rules: Dict[str, Any] = Field(default_factory=dict)
    geofences_file: str = "config/geofences.yaml"
    device: str = "cpu"
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)


class TrackingConfig(BaseModel):
    enabled: bool = True
    max_age: int = 10
    iou_threshold: float = 0.3


class MachineDangerConfig(BaseModel):
    enabled: bool = True
    buffer: float = 0.05  # normalized expansion around the machine bbox
    smoothing: float = 0.7  # EMA factor for zone smoothing (0=no smoothing)


class DynamicZonesConfig(BaseModel):
    machine_danger: MachineDangerConfig = Field(default_factory=MachineDangerConfig)


class VisualizationConfig(BaseModel):
    show_person_bbox: bool = False
    show_person_polygon: bool = True
    fill_person_polygon: bool = True
    show_machine_bbox: bool = True
    show_machine_zone: bool = True
    show_static_zones: bool = True


class ThresholdConfig(BaseModel):
    warn: float
    critical: float


class IiotConfig(BaseModel):
    sensor_ids: List[str] = Field(default_factory=list)
    sampling_interval_s: float = 1.0
    protocol: str = "emulated"
    broker_host: str = "127.0.0.1"
    broker_port: int = 1883
    modbus_unit: int = 1
    thresholds: Dict[str, ThresholdConfig] = Field(default_factory=dict)


class RulConfig(BaseModel):
    nominal_temp_c: float = 60.0
    activation_temp_c: float = 70.0
    max_rated_life_h: float = 100_000.0


class ControlConfig(BaseModel):
    estop_gpio: int = 17
    relay_channel: str = "GPIO-3"
    line_id: str = "LINE_01"
    trip_delay_ms: int = 0
    use_gpio: bool = False


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    edge_prefix: str = "/edge"


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501
    refresh_ms: int = 500


class LoggingConfig(BaseModel):
    audit_dir: str = "logs"
    level: str = "INFO"


class EdgeConfig(BaseModel):
    node_id: str = "edge-node-01"
    hardware: str = "nvidia-jetson-orin-nano"
    inference_backend: str = "yolov8"
    target_latency_ms: int = 20
    fps_limit: int = 30


class Settings(BaseModel):
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    iiot: IiotConfig = Field(default_factory=IiotConfig)
    rul: RulConfig = Field(default_factory=RulConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    dynamic_zones: DynamicZonesConfig = Field(default_factory=DynamicZonesConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
