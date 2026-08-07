from isip.config import Settings
from isip.orchestrator import EdgeOrchestrator, _STATUS_SEVERITY


def test_all_sample_statuses_map_to_severity():
    """Every status the telemetry/RUL loops can emit must map to a Severity,
    otherwise the edge node crashes."""
    for status in ("NORMAL", "HEALTHY", "WARN", "CRITICAL"):
        assert _STATUS_SEVERITY[status] is not None


def test_synthetic_camera_never_opens_capture():
    """With synthetic_camera enabled the camera must never be opened."""
    settings = Settings()
    assert settings.video.synthetic_camera is True
    orchestrator = EdgeOrchestrator(settings)
    orchestrator._read_frame()
    assert orchestrator._cap is None