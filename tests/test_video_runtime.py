"""Video stream status tests."""

from unittest.mock import patch

from isip.config import Settings, resolve_video_source
from isip.runtime import VideoStreamProducer


def test_resolve_video_source_uses_project_root(tmp_path):
    media = tmp_path / "demo.mp4"
    media.write_bytes(b"video")

    resolved = resolve_video_source("demo.mp4", project_root=tmp_path)

    assert resolved == str(media.resolve())


def test_video_status_reports_not_started_state():
    settings = Settings()
    producer = VideoStreamProducer(settings)

    status = producer.status()

    assert status["ready"] is False
    assert status["alive"] is False
    assert status["source"] == str(settings.video.source)
    assert status["error"] is None


def test_video_status_reports_source_open_failure():
    settings = Settings()
    settings.video.synthetic_camera = False
    producer = VideoStreamProducer(settings)

    with patch("isip.runtime.cv2.VideoCapture") as capture:
        capture.return_value.isOpened.return_value = False
        producer.start()
        producer._thread.join(timeout=2)

    status = producer.status()

    assert status["ready"] is False
    assert "could not open video source" in status["error"]
    producer.stop()
