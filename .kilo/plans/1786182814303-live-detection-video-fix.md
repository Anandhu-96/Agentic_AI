# Fix live detection video fallback

## Findings
- `dashboard/isip-dashboard.html` requests the annotated MJPEG stream at `/edge/video-feed` only after `/edge/metrics` succeeds.
- When the backend probe fails, `probeBackend()` replaces the source with static `detection_output.jpg` (`dashboard/isip-dashboard.html:1437-1445`), which cannot show live detections.
- The live annotated frames are produced by `VideoStreamProducer` in `src/isip/runtime.py:160-249`; detections are drawn at `src/isip/runtime.py:228-242` and exposed as MJPEG by `src/isip/api/server.py:252-273`.
- The message `Cannot read image (this model does not support image input)` is from the attached-media viewer, not the application. It should not be displayed as an application video fallback.
- `config/config.yaml` currently uses `vision.backend: synthetic`, so once the edge server is running the feed will show synthetic detection overlays, not YOLO model detections. `video.synthetic_camera: false` and the configured MP4 source are otherwise suitable.

## Implementation steps
1. Update the HTML dashboard video error handling so it does not silently replace a failed live stream with a static JPEG. Show a clear in-panel backend/video error state containing the configured API base and a start command, while retaining an explicit optional static-preview action if needed.
2. Add a backend video readiness/status endpoint or extend `/edge/metrics` with video producer state (`is_alive`, source, and error) so the dashboard can distinguish “backend unavailable,” “video producer failed,” and “stream connected but no frame.” Do not expose secrets.
3. Update `probeBackend()` and `startLive()` to use the readiness information and keep `/edge/video-feed` as the source whenever the backend is online. Add retry behavior that restores the live stream after the backend becomes available.
4. Ensure the video producer reports startup failures clearly. In particular, validate the configured MP4 path relative to the project root and log a useful error when `cv2.VideoCapture` cannot open it instead of serving an indefinite blank/fallback experience.
5. Decide and document the intended inference mode in the active config: keep `synthetic` for deterministic demo overlays, or switch to `yolov8` only if the model/dependencies are installed. Do not silently downgrade between modes.
6. Apply equivalent dashboard changes to the generated/static copies (`index.html`, `dashboard/isip-dashboard.html`, and `docs/*` copies) so the displayed page does not depend on which copy is opened.

## Validation
- Start the edge process with `python scripts/run_edge.py --config config/config.yaml` and verify `/edge/metrics` returns 200.
- Open `/edge/video-feed` directly and verify it returns `multipart/x-mixed-replace` frames.
- Open the dashboard and confirm the video panel shows changing annotated frames, not `detection_output.jpg`.
- Stop the backend and confirm the panel shows a diagnostic state rather than the misleading attached-image/model error.
- Test a missing video source and confirm the backend log/status identifies the source-open failure.
- Run the existing test suite and add focused tests for video readiness/error reporting and dashboard fallback behavior.

## Scope decision
The attached-media viewer error itself cannot be fixed in the repository because it is produced by the current model/tool interface, which does not support image input. The repository fix targets the application behavior that currently masks live-feed failures with a static image.
