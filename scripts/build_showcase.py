"""Regenerate the GitHub Pages showcase site.

Two deployment layouts are supported so the site works with no GitHub
Settings changes:

- ``docs/index.html``    — for repos configured with Pages "deploy from a
                           branch: main, folder: /docs".
- ``index.html`` (root)  — for repos whose Pages currently deploys from the
                           repo root (Jekyll renders a root ``index.html`` if
                           present, instead of the README). The root page
                           references the already-tracked ``dashboard/``
                           assets, so nothing else needs to move.

Run after regenerating ``dashboard/detection_output.jpg``:

    python scripts/build_showcase.py

It only (re)writes ``docs/`` and the root ``index.html``; it is idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
DOCS = ROOT / "docs"

LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ISIP — Industrial Safety Intelligence Platform</title>
<meta name="description" content="Edge-first multi-modal safety AI for industrial facilities: real-time vision, geofencing, PPE compliance, IIoT telemetry, RUL estimation, and PLC E-Stop control." />
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0b1220; color: #e2e8f0; line-height: 1.6;
    padding: 2.5rem 1.25rem;
  }
  .wrap { max-width: 900px; margin: 0 auto; }
  header { border-bottom: 1px solid #1e293b; padding-bottom: 1.5rem; margin-bottom: 1.5rem; }
  h1 { font-size: 1.6rem; color: #fff; letter-spacing: 0.01em; }
  .sub { color: #22d3ee; font-size: 0.85rem; letter-spacing: 0.2em; margin-top: 0.25rem; }
  p { color: #94a3b8; margin: 0.6rem 0; }
  ul { list-style: none; margin: 1rem 0; }
  li::before { content: "▸ "; color: #34d399; }
  li { padding: 0.15rem 0; }
  .img-row { display: grid; gap: 1rem; margin: 1.5rem 0; }
  @media (min-width: 720px) { .img-row { grid-template-columns: 1fr 1fr; } }
  figure { border: 1px solid #1e293b; border-radius: 10px; overflow: hidden; background: #0f172a; }
  img { width: 100%; height: auto; display: block; }
  figcaption { padding: 0.5rem 0.75rem; font-size: 0.8rem; color: #64748b; }
  a { color: #22d3ee; }
  .btn {
    display: inline-block; margin: 0.25rem 0.5rem 0.25rem 0; padding: 0.55rem 1rem;
    border: 1px solid #164e63; border-radius: 8px; background: #0e2a38; color: #67e8f9;
    text-decoration: none; font-size: 0.9rem;
  }
  .btn:hover { background: #134e5f; }
  .note {
    margin-top: 1.5rem; padding: 0.9rem 1rem; border: 1px solid #3b2f0a;
    background: #1a1405; border-radius: 8px; color: #fbbf24; font-size: 0.85rem;
  }
  footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #1e293b; color: #475569; font-size: 0.8rem; }
  code { background: #1e293b; padding: 0.1rem 0.35rem; border-radius: 4px; color: #a5b4fc; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ISIP — Industrial Safety Intelligence Platform</h1>
    <div class="sub">EDGE-FIRST · MULTI-MODAL · OFFLINE-SAFE</div>
  </header>

  <p>
    An edge-first, multi-modal safety AI system for industrial facilities. It
    fuses real-time vision (YOLOv8 / YOLO-World, person segmentation, PPE
    compliance, polygon geofencing and machine danger zones) with IIoT
    telemetry (temperature, vibration, toxic gas), anomaly detection, Remaining
    Useful Life (RUL) estimation, and a FastAPI control plane that trips a PLC
    E-Stop relay &mdash; all with a sub-20ms safety loop, running fully offline.
  </p>

  <ul>
    <li>Real-time object detection + person segmentation polygons</li>
    <li>Static geofencing and dynamic machine-derived danger zones</li>
    <li>PPE compliance rules (helmet / vest / gloves)</li>
    <li>IIoT telemetry, anomaly detection, and RUL estimation</li>
    <li>Async event broker + audit ledger + FastAPI control plane</li>
    <li>Streamlit operator dashboard with live video overlay</li>
  </ul>

  <div class="img-row">
    <figure>
      <img src="__IMG__detection_output.jpg" alt="Annotated detection frame with person polygons, machine danger zone and geofences" />
      <figcaption>Detection overlay &mdash; person polygons, machine danger zone, static geofences</figcaption>
    </figure>
    <figure>
      <img src="__IMG__geofence_test.jpg" alt="Geofence zone visualization" />
      <figcaption>Restricted-zone geofencing</figcaption>
    </figure>
  </div>

  <p>
    <a class="btn" href="__DASH__">Open interactive dashboard demo</a>
    <a class="btn" href="https://github.com/Anandhu-96/Agentic_AI">View source on GitHub</a>
  </p>

  <h2 style="margin-top:1.4rem; color:#fff; font-size:1.1rem;">Live detection from the demo video</h2>
  <p style="color:#64748b; font-size:0.85rem; margin:0.3rem 0 0.8rem;">
    Real annotated frames extracted from <code>We_want_the_vedio_of_the_engin.mp4</code>
    using the same renderer as the edge node.
  </p>
  <div class="img-row" id="gallery">
    __GALLERY__
  </div>

  <div class="note">
    <strong>Live video &amp; telemetry note:</strong> this static page cannot run the
    Python edge node. To see the real-time feed and control plane, run
    <code>python scripts/run_edge.py</code> and open
    <code>http://127.0.0.1:8080/edge/video-snapshot</code> locally.
  </div>

  <footer>ISIP &middot; Neurobots Championship 2026 &middot; MIT License</footer>
</div>
</body>
</html>
"""


def _gallery_figures() -> str:
    """Figure markup for detection frames sampled from the demo video."""
    gallery = DASH / "gallery"
    frames = sorted(gallery.glob("detection_*.jpg")) if gallery.is_dir() else []
    if not frames:
        return '<p style="color:#64748b; font-size:0.85rem;">(run <code>scripts/generate_detection_gallery.py</code> to add frames)</p>'
    figs = []
    for f in frames:
        figs.append(
            f'<figure><img src="__IMG__gallery/{f.name}" alt="Detection frame from the demo video" />'
            f'<figcaption>{f.stem.replace("detection_", "Frame ")}</figcaption></figure>'
        )
    return "\n    ".join(figs)


def main() -> None:
    DOCS.mkdir(exist_ok=True)

    # Preview images (from the dashboard folder so one source of truth stays).
    for name in ("detection_output.jpg", "geofence_test.jpg"):
        src = DASH / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src} — run scripts/generate_dashboard_image.py first")
        shutil.copyfile(src, DOCS / name)

    # Interactive dashboard demo (static, CDN-based).
    demo = DASH / "isip-dashboard.html"
    if demo.exists():
        shutil.copyfile(demo, DOCS / "isip-dashboard.html")

    # Detection frames sampled from the demo video.
    gallery_src = DASH / "gallery"
    if gallery_src.is_dir():
        dst = DOCS / "gallery"
        dst.mkdir(exist_ok=True)
        for f in sorted(gallery_src.glob("detection_*.jpg")):
            shutil.copyfile(f, dst / f.name)

    gallery = _gallery_figures()
    page = LANDING_PAGE.replace("__GALLERY__", gallery)

    # docs/ layout (Pages configured with folder /docs).
    (DOCS / "index.html").write_text(
        page.replace("__IMG__", "").replace("__DASH__", "isip-dashboard.html"),
        encoding="utf-8",
    )

    # Root layout (Pages deploying from repo root): reuse the already-tracked
    # dashboard/ assets so the site shows the showcase with no Settings
    # changes and nothing gets duplicated at the repo root.
    (ROOT / "index.html").write_text(
        page.replace("__IMG__", "dashboard/").replace("__DASH__", "dashboard/isip-dashboard.html"),
        encoding="utf-8",
    )

    print(f"showcase updated under {DOCS} and {ROOT / 'index.html'}")
    for p in sorted(DOCS.iterdir()):
        print(f"  docs/{p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()