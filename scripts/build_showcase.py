"""Regenerate the GitHub Pages site (root <-> docs mirror).

The GitHub Pages site IS the interactive operator dashboard
(``dashboard/isip-dashboard.html``), so visitors land directly on the product
instead of a landing page.

Two deployment layouts are supported so the site works with no GitHub
Settings changes:

- ``docs/index.html``    — for repos configured with Pages "deploy from a
                           branch: main, folder: /docs".
- ``index.html`` (root)  — for repos whose Pages currently deploys from the
                           repo root (Jekyll renders a root ``index.html`` if
                           present, instead of the README).

The dashboard template contains a single ``__A__`` marker that prefixes
relative assets (``detection_output.jpg``, gallery frames, …). It is replaced
with ``dashboard/`` for the root layout and left empty for the ``docs/``
layout, where assets are copied alongside the page.

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


def main() -> None:
    DOCS.mkdir(exist_ok=True)

    # Preview/fallback images (from the dashboard folder so one source of
    # truth stays).
    for name in ("detection_output.jpg", "geofence_test.jpg"):
        src = DASH / name
        if not src.exists():
            raise FileNotFoundError(
                f"missing {src} — run scripts/generate_dashboard_image.py first"
            )
        shutil.copyfile(src, DOCS / name)

    # Interactive operator dashboard (self-contained, CDN-based).
    demo = DASH / "isip-dashboard.html"
    if not demo.exists():
        raise FileNotFoundError(f"missing {demo}")
    template = demo.read_text(encoding="utf-8")

    # Detection frames sampled from the demo video.
    gallery_src = DASH / "gallery"
    if gallery_src.is_dir():
        dst = DOCS / "gallery"
        dst.mkdir(exist_ok=True)
        for f in sorted(gallery_src.glob("detection_*.jpg")):
            shutil.copyfile(f, dst / f.name)

    # docs/ layout (Pages configured with folder /docs): assets colocated.
    (DOCS / "index.html").write_text(
        template.replace("__A__", ""), encoding="utf-8"
    )
    (DOCS / "isip-dashboard.html").write_text(
        template.replace("__A__", ""), encoding="utf-8"
    )

    # Root layout (Pages deploying from repo root): reuse the already-tracked
    # dashboard/ assets so the site shows the dashboard with no Settings
    # changes and nothing gets duplicated at the repo root.
    (ROOT / "index.html").write_text(
        template.replace("__A__", "dashboard/"), encoding="utf-8"
    )

    print(f"dashboard deployed under {DOCS} and {ROOT / 'index.html'}")
    for p in sorted(DOCS.iterdir()):
        print(f"  docs/{p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
