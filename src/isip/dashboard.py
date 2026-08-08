"""Legacy module name shim for the operator dashboard.

The authoritative dashboard lives in :mod:`isip.dashboard.app` (Streamlit).
This ``dashboard.py`` file historically hosted a standalone Flask dashboard
that duplicated the vision pipeline; that implementation has been retired. It
is preserved only as a launcher so any old invocations keep working without
executing a second, conflicting vision implementation.

    python -m isip.dashboard
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def main() -> None:
    from .dashboard.app import main as streamlit_main  # type: ignore[attr-defined]

    streamlit_main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        os.system("streamlit run src/isip/dashboard/app.py --server.port 8501")
    except KeyboardInterrupt:  # pragma: no cover
        pass