"""Streamlit operator dashboard.

Live video overlay + telemetry monitoring UI. Uses SSE for real-time events
instead of REST polling. When the API is unreachable it falls back to local
emulation so the dashboard always renders during demos.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Configurable via env so the same build can point at a different node/API
# without a code change (staging vs. production edge nodes, etc).
API_BASE = os.environ.get("ISIP_API_BASE", "http://127.0.0.1:8080/edge")
API_KEY = os.environ.get("ISIP_API_KEY")  # must match the server's key
MAX_POINTS = 120
_AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(
    page_title="ISIP // Operator Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# Small amount of global CSS so section cards have breathing room and don't
# visually run into one another.
st.markdown(
    """
    <style>
      div[data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 0.75rem; }
      div[data-testid="stMetric"] { overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=2, show_spinner=False)
def api_get(path: str) -> Dict:
    """Read-only GETs are safe to cache briefly to cut request volume."""
    try:
        resp = requests.get(f"{API_BASE}{path}", headers=_AUTH_HEADERS, timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("GET %s failed: %s", path, exc)
        return {}


def api_post(path: str, body: Dict | None = None) -> Dict:
    """Mutating calls (E-Stop trigger/release) must NEVER be cached — a
    cached response here would mean a control action doesn't reliably reach
    the hardware. This must stay a plain, uncached function."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=body or {}, headers=_AUTH_HEADERS, timeout=5)
        if resp.status_code == 401:
            return {"status": "ERROR", "detail": "Unauthorized — check ISIP_API_KEY"}
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("POST %s failed: %s", path, exc)
        return {"status": "ERROR", "detail": str(exc)}


def fallback_metrics() -> Dict:
    return {
        "latency_ms": round(13.0 + random.random() * 2.5, 1),
        "fps": round(68 + random.random() * 5, 1),
        "alerts": st.session_state.get("fallback_alerts", 0),
        "is_estop_locked": False,
        "uptime_s": time.time() - st.session_state.get("boot", time.time()),
        "online": False,
    }


def fallback_sample() -> Dict:
    return {
        "timestamp": time.time(),
        "sensor_id": random.choice(["TMP_BRG_02", "VIB_MTR_01", "GAS_CO_03"]),
        "metric": random.choice(["temperature", "vibration", "co_ppm"]),
        "value": round(random.uniform(38, 78), 2),
        "unit": random.choice(["C", "Hz", "ppm"]),
        "status": "NORMAL",
    }


def render_kpis(metrics: Dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    c1.metric("Latency", f"{metrics.get('latency_ms', 0):.1f} ms")
    c2.metric("FPS", f"{metrics.get('fps', 0):.1f}")
    c3.metric("Alerts", metrics.get("alerts", 0))
    c4.metric("Uptime", f"{metrics.get('uptime_s', 0):.0f}s")
    estop = metrics.get("is_estop_locked", False)
    c5.metric("E-Stop", "LOCKED" if estop else "ARMED")


def render_status_banner(metrics: Dict) -> None:
    online = metrics.get("online", False)
    stamp = time.strftime("%H:%M:%S")
    if online:
        st.success(
            f"**EDGE NODE ONLINE** — live data from `{API_BASE}` · "
            f"node `{metrics.get('node_id', '?')}` · checked {stamp}"
        )
    else:
        st.warning(
            f"**API OFFLINE** — could not reach `{API_BASE}` · showing **local "
            f"emulation** (not live) · checked {stamp}"
        )


def render_system_check(metrics: Dict) -> None:
    with st.expander("System Check (live diagnostic)", expanded=False):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        c1.metric("Edge Node", metrics.get("node_id", "—"))
        c2.metric("Uptime", f"{metrics.get('uptime_s', 0):.0f}s")
        c3.metric("Latency", f"{metrics.get('latency_ms', 0):.1f} ms")
        c4.metric("Inferences", metrics.get("inferences", 0))
        st.caption(
            f"SSE stream `{API_BASE}/stream` · last refresh "
            f"{time.strftime('%H:%M:%S')} · data source: "
            f"{'**LIVE**' if metrics.get('online') else '**EMULATED**'}"
        )


def build_telemetry_chart(samples: List[Dict]) -> go.Figure:
    df = pd.DataFrame(samples)
    fig = go.Figure()
    if not df.empty:
        for metric in ["temperature", "vibration", "co_ppm"]:
            sub = df[df["metric"] == metric]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub["timestamp"],
                        y=sub["value"],
                        mode="lines",
                        name=metric,
                        line=dict(width=2),
                    )
                )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="time",
        yaxis_title="value",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def rul_gauge(health_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_pct,
            title={"text": "RUL Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#10b981"},
                "steps": [
                    {"range": [0, 20], "color": "#7f1d1d"},
                    {"range": [20, 50], "color": "#92400e"},
                    {"range": [50, 100], "color": "#064e3b"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# Single video component: the event log is now a fixed strip BELOW the video
# (with its own solid background) instead of floating on top of it, so text
# never overlaps the picture.
_SSE_HTML = """
<div id="isip-sse-root" style="width:100%;border-radius:8px;overflow:hidden;background:#0b1118;">
  <div style="position:relative;width:100%;height:420px;background:#000;">
    <img id="isip-video" src="{video_url}" style="width:100%;height:100%;object-fit:contain;display:block;" />
  </div>
  <div id="isip-events" style="width:100%;height:130px;overflow-y:auto;font:11px/1.5 monospace;
       background:#0b1118;border-top:1px solid #1e293b;padding:6px 8px;box-sizing:border-box;"></div>
</div>
<script>
  (function() {{
    const streamUrl = "{stream_url}";
    const img = document.getElementById("isip-video");
    const evts = document.getElementById("isip-events");
    if (!img || !evts) return;
    img.onerror = function() {{
      img.style.display = "none";
      evts.innerHTML = '<div style="color:#f87171">VIDEO FEED UNAVAILABLE</div>';
    }};
    const es = new EventSource(streamUrl);
    es.onmessage = function(e) {{
      try {{
        const d = JSON.parse(e.data);
        const div = document.createElement("div");
        const sev = d.severity === "CRITICAL" ? "#f87171" : d.severity === "WARNING" ? "#fbbf24" : "#94a3b8";
        div.style.color = sev;
        div.textContent = "> " + d.event_type + " [" + d.severity + "] " + (d.latency_ms ? d.latency_ms.toFixed(1)+"ms " : "") + JSON.stringify(d.payload || {{}}).slice(0, 120);
        evts.appendChild(div);
        while (evts.children.length > 80) evts.removeChild(evts.firstChild);
        evts.scrollTop = evts.scrollHeight;
      }} catch(err) {{}}
    }};
    es.onerror = function() {{
      const div = document.createElement("div");
      div.style.color = "#fbbf24";
      div.textContent = "SSE RECONNECTING...";
      evts.appendChild(div);
      evts.scrollTop = evts.scrollHeight;
    }};
  }})();
</script>
"""


def render_video_panel() -> None:
    st.components.v1.html(
        _SSE_HTML.format(
            video_url=f"{API_BASE}/video-feed",
            stream_url=f"{API_BASE}/stream",
        ),
        height=560,
        scrolling=False,
    )


def main() -> None:
    st.title("🛡️ ISIP — Operator Dashboard")
    st.caption("Industrial Safety Intelligence Platform · Edge Node 01 · OFFLINE EDGE RESILIENT")

    logger.debug("dashboard render started")
    metrics = api_get("/metrics")
    if not metrics:
        metrics = fallback_metrics()
        st.session_state["fallback_alerts"] = metrics["alerts"]
        metrics["online"] = False
        logger.warning("API unreachable, using fallback metrics")
    else:
        metrics["online"] = True
        logger.debug("API reachable, latency=%.1fms fps=%.1f", metrics.get("latency_ms", 0), metrics.get("fps", 0))

    with st.sidebar:
        st.header("Control Plane")
        if st.button("🚨 TRIGGER E-STOP", type="primary", use_container_width=True, key="btn_estop_trigger"):
            result = api_post("/trigger-estop", {"mode": "HARD"})
            if result.get("status") == "ERROR":
                st.error(f"E-Stop trigger failed: {result.get('detail', 'unknown error')}")
            else:
                st.write(result)
        if st.button("Release E-Stop", use_container_width=True, key="btn_estop_release"):
            result = api_post("/release-estop")
            if result.get("status") == "ERROR":
                st.error(f"E-Stop release failed: {result.get('detail', 'unknown error')}")
            else:
                st.write(result)
        st.divider()
        if st.button("Run connection check", use_container_width=True, key="btn_conn_check"):
            try:
                check = requests.get(f"{API_BASE}/health", timeout=2).json()
                st.success(f"CONNECTED — {check}")
            except requests.RequestException as exc:
                st.error(f"UNREACHABLE — {API_BASE} ({exc.__class__.__name__})")
        st.divider()
        st.subheader("Audit Ledger")
        for row in api_get("/audit?limit=8") or []:
            st.code(f"{row.get('event_type', '?')} [{row.get('severity', '?')}]", language=None)

    render_status_banner(metrics)

    with st.container(border=True):
        render_kpis(metrics)
        render_system_check(metrics)

    events = api_get("/events?limit=200") or []

    # Telemetry + RUL, each in its own bordered card so they don't visually
    # bleed together; gauge column widened so its number/title has room.
    col1, col2 = st.columns([2, 1], gap="medium")
    with col1:
        with st.container(border=True):
            st.subheader("IIoT Telemetry — Live")
            samples = [
                e["payload"]
                for e in events
                if e.get("event_type") == "TELEMETRY" and "payload" in e
            ]
            if not samples:
                samples = [fallback_sample() for _ in range(30)]
            samples = samples[-MAX_POINTS:]
            st.plotly_chart(build_telemetry_chart(samples), use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("RUL Estimation")
            rul_events = [
                e["payload"]
                for e in events
                if e.get("event_type") == "RUL_UPDATE" and "payload" in e
            ]
            health = rul_events[-1].get("health_pct", 100.0) if rul_events else 100.0
            if rul_events:
                last = rul_events[-1]
                st.metric("Remaining Life", f"{last.get('remaining_hours', 0):,.0f} h")
            st.plotly_chart(rul_gauge(health), use_container_width=True)

    with st.container(border=True):
        st.subheader("Live Detection Video Feed")
        render_video_panel()

    with st.container(border=True):
        st.subheader("Safety Events")
        rows = [
            {
                "time": pd.to_datetime(e.get("timestamp", 0), unit="s").strftime("%H:%M:%S"),
                "event_type": e.get("event_type"),
                "severity": e.get("severity"),
                "latency_ms": e.get("latency_ms"),
            }
            for e in events[:50]
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
        else:
            st.info("No events yet — start the edge orchestrator.")


if __name__ == "__main__":
    main()