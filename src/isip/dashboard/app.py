"""Streamlit operator dashboard.

Live video overlay + telemetry monitoring UI. Uses SSE for real-time events
instead of REST polling. When the API is unreachable it falls back to local
emulation so the dashboard always renders during demos.
"""

from __future__ import annotations

import json
import random
import time
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8080/edge"
MAX_POINTS = 120

st.set_page_config(
    page_title="ISIP // Operator Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(ttl=2, show_spinner=False)
def api_get(path: str) -> Dict:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {}


@st.cache_data(ttl=2, show_spinner=False)
def api_post(path: str, body: Dict | None = None) -> Dict:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=body or {}, timeout=2)
        return resp.json()
    except requests.RequestException:
        return {}


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
    c1, c2, c3, c4, c5 = st.columns(5)
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
        c1, c2, c3, c4 = st.columns(4)
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
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="time",
        yaxis_title="value",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
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
    fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_zone_status(events: List[Dict]) -> None:
    zone_events = [
        e for e in events
        if e.get("event_type") in ("ZONE_INTRUSION", "ZONE_CLEARED") and "payload" in e
    ]
    active_zones = {}
    for e in zone_events:
        payload = e.get("payload", {})
        zone_name = payload.get("zone", "UNKNOWN")
        event_type = e.get("event_type")
        if event_type == "ZONE_INTRUSION":
            active_zones[zone_name] = {
                "status": "ACTIVE",
                "severity": "CRITICAL" if e.get("severity") == "CRITICAL" else "WARNING",
                "description": payload.get("description", ""),
            }
        elif event_type == "ZONE_CLEARED" and zone_name in active_zones:
            del active_zones[zone_name]

    if not active_zones:
        st.success("All zones clear — no active intrusions")
        return

    for zone_name, info in active_zones.items():
        severity_color = "🔴" if info["severity"] == "CRITICAL" else "🟡"
        st.error(
            f"{severity_color} **{zone_name}** — {info['status']}  \n"
            f"{info['description']}"
        )


def render_critical_alerts(events: List[Dict]) -> None:
    critical_events = [
        e for e in events
        if e.get("severity") == "CRITICAL" and "payload" in e
    ]
    if not critical_events:
        return

    st.markdown("---")
    st.subheader("🚨 Critical Alerts")
    for e in critical_events[-10:]:
        payload = e.get("payload", {})
        event_type = e.get("event_type", "UNKNOWN")
        latency = e.get("latency_ms", 0)
        timestamp = pd.to_datetime(e.get("timestamp", 0), unit="s").strftime("%H:%M:%S")

        if event_type == "RELAY_TRIP":
            st.error(
                f"**{timestamp}** — RELAY TRIPPED | "
                f"Reason: `{payload.get('reason', 'unknown')}` | "
                f"Latency: {latency:.1f}ms"
            )
        elif event_type == "ZONE_INTRUSION":
            st.error(
                f"**{timestamp}** — ZONE BREACH | "
                f"Zone: `{payload.get('zone', 'UNKNOWN')}` | "
                f"Confidence: {payload.get('confidence', 0):.2f}"
            )
        elif event_type == "THERMAL_ANOMALY":
            st.error(
                f"**{timestamp}** — THERMAL ANOMALY | "
                f"Sensor: `{payload.get('sensor_id', 'UNKNOWN')}` | "
                f"Value: {payload.get('value', 0):.1f}{payload.get('unit', '')}"
            )
        elif event_type == "E_STOP_TRIGGERED":
            st.error(
                f"**{timestamp}** — E-STOP TRIGGERED | "
                f"Mode: `{payload.get('mode', 'UNKNOWN')}` | "
                f"Relay: {payload.get('relay', 'UNKNOWN')}"
            )
        else:
            st.error(
                f"**{timestamp}** — {event_type} | "
                f"Payload: `{str(payload)[:120]}`"
            )


_SSE_HTML = """
<div id="isip-sse-root" style="width:100%;height:500px;border-radius:8px;overflow:hidden;background:#0b1118;position:relative;">
  <img id="isip-video" src="{video_url}" style="width:100%;height:100%;object-fit:contain;display:block;" />
  <div id="isip-events" style="position:absolute;bottom:8px;left:8px;right:8px;max-height:120px;overflow-y:auto;font:10px monospace;pointer-events:none;"></div>
</div>
<script>
  (function() {{
    const videoUrl = "{video_url}";
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
      evts.innerHTML = '<div style="color:#fbbf24">SSE RECONNECTING...</div>';
    }};
  }})();
</script>
"""


def main() -> None:
    st.title("🛡️ ISIP — Operator Dashboard")
    st.caption("Industrial Safety Intelligence Platform · Edge Node 01 · OFFLINE EDGE RESILIENT")

    metrics = api_get("/metrics")
    if not metrics:
        metrics = fallback_metrics()
        st.session_state["fallback_alerts"] = metrics["alerts"]
        metrics["online"] = False
    else:
        metrics["online"] = True

    with st.sidebar:
        st.header("Control Plane")
        if st.button("🚨 TRIGGER E-STOP", type="primary", use_container_width=True):
            result = api_post("/trigger-estop", {"mode": "HARD"})
            st.write(result or {"status": "emulated trip"})
        if st.button("Release E-Stop", use_container_width=True):
            result = api_post("/release-estop")
            st.write(result or {"status": "released (emulated)"})
        st.divider()
        if st.button("Run connection check", use_container_width=True):
            try:
                check = requests.get(f"{API_BASE}/health", timeout=2).json()
                st.success(f"CONNECTED — {check}")
            except requests.RequestException as exc:
                st.error(f"UNREACHABLE — {API_BASE} ({exc.__class__.__name__})")
        st.divider()
        st.subheader("Audit Ledger")
        for row in api_get("/audit?limit=8"):
            st.code(
                f"{row['event_type']} [{row['severity']}]",
                language=None,
            )

    render_status_banner(metrics)
    render_kpis(metrics)
    render_system_check(metrics)

    events = api_get("/events?limit=200") or []

    # Critical Alerts Section
    render_critical_alerts(events)

    # Detection Video + Zone Status
    col_video, col_zone = st.columns([3, 1])
    with col_video:
        st.subheader("🎥 Detection Feed — YOLO Inference")
        snapshot_url = f"{API_BASE}/video-snapshot?t={int(time.time())}"
        st.image(snapshot_url, use_container_width=True)

    with col_zone:
        st.subheader("🗺️ Zone Intrusion Status")
        render_zone_status(events)

        st.divider()
        st.subheader("📊 IIoT Telemetry — Live")
        samples = [
            e["payload"]
            for e in events
            if e.get("event_type") == "TELEMETRY" and "payload" in e
        ]
        if not samples:
            samples = [fallback_sample() for _ in range(30)]
        samples = samples[-MAX_POINTS:]
        st.plotly_chart(build_telemetry_chart(samples), use_container_width=True)

    # RUL + Safety Events
    col_rul, col_events = st.columns([1, 2])
    with col_rul:
        st.subheader("RUL Estimation")
        rul_events = [
            e["payload"]
            for e in events
            if e.get("event_type") == "RUL_UPDATE" and "payload" in e
        ]
        health = rul_events[-1].get("health_pct", 100.0) if rul_events else 100.0
        st.plotly_chart(rul_gauge(health), use_container_width=True)
        if rul_events:
            last = rul_events[-1]
            st.metric("Remaining Life", f"{last.get('remaining_hours', 0):,.0f} h")

    with col_events:
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

    st.divider()
    st.subheader("Live Detection Video Feed")
    snapshot_url = f"{API_BASE}/video-snapshot?t={int(time.time())}"
    st.image(snapshot_url, use_container_width=True)


if __name__ == "__main__":
    main()
