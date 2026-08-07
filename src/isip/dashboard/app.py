"""Streamlit operator dashboard.

Live video overlay + telemetry monitoring UI. Polls the edge control plane
(FastAPI). When the API is unreachable it falls back to local emulation so the
dashboard always renders during demos.
"""

from __future__ import annotations

import math
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
            f"Polling `{API_BASE}/metrics` · last refresh "
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

    # Telemetry + RUL
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("IIoT Telemetry — Live")
        events = api_get("/events?limit=200") or []
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

    st.divider()
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
