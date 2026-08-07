from isip.orchestrator import _STATUS_SEVERITY


def test_all_sample_statuses_map_to_severity():
    """Every status the telemetry/RUL loops can emit must map to a Severity,
    otherwise the edge node crashes."""
    for status in ("NORMAL", "HEALTHY", "WARN", "CRITICAL"):
        assert _STATUS_SEVERITY[status] is not None