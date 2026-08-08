from isip.vision.geofence import GeofenceEngine, point_in_polygon

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_point_inside():
    assert point_in_polygon((0.5, 0.5), SQUARE)


def test_point_outside():
    assert not point_in_polygon((1.5, 0.5), SQUARE)


def test_point_on_edge_counts_inside():
    assert point_in_polygon((0.5, 1.0), SQUARE)


def test_point_on_vertex_counts_inside():
    assert point_in_polygon((1.0, 0.0), SQUARE)


def test_concave_polygon_correct():
    concave = [(0, 0), (2, 0), (2, 2), (1, 1), (0, 2)]
    assert point_in_polygon((0.2, 1.8), concave)     # on left diagonal edge
    assert point_in_polygon((1.0, 0.5), concave)     # below notch, inside
    assert point_in_polygon((0.75, 0.75), concave)   # below notch, inside
    assert not point_in_polygon((1.0, 1.5), concave)  # inside notch cut-out
    assert not point_in_polygon((0.5, 1.8), concave)  # top triangle, outside
    assert not point_in_polygon((2.5, 1.0), concave)  # beyond right edge


def test_engine_from_yaml(tmp_path):
    import yaml

    p = tmp_path / "geofences.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "zones": {
                    "ZONE_A": {
                        "severity": "CRITICAL",
                        "action": "TRIP_RELAY",
                        "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    }
                }
            }
        )
    )
    engine = GeofenceEngine.from_yaml(p)
    assert engine.locate_point((0.5, 0.5)).name == "ZONE_A"
    assert engine.locate_point((0.99, 0.99)) is None


def test_as_pixels_scales_normalized_polygon():
    zone = {"severity": "CRITICAL", "polygon": [[0.25, 0.5], [0.75, 0.5], [0.75, 1.0], [0.25, 1.0]]}
    engine = GeofenceEngine({"ZONE_A": zone})
    assert engine.zones["ZONE_A"].as_pixels(1280, 720) == [(320, 360), (960, 360), (960, 720), (320, 720)]


def test_overlay_polys_includes_geometry_and_metadata():
    engine = GeofenceEngine({
        "ZONE_A": {"severity": "CRITICAL", "action": "TRIP_RELAY", "polygon": [[0, 0], [1, 0], [1, 1]]}
    })
    out = engine.overlay_polys(w=100, h=50)
    assert "ZONE_A" in out
    assert out["ZONE_A"]["severity"] == "CRITICAL"
    assert out["ZONE_A"]["action"] == "TRIP_RELAY"
    assert out["ZONE_A"]["polygon"][0] == [0.0, 0.0]
    assert out["ZONE_A"]["polygon"][2] == [100.0, 50.0]

    # normalized mode returns raw 0..1 polygon
    normalized = engine.overlay_polys(normalize=True)
    assert normalized["ZONE_A"]["polygon"] == [[0, 0], [1, 0], [1, 1]]
