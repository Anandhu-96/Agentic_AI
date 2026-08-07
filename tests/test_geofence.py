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
    assert engine.locate((0.5, 0.5)).name == "ZONE_A"
    assert engine.locate((0.99, 0.99)) is None
