from isip.config import RulConfig
from isip.iiot.rul import ThermalRulEstimator


def test_no_damage_below_nominal():
    estimator = ThermalRulEstimator(RulConfig(nominal_temp_c=60.0, max_rated_life_h=1000.0))
    assert estimator._acceleration_factor(60.0, 60.0) == 1.0
    assert estimator._acceleration_factor(50.0, 60.0) == 1.0


def test_acceleration_doubles_per_10c():
    estimator = ThermalRulEstimator(RulConfig(nominal_temp_c=60.0))
    af_70 = estimator._acceleration_factor(70.0, 60.0)
    af_80 = estimator._acceleration_factor(80.0, 60.0)
    assert af_70 == 2.0
    assert abs(af_80 - 4.0) < 1e-9


def test_consumption_reduces_health():
    estimator = ThermalRulEstimator(RulConfig(nominal_temp_c=60.0, max_rated_life_h=1000.0))
    estimator.consume(60.0, 500.0)
    state = estimator.observe(60.0)
    assert state.health_pct == 50.0
    assert state.remaining_hours == 500.0


def test_status_escalates():
    estimator = ThermalRulEstimator(RulConfig(nominal_temp_c=60.0, max_rated_life_h=100.0))
    estimator.consume(100.0, 99.0)  # 40C above nominal -> extreme aging
    state = estimator.observe(100.0)
    assert state.status == "CRITICAL"
