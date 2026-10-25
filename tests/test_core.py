import math

from tempo.kitchen import simulate
from tempo.physics import BOIL_C, BoilEstimator, heat_params, nominal_boil_s, temp_at, time_to


def test_heating_model_is_consistent():
    T_inf, tau = heat_params(550, 2.4)
    tb = time_to(BOIL_C, 17.0, T_inf, tau)
    assert 150 < tb < 400
    assert math.isclose(temp_at(tb, 17.0, T_inf, tau), BOIL_C, abs_tol=1e-6)


def test_estimator_is_exact_on_linear_heating_after_calibration():
    # linear heating 17 C -> 99.5 C in 300 s; stages calibrated to the same line
    rate = (BOIL_C - 17.0) / 300.0
    st = {1: 17.0 + rate * 180, 2: 17.0 + rate * 250}
    est = BoilEstimator(0.0, prior_s=200.0, curvature=1.0)
    est.stage_temp.update(st)
    est.observe(1, 180.0)
    assert abs(est.eta(181.0)["t_boil"] - 300.0) < 1.0
    est.observe(2, 250.0)
    assert abs(est.eta(251.0)["t_boil"] - 300.0) < 1.0


def test_estimator_never_predicts_the_past():
    est = BoilEstimator(0.0, prior_s=100.0)
    assert est.eta(500.0)["t_boil"] >= 502.0


def test_every_policy_serves_every_ticket():
    for p in ("fifo", "plan_fixed", "plan_alert", "tempo"):
        m, k = simulate(3, p)
        assert all("served" in t for t in k.tickets.values()), p
        assert m["makespan_s"] < 4 * 3600


def test_tempo_prevents_most_boilovers():
    tempo = sum(simulate(s, "tempo")[0]["boilovers"] for s in range(20))
    fifo = sum(simulate(s, "fifo")[0]["boilovers"] for s in range(20))
    assert tempo * 4 <= fifo


def test_nominal_boil_monotone_in_power_and_volume():
    assert nominal_boil_s(500, 3.0) < nominal_boil_s(500, 2.0) < nominal_boil_s(1000, 2.0)
