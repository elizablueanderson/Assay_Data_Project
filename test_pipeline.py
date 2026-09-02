"""
Tests for the analysis code.

The approach throughout: simulate data from parameters that are known exactly,
then check the fitting code recovers them. A curve-fitting routine that has
never been run against a known answer has not been tested, only executed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import calibration as cal_mod
from src import kinetics as kin_mod
from src import qc as qc_mod


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_linear_fit_recovers_known_parameters():
    x = np.repeat([0.0, 0.25, 0.5, 1.0, 2.0], 3)
    y = 0.6 * x + 0.04  # noiseless

    cal = cal_mod.fit_linear(x, y)

    assert cal.slope == pytest.approx(0.6, rel=1e-9)
    assert cal.intercept == pytest.approx(0.04, abs=1e-9)
    assert cal.r_squared == pytest.approx(1.0, abs=1e-9)
    assert cal.n == 15


def test_fit_rejects_insufficient_points():
    with pytest.raises(ValueError, match="at least 3"):
        cal_mod.fit_linear([0.0, 1.0], [0.0, 0.6])


def test_lod_loq_match_ich_formula():
    """LOD = 3.3*sigma/S and LOQ = 10*sigma/S, computed by hand."""
    rng = np.random.default_rng(7)
    x = np.repeat([0.0, 0.25, 0.5, 1.0, 2.0], 3)
    y = 0.6 * x + 0.04 + rng.normal(0, 0.01, x.size)

    cal = cal_mod.fit_linear(x, y)
    limits = cal_mod.detection_limits(cal, sigma_source="residual")

    assert limits["lod"] == pytest.approx(3.3 * cal.residual_sd / cal.slope)
    assert limits["loq"] == pytest.approx(10.0 * cal.residual_sd / cal.slope)
    assert limits["loq"] > limits["lod"]


def test_lod_scales_with_noise():
    """Noisier data must give a worse (higher) detection limit."""
    x = np.repeat([0.0, 0.25, 0.5, 1.0, 2.0], 3)
    rng = np.random.default_rng(11)

    quiet = cal_mod.fit_linear(x, 0.6 * x + rng.normal(0, 0.005, x.size))
    noisy = cal_mod.fit_linear(x, 0.6 * x + rng.normal(0, 0.050, x.size))

    lod_quiet = cal_mod.detection_limits(quiet)["lod"]
    lod_noisy = cal_mod.detection_limits(noisy)["lod"]

    assert lod_noisy > lod_quiet


def test_blank_sigma_requires_blanks():
    cal = cal_mod.fit_linear([0, 1, 2, 3], [0.0, 0.6, 1.2, 1.8])
    with pytest.raises(ValueError, match="requires blank_signal"):
        cal_mod.detection_limits(cal, sigma_source="blank")


# --------------------------------------------------------------------------
# Inverse prediction
# --------------------------------------------------------------------------


def test_inverse_prediction_round_trips():
    x = np.repeat([0.0, 0.25, 0.5, 1.0, 2.0], 3)
    cal = cal_mod.fit_linear(x, 0.6 * x + 0.04)

    known = np.array([0.3, 0.8, 1.5])
    signals = cal.predict(known)
    result = cal_mod.inverse_predict(cal, signals)

    np.testing.assert_allclose(result["concentration"], known, rtol=1e-8)


def test_confidence_interval_widens_away_from_centroid():
    """
    The defining property of inverse prediction: uncertainty is minimised at
    the centre of the calibration standards and grows toward either end.
    """
    rng = np.random.default_rng(3)
    x = np.repeat([0.0, 0.5, 1.0, 1.5, 2.0], 3)
    cal = cal_mod.fit_linear(x, 0.6 * x + 0.04 + rng.normal(0, 0.01, x.size))

    at_centre = cal_mod.inverse_predict(cal, cal.predict(cal.x_mean))
    at_edge = cal_mod.inverse_predict(cal, cal.predict(2.0))

    assert at_edge["se"][0] > at_centre["se"][0]


def test_more_replicates_narrow_the_interval():
    rng = np.random.default_rng(5)
    x = np.repeat([0.0, 0.5, 1.0, 1.5, 2.0], 3)
    cal = cal_mod.fit_linear(x, 0.6 * x + 0.04 + rng.normal(0, 0.01, x.size))

    single = cal_mod.inverse_predict(cal, 0.64, n_replicates=1)
    triple = cal_mod.inverse_predict(cal, 0.64, n_replicates=3)

    assert triple["se"][0] < single["se"][0]


# --------------------------------------------------------------------------
# Working range
# --------------------------------------------------------------------------


def test_working_range_excludes_poor_recovery():
    """A deliberately biased low standard must fall out of the range."""
    x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    y = 0.6 * x + 0.04
    y[:3] += 0.15  # inflate the bottom level well beyond tolerance

    cal = cal_mod.fit_linear(x, y)
    wr = cal_mod.working_range(cal, tolerance=0.20)

    levels = {row["nominal"]: row["in_range"] for row in wr["levels"]}
    assert levels[0.1] is False
    assert wr["range_low"] > 0.1


# --------------------------------------------------------------------------
# Kinetics
# --------------------------------------------------------------------------


def test_michaelis_menten_recovers_known_parameters():
    s = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    v = kin_mod.michaelis_menten(s, vmax=0.85, km=3.2)

    fit = kin_mod.fit_kinetics(s, v, model="mm")["best"]

    assert fit.params["vmax"] == pytest.approx(0.85, rel=1e-4)
    assert fit.params["km"] == pytest.approx(3.2, rel=1e-4)


def test_kinetics_requires_enough_substrate_levels():
    with pytest.raises(ValueError, match="at least 4"):
        kin_mod.fit_kinetics([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])


def test_unidentifiable_ki_is_rejected():
    """
    Data generated without inhibition must not be reported as inhibited, even
    when the extra parameter happens to lower the AIC.
    """
    s = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    rng = np.random.default_rng(42)
    v = kin_mod.michaelis_menten(s, 0.85, 3.2) + rng.normal(0, 0.004, s.size)

    result = kin_mod.fit_kinetics(s, v, model="auto")

    assert result["best"].model == "michaelis-menten"


def test_real_inhibition_is_detected():
    """The guard must not be so strict that genuine inhibition is missed."""
    s = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    v = kin_mod.substrate_inhibition(s, vmax=1.2, km=3.0, ki=25.0)

    result = kin_mod.fit_kinetics(s, v, model="auto")

    assert result["best"].model == "substrate-inhibition"
    assert result["best"].params["ki"] == pytest.approx(25.0, rel=1e-3)


def test_initial_rate_restricts_to_early_window():
    """Fitting the whole progress curve would underestimate the rate."""
    t = np.array([0, 2, 4, 6, 8, 10, 15, 20], dtype=float)
    true_rate = 0.5
    product = true_rate * t * (1 - 0.02 * t)  # visible depletion curvature

    est = kin_mod.initial_rate(t, product, substrate_umol=10.0)

    assert est["n_points_used"] < t.size
    assert est["rate_umol_per_min"] > 0.9 * true_rate


def test_specific_activity_rejects_zero_biomass():
    with pytest.raises(ValueError, match="must be positive"):
        kin_mod.specific_activity([0.5], [0.0])


def test_activity_retention_is_a_ratio():
    retention = kin_mod.activity_retention([0.32], [0.64])
    assert retention[0] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------


def test_z_prime_perfect_separation():
    """Zero-variance controls with a wide window give Z' approaching 1."""
    result = qc_mod.z_prime([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert result["z_prime"] == pytest.approx(1.0)
    assert result["interpretation"] == "excellent"


def test_z_prime_flags_overlapping_controls():
    rng = np.random.default_rng(1)
    pos = rng.normal(1.0, 0.4, 8)
    neg = rng.normal(0.9, 0.4, 8)

    assert qc_mod.z_prime(pos, neg)["z_prime"] < 0


def test_outlier_detection_uses_median_not_mean():
    """One extreme replicate must be flagged without dragging the centre."""
    values = [0.50, 0.51, 0.49, 0.52, 2.50]
    result = qc_mod.flag_outliers(values)

    assert result["is_outlier"][-1]
    assert not result["is_outlier"][:-1].any()


def test_spike_recovery_flags_matrix_effect():
    result = qc_mod.spike_recovery(measured=[0.55], nominal_spike=[1.0])
    assert not result["all_pass"]
    assert result["recovery_pct"][0] == pytest.approx(55.0)
