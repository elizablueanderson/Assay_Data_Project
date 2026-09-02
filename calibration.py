"""
Standard-curve calibration for assay

Implements the analytical steps required to turn raw absorbance into a
defensible concentration estimate:

  1. Regression of the calibration standards (linear or 4-parameter logistic)
  2. Detection limits following ICH Q2(R1)
  3. Inverse prediction of unknowns, with propagated uncertainty
  4. Working-range determination by back calculated recovery


from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


# Linear calibration


@dataclass
class LinearCalibration:
    """A fitted linear standard curve with full regression statistics."""

    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    residual_sd: float  # s_(y/x), the standard error of the regression
    r_squared: float
    n: int
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    @property
    def x_mean(self) -> float:
        return float(np.mean(self.x))

    @property
    def sxx(self) -> float:
        """Sum of squared deviations of x. Drives how fast the CI widens."""
        return float(np.sum((self.x - self.x_mean) ** 2))

    def predict(self, concentration):
        """Forward: concentration -> expected signal."""
        return self.slope * np.asarray(concentration, dtype=float) + self.intercept


def fit_linear(concentration, signal) -> LinearCalibration:
    """
    Ordinary least-squares fit of signal on concentration.

    Pass every replicate as its own row rather than pre-averaging. The
    replicate scatter is what defines the residual standard deviation, and
    the residual SD is what sets the detection limits downstream.
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(signal, dtype=float)

    if x.size != y.size:
        raise ValueError("concentration and signal must be the same length")
    if x.size < 3:
        raise ValueError("need at least 3 calibration points to estimate error")

    result = stats.linregress(x, y)

    # linregress gives us the SE of the slope directly; recover the residual
    # SD from it, then use that for the intercept SE.
    dof = x.size - 2
    residuals = y - (result.slope * x + result.intercept)
    residual_sd = float(np.sqrt(np.sum(residuals**2) / dof))

    sxx = float(np.sum((x - np.mean(x)) ** 2))
    se_intercept = residual_sd * np.sqrt(1.0 / x.size + np.mean(x) ** 2 / sxx)

    return LinearCalibration(
        slope=float(result.slope),
        intercept=float(result.intercept),
        se_slope=float(result.stderr),
        se_intercept=float(se_intercept),
        residual_sd=residual_sd,
        r_squared=float(result.rvalue**2),
        n=int(x.size),
        x=x,
        y=y,
    )



# Detection limits



def detection_limits(cal: LinearCalibration, sigma_source="residual", blank_signal=None):
    """
    LOD and LOQ following ICH Q2(R1):

        LOD = 3.3 * sigma / S
        LOQ = 10.0 * sigma / S

    where S is the calibration slope and sigma is the standard deviation of
    the response. ICH permits three estimates of sigma, and they do not agree
    with each other, so the choice has to be stated explicitly in any method
    write-up rather than left implicit:

      "residual"  - residual SD of the regression (default; uses the whole curve)
      "intercept" - standard error of the y-intercept (most conservative)
      "blank"     - SD of replicate blank measurements (requires blank_signal)

    Reporting an LOD without naming the sigma source makes the number
    impossible to reproduce, which is the usual reason a reviewer sends a
    method back.
    """
    if sigma_source == "residual":
        sigma = cal.residual_sd
    elif sigma_source == "intercept":
        sigma = cal.se_intercept
    elif sigma_source == "blank":
        if blank_signal is None:
            raise ValueError("sigma_source='blank' requires blank_signal replicates")
        blank = np.asarray(blank_signal, dtype=float)
        if blank.size < 3:
            raise ValueError("need at least 3 blank replicates")
        sigma = float(np.std(blank, ddof=1))
    else:
        raise ValueError(f"unknown sigma_source: {sigma_source!r}")

    slope = abs(cal.slope)
    if slope == 0:
        raise ValueError("slope is zero; detection limits are undefined")

    return {
        "sigma_source": sigma_source,
        "sigma": sigma,
        "slope": cal.slope,
        "lod": 3.3 * sigma / slope,
        "loq": 10.0 * sigma / slope,
    }



# Inverse prediction



def inverse_predict(cal: LinearCalibration, signal, n_replicates=1, alpha=0.05):
    """
    Back-calculate concentration from measured signal, with a confidence
    interval that accounts for error in both the calibration and the unknown.

    Point estimate:

        x_hat = (y - b) / m

    Standard error (Draper & Smith, classical inverse prediction):

        se = (s / m) * sqrt( 1/M + 1/n + (x_hat - x_bar)^2 / Sxx )

    with s the residual SD, M the replicates of the unknown, and n the number
    of calibration points. The third term is the important one: uncertainty is
    smallest at the centre of the standards and widens toward either end. A
    sample read near the top of the curve carries materially more error than
    one read mid-range, even when the R-squared looks immaculate.

    Returns point estimate, SE, and CI bounds for each input signal.
    """
    y = np.atleast_1d(np.asarray(signal, dtype=float))
    m, b, s = cal.slope, cal.intercept, cal.residual_sd

    x_hat = (y - b) / m

    dof = cal.n - 2
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, dof)

    variance_term = (
        1.0 / n_replicates + 1.0 / cal.n + (x_hat - cal.x_mean) ** 2 / cal.sxx
    )
    se = (s / abs(m)) * np.sqrt(variance_term)

    return {
        "signal": y,
        "concentration": x_hat,
        "se": se,
        "ci_lower": x_hat - t_crit * se,
        "ci_upper": x_hat + t_crit * se,
        "alpha": alpha,
    }



# Working range



def working_range(cal: LinearCalibration, tolerance=0.20, loq=None):
    """
    Identify which standards back-calculate to within +/- tolerance of their
    nominal value. This is the accuracy-based definition of the working range
    and is stricter than eyeballing R-squared.

    A curve can post R-squared = 0.999 and still recover the bottom standard
    at 60% of nominal, because R-squared is dominated by the high-signal
    points. Recovery is computed per level and judged per level.

    Levels at or below the LOQ are reported but excluded from the range, since
    quantitation is not claimed below the LOQ by definition.
    """
    levels = np.unique(cal.x)
    levels = levels[levels > 0]  # recovery is undefined at nominal zero

    rows = []
    for level in levels:
        mask = cal.x == level
        mean_signal = float(np.mean(cal.y[mask]))
        back_calc = (mean_signal - cal.intercept) / cal.slope
        recovery = 100.0 * back_calc / level

        within_tol = abs(recovery - 100.0) <= tolerance * 100.0
        above_loq = True if loq is None else level >= loq

        rows.append(
            {
                "nominal": float(level),
                "mean_signal": mean_signal,
                "back_calculated": float(back_calc),
                "recovery_pct": float(recovery),
                "n_replicates": int(np.sum(mask)),
                "in_range": bool(within_tol and above_loq),
            }
        )

    passing = [r["nominal"] for r in rows if r["in_range"]]

    return {
        "levels": rows,
        "range_low": min(passing) if passing else None,
        "range_high": max(passing) if passing else None,
        "tolerance_pct": tolerance * 100.0,
    }



# Four-parameter logistic (for pH-indicator assays)



def four_pl(x, bottom, top, ec50, hill):
    """Four-parameter logistic. Guard against x=0 under a fractional power."""
    x = np.asarray(x, dtype=float)
    safe = np.where(x <= 0, 1e-12, x)
    return bottom + (top - bottom) / (1.0 + (ec50 / safe) ** hill)


def fit_four_pl(concentration, signal):
    """
    Fit a 4PL curve. Appropriate for indicator-dye readouts such as the phenol
    red pH-shift assay, where the response saturates at both ends and a linear
    fit is only valid across a narrow middle band.

    Forcing a straight line through a sigmoid is a common source of
    underestimated low-end concentrations.
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(signal, dtype=float)

    positive = x[x > 0]
    p0 = [
        float(np.min(y)),
        float(np.max(y)),
        float(np.median(positive)) if positive.size else 1.0,
        1.0,
    ]

    popt, pcov = curve_fit(four_pl, x, y, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    residuals = y - four_pl(x, *popt)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    names = ["bottom", "top", "ec50", "hill"]
    return {
        "params": dict(zip(names, popt.astype(float))),
        "se": dict(zip(names, perr.astype(float))),
        "r_squared": 1.0 - ss_res / ss_tot,
        "residual_sd": float(np.sqrt(ss_res / (x.size - 4))),
        "n": int(x.size),
    }
