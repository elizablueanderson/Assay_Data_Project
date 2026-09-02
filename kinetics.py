"""
Urease kinetics: initial rates, Michaelis-Menten parameters, specific activity.

Parameters are estimated by non-linear least squares on
rate data. The double-reciprocal transform is 
fine as a diagnostic plot and poor as an estimator.

Activity is normalized to biomass before free and encapsulated cells
are compared. Encapsulated preparations usually don't contain the same cell count as
the free cell control, so raw rates confuse "this formulation preserves
activity" with "this formulation happens to hold more cells".
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Urea hydrolysis: (NH2)2CO + H2O -> 2 NH3 + CO2
NH3_PER_UREA = 2.0

# One unit of urease liberates 1.0 umol of NH3 per minute (pH 7.0, 25 C).
UMOL_NH3_PER_UNIT_PER_MIN = 1.0



# Initial rates



def initial_rate(time_min, concentration_umol, max_fraction=0.10, substrate_umol=None):
    """
    Estimate an initial rate by linear regression over the early timepoints.

    Returns the rate in umol NH3 per minute, its standard error, and the R^2
    of the window actually fitted. An R^2 far below ~0.98 means the window
    is too wide.
    """
    t = np.asarray(time_min, dtype=float)
    c = np.asarray(concentration_umol, dtype=float)

    order = np.argsort(t)
    t, c = t[order], c[order]

    if substrate_umol is not None:
        produced = c - c[0]
        max_product = max_fraction * substrate_umol * NH3_PER_UREA
        mask = produced <= max_product
        if mask.sum() < 3:
            mask = np.zeros_like(t, dtype=bool)
            mask[: max(3, t.size // 2)] = True
    else:
        mask = np.zeros_like(t, dtype=bool)
        mask[: max(3, t.size // 2)] = True

    fit = stats.linregress(t[mask], c[mask])

    return {
        "rate_umol_per_min": float(fit.slope),
        "se": float(fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "n_points_used": int(mask.sum()),
        "n_points_total": int(t.size),
    }


# Models



def michaelis_menten(s, vmax, km):
    """v = Vmax * [S] / (Km + [S])"""
    s = np.asarray(s, dtype=float)
    return vmax * s / (km + s)


def substrate_inhibition(s, vmax, km, ki):
    """
    v = Vmax * [S] / (Km + [S] * (1 + [S]/Ki))

    Urease shows measurable substrate inhibition at high urea in several
    systems. If rates level off at the top of the substrate series, a plain
    Michaelis-Menten fit will absorb the roll-off by understating Vmax.
    """
    s = np.asarray(s, dtype=float)
    return vmax * s / (km + s * (1.0 + s / ki))


@dataclass
class KineticFit:
    model: str
    params: dict
    se: dict
    r_squared: float
    aic: float
    n: int

    def summary_line(self) -> str:
        parts = [f"{k} = {v:.4g} +/- {self.se[k]:.2g}" for k, v in self.params.items()]
        return f"{self.model}: " + ", ".join(parts) + f" (R^2 = {self.r_squared:.4f})"


def _aic(residuals, n_params):
    """Akaike information criterion under Gaussian errors."""
    n = residuals.size
    ss_res = float(np.sum(residuals**2))
    if ss_res <= 0:
        return -np.inf
    return n * np.log(ss_res / n) + 2 * n_params


def _is_identifiable(fit: KineticFit, substrate, max_rel_se=0.5, range_multiple=3.0):
    """
    Decide whether the substrate-inhibition Ki is actually constrained by the
    data, rather than being a free parameter that soaked up noise.

    Two failure modes:

      1. Ki lands far above the highest substrate tested = the data can't distinguish it
         from no inhibition at all. 
      2. Ki carries a large relative standard error =  fit is indifferent to its value.

    """
    if "ki" not in fit.params:
        return True

    ki = fit.params["ki"]
    ki_se = fit.se.get("ki", np.inf)
    s_max = float(np.max(substrate))

    within_range = ki <= range_multiple * s_max
    well_determined = np.isfinite(ki_se) and (ki_se / ki) <= max_rel_se

    return bool(within_range and well_determined)


def _select_model(fits, substrate):
    """
    Lowest AIC among the models whose parameters are identifiable.

    Falls back to Michaelis-Menten if nothing passes, since it is the model
    that makes the fewest claims.
    """
    eligible = {
        name: fit for name, fit in fits.items() if _is_identifiable(fit, substrate)
    }
    if not eligible:
        return fits["michaelis-menten"]
    return min(eligible.values(), key=lambda f: f.aic)


def fit_kinetics(substrate_mm, rate, model="auto"):
    """
    Fit Michaelis-Menten and, optionally, substrate inhibition.

    model="auto" fits both and returns whichever has the lower AIC, so the
    extra parameter has to be necessary rather than being justified by the
    improvement in R^2.
    """
    s = np.asarray(substrate_mm, dtype=float)
    v = np.asarray(rate, dtype=float)

    if s.size < 4:
        raise ValueError("need at least 4 substrate levels to fit kinetics")

    fits = {}

    if model in ("auto", "mm"):
        p0 = [float(np.max(v)) * 1.2, float(np.median(s))]
        popt, pcov = curve_fit(
            michaelis_menten, s, v, p0=p0, bounds=(0, np.inf), maxfev=20000
        )
        resid = v - michaelis_menten(s, *popt)
        fits["michaelis-menten"] = KineticFit(
            model="michaelis-menten",
            params={"vmax": float(popt[0]), "km": float(popt[1])},
            se=dict(zip(["vmax", "km"], np.sqrt(np.diag(pcov)).astype(float))),
            r_squared=1.0 - np.sum(resid**2) / np.sum((v - v.mean()) ** 2),
            aic=_aic(resid, 2),
            n=int(s.size),
        )

    if model in ("auto", "substrate-inhibition") and s.size >= 5:
        p0 = [float(np.max(v)) * 1.5, float(np.median(s)), float(np.max(s)) * 3]
        try:
            popt, pcov = curve_fit(
                substrate_inhibition, s, v, p0=p0, bounds=(0, np.inf), maxfev=20000
            )
            resid = v - substrate_inhibition(s, *popt)
            fits["substrate-inhibition"] = KineticFit(
                model="substrate-inhibition",
                params={
                    "vmax": float(popt[0]),
                    "km": float(popt[1]),
                    "ki": float(popt[2]),
                },
                se=dict(
                    zip(["vmax", "km", "ki"], np.sqrt(np.diag(pcov)).astype(float))
                ),
                r_squared=1.0 - np.sum(resid**2) / np.sum((v - v.mean()) ** 2),
                aic=_aic(resid, 3),
                n=int(s.size),
            )
        except RuntimeError:
            pass  # did not converge; Michaelis-Menten stands

    if model == "auto":
        best = _select_model(fits, s)
        return {"best": best, "all": fits}

    key = "michaelis-menten" if model == "mm" else model
    return {"best": fits[key], "all": fits}


# Normalization



def specific_activity(rate_umol_per_min, biomass, biomass_unit="OD600"):
    """
    Convert a raw rate into units per unit biomass.

    1 U = 1 umol NH3 released per minute. Normalising by OD600 (or by mg
    protein, or by dry cell weight) is what makes a free cell control and
    encapsulated strain comparable.
    """
    rate = np.asarray(rate_umol_per_min, dtype=float)
    mass = np.asarray(biomass, dtype=float)

    if np.any(mass <= 0):
        raise ValueError("biomass must be positive")

    units = rate / UMOL_NH3_PER_UNIT_PER_MIN
    return {
        "units": units,
        "specific_activity": units / mass,
        "unit_label": f"U per {biomass_unit}",
    }


def activity_retention(encapsulated_specific, free_specific):
    """
    Retention (%) of specific activity after encapsulation.
     Below 100% means the encapsulation process cost activity during printing,
    crosslinker chemistry, or diffusional limitation across the capsule wall.
    Above 100% is not impossible but is a
    sign that the biomass normalization is wrong.
    """
    enc = np.asarray(encapsulated_specific, dtype=float)
    free = np.asarray(free_specific, dtype=float)
    return 100.0 * enc / free
