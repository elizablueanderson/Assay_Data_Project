"""
Assay quality control.

These are the numbers that decide whether a plate is reportable. Running them
before the biology means a bad plate gets caught as a bad plate, rather than
being interpreted as a surprising result.
"""

import numpy as np
import pandas as pd


def replicate_cv(df, group_cols, value_col):
    """
    Coefficient of variation within each replicate group.

    Rough expectations for a plate-based colorimetric assay: under 10% CV is
    good, 10-20% is usable for a screen, and above 20% usually points at
    pipetting or an incompletely mixed plate rather than at biology.
    """
    grouped = df.groupby(group_cols)[value_col]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["cv_pct"] = 100.0 * out["std"] / out["mean"]
    out["flag"] = np.where(
        out["cv_pct"] > 20, "high", np.where(out["cv_pct"] > 10, "watch", "ok")
    )
    return out


def z_prime(positive_controls, negative_controls):
    """
    Z'-factor (Zhang, Chung & Oldenburg 1999):

        Z' = 1 - (3*sd_pos + 3*sd_neg) / |mean_pos - mean_neg|

    Interpretation:
        Z' > 0.5        excellent separation
        0 < Z' <= 0.5   usable but the bands are close
        Z' <= 0         controls overlap; the plate cannot discriminate

    Worth computing even outside formal screening, because it summarises
    dynamic range and noise in a single number and makes plate-to-plate
    comparison honest.
    """
    pos = np.asarray(positive_controls, dtype=float)
    neg = np.asarray(negative_controls, dtype=float)

    if pos.size < 2 or neg.size < 2:
        raise ValueError("need at least 2 replicates of each control")

    separation = abs(float(np.mean(pos)) - float(np.mean(neg)))
    if separation == 0:
        return {"z_prime": -np.inf, "interpretation": "controls are identical"}

    z = 1.0 - (3 * np.std(pos, ddof=1) + 3 * np.std(neg, ddof=1)) / separation

    if z > 0.5:
        interpretation = "excellent"
    elif z > 0:
        interpretation = "marginal"
    else:
        interpretation = "unusable"

    return {
        "z_prime": float(z),
        "signal_window": separation,
        "sd_positive": float(np.std(pos, ddof=1)),
        "sd_negative": float(np.std(neg, ddof=1)),
        "interpretation": interpretation,
    }


def spike_recovery(measured, nominal_spike, background=0.0, tolerance=0.20):
    """
    Percent recovery of a known spike into real sample matrix.

    This is the check for matrix effects. A curve built in clean buffer can be
    perfectly linear and still misread a sample, because the capsule material,
    residual media, or turbidity shifts the readout. Buffer standards cannot
    detect that; a spike into matrix can.
    """
    measured = np.atleast_1d(np.asarray(measured, dtype=float))
    nominal = np.atleast_1d(np.asarray(nominal_spike, dtype=float))
    background = np.asarray(background, dtype=float)

    recovery = 100.0 * (measured - background) / nominal
    passes = np.abs(recovery - 100.0) <= tolerance * 100.0

    return {
        "recovery_pct": recovery,
        "passes": passes,
        "tolerance_pct": tolerance * 100.0,
        "all_pass": bool(np.all(passes)),
    }


def flag_outliers(values, z_threshold=3.0):
    """
    Flag replicates by modified Z-score (median absolute deviation).

    MAD is used instead of mean/SD because with three or four replicates a
    single bad well drags the mean and inflates the SD enough to hide itself.
    The median does not move.

    Flagging is not deleting. Excluded wells belong in the record with a
    stated reason.
    """
    v = np.asarray(values, dtype=float)
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))

    if mad == 0:
        return {"modified_z": np.zeros_like(v), "is_outlier": np.zeros(v.size, bool)}

    modified_z = 0.6745 * (v - median) / mad
    return {
        "median": median,
        "mad": mad,
        "modified_z": modified_z,
        "is_outlier": np.abs(modified_z) > z_threshold,
    }


def plate_summary(df, group_cols, value_col):
    """One-line-per-group QC table, suitable for pasting into a lab notebook."""
    cv = replicate_cv(df, group_cols, value_col)
    n_high = int((cv["flag"] == "high").sum())
    n_watch = int((cv["flag"] == "watch").sum())

    return {
        "table": cv,
        "n_groups": len(cv),
        "n_high_cv": n_high,
        "n_watch_cv": n_watch,
        "median_cv_pct": float(cv["cv_pct"].median()),
        "verdict": "review" if n_high else "pass",
    }
