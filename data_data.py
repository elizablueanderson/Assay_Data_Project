"""
Ground truth used to simulate (for validating the fitting code):
    Km   = 3.20 mM
    Vmax = 0.850 umol NH3 / min
    Berthelot slope = 0.612 AU / mM, intercept = 0.041 AU
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260901)

TRUE_KM = 3.20
TRUE_VMAX = 0.850
TRUE_SLOPE = 0.612
TRUE_INTERCEPT = 0.041


def make_standards():
    """Ammonium standards, Berthelot readout at 625 nm, triplicate."""
    levels = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00])
    rows = []
    for level in levels:
        for rep in range(1, 4):
            signal = TRUE_SLOPE * level + TRUE_INTERCEPT
            # noise scales slightly with signal, as it does on a real plate
            noise = RNG.normal(0, 0.008 + 0.010 * signal)
            rows.append(
                {
                    "assay": "berthelot",
                    "nh4_mM": level,
                    "replicate": rep,
                    "absorbance": round(float(signal + noise), 4),
                }
            )
    return pd.DataFrame(rows)


def make_kinetics():
    """Progress curves at seven urea concentrations, duplicate."""
    substrates = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    times = np.array([0, 2, 4, 6, 8, 10, 15, 20])
    rows = []
    for s in substrates:
        v = TRUE_VMAX * s / (TRUE_KM + s)
        for rep in range(1, 3):
            for t in times:
                # mild first-order curvature from substrate depletion
                produced = v * t * (1 - 0.004 * t)
                noise = RNG.normal(0, 0.015 + 0.02 * produced)
                rows.append(
                    {
                        "urea_mM": s,
                        "replicate": rep,
                        "time_min": int(t),
                        "nh3_umol": round(max(0.0, float(produced + noise)), 4),
                    }
                )
    return pd.DataFrame(rows)


def make_formulations():
    """
    Six encapsulation formulations plus a free-cell control
    Retention is pulled down by crosslinker concentration. 
    Biomass loading deliberately varies between formulations so that the
    normalization step in the analysis actually matters.
    """
    specs = [
        ("Free cells", 1.00, 0.95),
        ("F1 2% alg / 100 mM Ca", 0.82, 0.71),
        ("F2 2% alg / 200 mM Ca", 0.71, 0.78),
        ("F3 3% alg / 100 mM Ca", 0.64, 0.66),
        ("F4 3% alg / 200 mM Ca", 0.45, 0.80),
        ("F5 2% alg + 1% gel", 0.38, 0.52),
        ("F6 3% alg + 1% gel", 0.22, 0.61),
    ]
    free_specific_activity = 0.640  # U per OD600

    rows = []
    for label, retention, od in specs:
        for rep in range(1, 4):
            od_rep = od * (1 + RNG.normal(0, 0.05))
            specific = free_specific_activity * retention * (1 + RNG.normal(0, 0.07))
            rate = specific * od_rep
            rows.append(
                {
                    "formulation": label,
                    "replicate": rep,
                    "od600": round(float(od_rep), 4),
                    "rate_umol_per_min": round(float(rate), 4),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    make_standards().to_csv("data/standards.csv", index=False)
    make_kinetics().to_csv("data/kinetics.csv", index=False)
    make_formulations().to_csv("data/formulations.csv", index=False)
    print("wrote data/standards.csv, data/kinetics.csv, data/formulations.csv")
