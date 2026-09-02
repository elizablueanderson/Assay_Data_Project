"""
End-to-end analysis: raw plate data in, characterised assay out.

    python scripts/run_analysis.py

Produces four figures in figures/, a results table in results/summary.csv,
and a printed report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import calibration as cal_mod  # noqa: E402
from src import kinetics as kin_mod  # noqa: E402
from src import plotting as plot_mod  # noqa: E402
from src import qc as qc_mod  # noqa: E402

DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    summary_rows = []

    # ---------------------------------------------------------------- 1
    rule("1. CALIBRATION — ammonium standard curve")

    standards = pd.read_csv(DATA / "standards.csv")
    cal = cal_mod.fit_linear(standards["nh4_mM"], standards["absorbance"])

    print(f"  A = {cal.slope:.4f} * C + {cal.intercept:.4f}")
    print(f"  slope SE      {cal.se_slope:.5f}")
    print(f"  intercept SE  {cal.se_intercept:.5f}")
    print(f"  residual SD   {cal.residual_sd:.5f} AU")
    print(f"  R^2           {cal.r_squared:.5f}   (n = {cal.n})")

    print("\n  Detection limits by sigma source (ICH Q2(R1)):")
    blanks = standards.loc[standards["nh4_mM"] == 0, "absorbance"]
    for source in ("residual", "intercept", "blank"):
        lim = cal_mod.detection_limits(
            cal, sigma_source=source, blank_signal=blanks
        )
        print(
            f"    {source:<10s}  LOD = {lim['lod']:.4f} mM"
            f"   LOQ = {lim['loq']:.4f} mM"
        )

    limits = cal_mod.detection_limits(cal, sigma_source="residual")
    summary_rows += [
        {"metric": "slope_AU_per_mM", "value": cal.slope},
        {"metric": "intercept_AU", "value": cal.intercept},
        {"metric": "r_squared", "value": cal.r_squared},
        {"metric": "LOD_mM", "value": limits["lod"]},
        {"metric": "LOQ_mM", "value": limits["loq"]},
    ]

    # ---------------------------------------------------------------- 2
    rule("2. WORKING RANGE — accuracy at each level")

    wr = cal_mod.working_range(cal, tolerance=0.20, loq=limits["loq"])
    print(f"  {'nominal':>9}  {'back-calc':>10}  {'recovery':>9}  {'in range'}")
    for row in wr["levels"]:
        mark = "yes" if row["in_range"] else "no"
        print(
            f"  {row['nominal']:>9.3f}  {row['back_calculated']:>10.4f}"
            f"  {row['recovery_pct']:>8.1f}%  {mark}"
        )
    print(f"\n  Validated range: {wr['range_low']:.3f} - {wr['range_high']:.3f} mM")
    print(f"  (+/-{wr['tolerance_pct']:.0f}% recovery, levels at/above LOQ)")

    summary_rows += [
        {"metric": "range_low_mM", "value": wr["range_low"]},
        {"metric": "range_high_mM", "value": wr["range_high"]},
    ]

    # ---------------------------------------------------------------- 3
    rule("3. PLATE QC")

    plate = qc_mod.plate_summary(standards, ["nh4_mM"], "absorbance")
    print(f"  replicate groups   {plate['n_groups']}")
    print(f"  median CV          {plate['median_cv_pct']:.2f}%")
    print(f"  groups CV > 20%    {plate['n_high_cv']}")
    print(f"  verdict            {plate['verdict'].upper()}")

    top = standards.loc[standards["nh4_mM"] == standards["nh4_mM"].max(), "absorbance"]
    zp = qc_mod.z_prime(top, blanks)
    print(f"\n  Z'-factor          {zp['z_prime']:.3f}  ({zp['interpretation']})")
    print(f"  signal window      {zp['signal_window']:.3f} AU")

    summary_rows += [
        {"metric": "median_replicate_CV_pct", "value": plate["median_cv_pct"]},
        {"metric": "z_prime", "value": zp["z_prime"]},
    ]

    # ---------------------------------------------------------------- 4
    rule("4. INVERSE PREDICTION — samples back-calculated with uncertainty")

    sample_signals = np.array([0.118, 0.402, 0.735, 1.081])
    pred = cal_mod.inverse_predict(cal, sample_signals, n_replicates=3)

    print(f"  {'A625':>7}  {'NH4+ (mM)':>10}  {'95% CI':>20}  {'rel. err'}")
    for i, sig in enumerate(sample_signals):
        conc = pred["concentration"][i]
        lo, hi = pred["ci_lower"][i], pred["ci_upper"][i]
        rel = 100 * (hi - lo) / 2 / conc
        note = "  < LOQ" if conc < limits["loq"] else ""
        print(
            f"  {sig:>7.3f}  {conc:>10.4f}  [{lo:>7.4f}, {hi:>7.4f}]"
            f"  {rel:>6.1f}%{note}"
        )

    # ---------------------------------------------------------------- 5
    rule("5. KINETICS — initial rates and Michaelis-Menten parameters")

    kinetics = pd.read_csv(DATA / "kinetics.csv")
    substrates, rates = [], []

    print(f"  {'urea (mM)':>10}  {'rate':>10}  {'R^2':>7}  {'pts used'}")
    for urea, group in kinetics.groupby("urea_mM"):
        averaged = group.groupby("time_min")["nh3_umol"].mean().reset_index()
        est = kin_mod.initial_rate(
            averaged["time_min"], averaged["nh3_umol"], substrate_umol=urea
        )
        substrates.append(urea)
        rates.append(est["rate_umol_per_min"])
        print(
            f"  {urea:>10.1f}  {est['rate_umol_per_min']:>10.4f}"
            f"  {est['r_squared']:>7.4f}  {est['n_points_used']}/{est['n_points_total']}"
        )

    fit = kin_mod.fit_kinetics(substrates, rates, model="auto")
    print("\n  Model comparison by AIC (lower is better):")
    for name, candidate in fit["all"].items():
        if name == fit["best"].model:
            note = "  <- selected"
        elif not kin_mod._is_identifiable(candidate, substrates):
            ki = candidate.params.get("ki")
            note = f"  rejected: Ki = {ki:.0f} mM not identifiable in range"
        else:
            note = ""
        print(f"    {name:<22s} AIC = {candidate.aic:>8.2f}{note}")

    best = fit["best"]
    print(f"\n  {best.summary_line()}")
    print(
        f"  Catalytic efficiency Vmax/Km = "
        f"{best.params['vmax'] / best.params['km']:.4f} min^-1"
    )

    summary_rows += [
        {"metric": "Vmax_umol_per_min", "value": best.params["vmax"]},
        {"metric": "Km_mM", "value": best.params["km"]},
        {"metric": "kinetic_model", "value": best.model},
    ]

    # ---------------------------------------------------------------- 6
    rule("6. FORMULATION SCREEN — activity retention after encapsulation")

    forms = pd.read_csv(DATA / "formulations.csv")
    forms["specific_activity"] = forms["rate_umol_per_min"] / forms["od600"]

    grouped = (
        forms.groupby("formulation")["specific_activity"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    free_mean = float(
        grouped.loc[grouped["formulation"] == "Free cells", "mean"].iloc[0]
    )

    grouped["retention_pct"] = 100.0 * grouped["mean"] / free_mean
    grouped["retention_se"] = (
        100.0 * grouped["std"] / np.sqrt(grouped["count"]) / free_mean
    )
    grouped["cv_pct"] = 100.0 * grouped["std"] / grouped["mean"]
    grouped = grouped.sort_values("retention_pct", ascending=False)

    print(f"  {'formulation':<26} {'U/OD600':>9} {'retention':>11} {'CV'}")
    for _, row in grouped.iterrows():
        print(
            f"  {row['formulation']:<26} {row['mean']:>9.4f}"
            f" {row['retention_pct']:>10.1f}% {row['cv_pct']:>6.1f}%"
        )

    screened = grouped[grouped["formulation"] != "Free cells"]
    winner = screened.iloc[0]
    print(
        f"\n  Best formulation: {winner['formulation']} at "
        f"{winner['retention_pct']:.1f}% retention"
    )

    # ---------------------------------------------------------------- figures
    rule("FIGURES")

    plot_mod.plot_standard_curve(cal, limits, FIGURES / "01_standard_curve.png")
    plot_mod.plot_kinetics(substrates, rates, fit, FIGURES / "02_kinetics.png")
    plot_mod.plot_formulation_screen(
        screened["formulation"].tolist(),
        screened["retention_pct"].to_numpy(),
        screened["retention_se"].to_numpy(),
        FIGURES / "03_formulation_screen.png",
    )
    plot_mod.plot_inverse_prediction(
        cal, pred, limits["loq"], FIGURES / "04_inverse_prediction.png"
    )
    for name in sorted(p.name for p in FIGURES.glob("*.png")):
        print(f"  figures/{name}")

    out = RESULTS / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    grouped.to_csv(RESULTS / "formulation_screen.csv", index=False)
    print(f"  results/summary.csv\n  results/formulation_screen.csv")


if __name__ == "__main__":
    main()
