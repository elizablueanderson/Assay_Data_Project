"""Figure generation. Every plot shows the data, the fit, and the uncertainty."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import calibration as cal_mod
from . import kinetics as kin_mod

INK = "#1b2a33"
ACCENT = "#c1553d"
SECOND = "#3d7ea6"
MUTED = "#8a9aa3"
BAND = "#c1553d"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def plot_standard_curve(cal, limits, path, assay_name="Berthelot (indophenol)"):
    """Standard curve with confidence band, LOD and LOQ marked."""
    fig, (ax, ax_resid) = plt.subplots(
        2, 1, figsize=(7, 6.5), height_ratios=[3, 1], sharex=True
    )
    _style(ax)
    _style(ax_resid)

    grid = np.linspace(0, cal.x.max() * 1.05, 250)
    fitted = cal.predict(grid)

    se_fit = cal.residual_sd * np.sqrt(
        1.0 / cal.n + (grid - cal.x_mean) ** 2 / cal.sxx
    )
    ax.fill_between(
        grid,
        fitted - 1.96 * se_fit,
        fitted + 1.96 * se_fit,
        color=BAND,
        alpha=0.13,
        linewidth=0,
        label="95% CI of fit",
    )
    ax.plot(grid, fitted, color=ACCENT, linewidth=1.8, label="OLS fit", zorder=3)
    ax.scatter(
        cal.x, cal.y, s=34, color=INK, zorder=4, edgecolor="white",
        linewidth=0.7, label="standards",
    )

    ax.axvline(limits["lod"], color=SECOND, linestyle=":", linewidth=1.4)
    ax.axvline(limits["loq"], color=SECOND, linestyle="--", linewidth=1.4)

    top = ax.get_ylim()[1]
    ax.text(limits["lod"], top * 0.46, "LOD ", color=SECOND, fontsize=8.5,
            rotation=90, ha="right", va="center")
    ax.text(limits["loq"], top * 0.46, " LOQ", color=SECOND, fontsize=8.5,
            rotation=90, ha="left", va="center")

    eq = (
        f"A = {cal.slope:.4f}·C + {cal.intercept:.4f}\n"
        f"R² = {cal.r_squared:.4f}   n = {cal.n}\n"
        f"LOD = {limits['lod']:.3f} mM\n"
        f"LOQ = {limits['loq']:.3f} mM"
    )
    ax.text(
        0.97, 0.05, eq, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=MUTED, alpha=0.9),
    )

    ax.set_ylabel("Absorbance (625 nm)", fontsize=10, color=INK)
    ax.set_title(
        f"Ammonium standard curve — {assay_name}",
        fontsize=12, color=INK, pad=12, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    residuals = cal.y - cal.predict(cal.x)
    ax_resid.axhline(0, color=MUTED, linewidth=1)
    ax_resid.scatter(cal.x, residuals, s=26, color=ACCENT,
                     edgecolor="white", linewidth=0.6)
    ax_resid.set_xlabel("NH₄⁺ (mM)", fontsize=10, color=INK)
    ax_resid.set_ylabel("Residual", fontsize=9, color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_kinetics(substrate, rate, fit_result, path):
    """Michaelis-Menten fit with Km and Vmax annotated."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _style(ax)

    best = fit_result["best"]
    grid = np.linspace(0.01, max(substrate) * 1.08, 300)

    if best.model == "michaelis-menten":
        curve = kin_mod.michaelis_menten(grid, **best.params)
    else:
        curve = kin_mod.substrate_inhibition(grid, **best.params)

    ax.plot(grid, curve, color=ACCENT, linewidth=1.9, zorder=3,
            label=f"{best.model} fit")
    ax.scatter(substrate, rate, s=40, color=INK, zorder=4,
               edgecolor="white", linewidth=0.7, label="initial rates")

    vmax, km = best.params["vmax"], best.params["km"]
    ax.axhline(vmax, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(grid[0], vmax * 1.02, "$V_{max}$", va="bottom", ha="left",
            fontsize=9.5, color=MUTED)

    ax.plot([km, km], [0, vmax / 2], color=SECOND, linestyle="--", linewidth=1.2)
    ax.plot([0, km], [vmax / 2, vmax / 2], color=SECOND, linestyle="--", linewidth=1.2)
    ax.text(km, vmax / 2 * 0.12, f" $K_m$ = {km:.2f} mM", fontsize=9, color=SECOND)

    txt = (
        f"$V_{{max}}$ = {vmax:.3f} ± {best.se['vmax']:.3f} µmol/min\n"
        f"$K_m$ = {km:.3f} ± {best.se['km']:.3f} mM\n"
        f"R² = {best.r_squared:.4f}"
    )
    ax.text(
        0.97, 0.06, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=MUTED, alpha=0.9),
    )

    ax.set_xlabel("Urea (mM)", fontsize=10, color=INK)
    ax.set_ylabel("Initial rate (µmol NH₃ · min⁻¹)", fontsize=10, color=INK)
    ax.set_title("Urease kinetics — free-cell control", fontsize=12,
                 color=INK, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 0.32))
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_formulation_screen(labels, retention, errors, path):
    """Activity retention across encapsulation formulations."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _style(ax)

    order = np.argsort(retention)[::-1]
    labels = np.asarray(labels)[order]
    retention = np.asarray(retention)[order]
    errors = np.asarray(errors)[order]

    colors = [ACCENT if r >= 50 else MUTED for r in retention]
    positions = np.arange(len(labels))

    ax.bar(positions, retention, yerr=errors, capsize=4, color=colors,
           edgecolor="white", linewidth=0.8, width=0.66,
           error_kw=dict(ecolor=INK, linewidth=1.1))

    ax.axhline(100, color=INK, linestyle="--", linewidth=1,
               label="free-cell control")
    ax.axhline(50, color=MUTED, linestyle=":", linewidth=1,
               label="50% retention threshold")

    for pos, value in zip(positions, retention):
        ax.text(pos, value + 4, f"{value:.0f}%", ha="center",
                fontsize=8.5, color=INK)

    wrapped = [
        lab.replace(" ", "\n", 1).replace(" / ", "\n").replace(" + ", "\n+ ")
        for lab in labels
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(wrapped, fontsize=8.5, linespacing=1.4)
    ax.set_ylabel("Specific activity retained (%)", fontsize=10, color=INK)
    ax.set_title(
        "Encapsulation formulation screen — urease activity retention",
        fontsize=12, color=INK, pad=12, loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(retention.max() + 18, 118))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_inverse_prediction(cal, prediction, loq, path):
    """
    Relative uncertainty of a back-calculated concentration
    Absolute CI width is almost flat
    As a percentage of the value being reported, the picture inverts: a sample near the LOQ can
    have 20% relative error off a curve with R^2 = 0.999. This number
    decides whether a measurement is reportable
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style(ax)

    grid = np.linspace(max(loq * 0.35, 0.02), cal.x.max(), 300)
    pred_grid = cal_mod.inverse_predict(cal, cal.predict(grid), n_replicates=3)
    rel_pct = 100 * (pred_grid["ci_upper"] - pred_grid["ci_lower"]) / 2 / grid

    ax.plot(grid, rel_pct, color=ACCENT, linewidth=2.0,
            label="95% CI half-width (3 replicates)")
    ax.fill_between(grid, 0, rel_pct, color=BAND, alpha=0.10, linewidth=0)

    ax.axvline(loq, color=SECOND, linestyle="--", linewidth=1.4)
    ax.text(loq, rel_pct.max() * 0.80, "  LOQ", color=SECOND, fontsize=9)

    ax.axhline(10, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(cal.x.max(), 10.8, "10% relative error ", color=MUTED,
            fontsize=8.5, ha="right")

    conc = prediction["concentration"]
    sample_rel = 100 * (prediction["ci_upper"] - prediction["ci_lower"]) / 2 / conc
    ax.scatter(conc, sample_rel, s=48, color=INK, zorder=5,
               edgecolor="white", linewidth=0.8, label="measured samples")
    for c, r in zip(conc, sample_rel):
        ax.annotate(f"{r:.1f}%", (c, r), textcoords="offset points",
                    xytext=(7, 7), fontsize=8.5, color=INK)

    ax.set_xlabel("Back-calculated NH₄⁺ (mM)", fontsize=10, color=INK)
    ax.set_ylabel("Relative uncertainty (%)", fontsize=10, color=INK)
    ax.set_title(
        "Relative uncertainty rises sharply near the LOQ",
        fontsize=12, color=INK, pad=12, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(0, max(rel_pct.max() * 1.18, 24))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
