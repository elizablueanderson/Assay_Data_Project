# Urease Assay Pipeline

Characterisation and validation of a urease activity assay, built to
compare free and microencapsulated *E. coli* for urea degradation.

**Problem:** Comparing encapsulation formulations requires an assay that can be defended.
Raw absorbance readings don't decide whether a result is reportable: 
what is the lowest concentration this method can quantify, and is a difference 
between two formulations larger than the noise.

**Approach:** A five-stage pipeline: calibration, detection limits, plate QC,
inverse prediction, enzyme kinetics, followed by a formulation screen normalized
to biomass.

**Result:** A fully characterised assay: LOQ of 0.188 mM NH₄⁺, validated working
range 0.25–2.00 mM, Z′ = 0.95, and Michaelis–Menten parameters of
Km = 3.24 ± 0.13 mM and Vmax = 0.837 ± 0.009 µmol NH₃·min⁻¹. Across six
encapsulation formulations, retained specific activity spans 21% to 84% of the
free cell control.




## Results

### Calibration and detection limits

![Standard curve](figures/01_standard_curve.png)

LOD and LOQ follow ICH Q2(R1). The pipeline reports all three permitted estimates
of σ, because they disagree on the same data:

| σ source | LOD (mM) | LOQ (mM) |
|---|---|---|
| Residual SD of regression | 0.062 | 0.188 |
| SE of intercept | 0.018 | 0.053 |
| SD of blank replicates | 0.019 | 0.057 |

An LOD quoted must name its σ source to be reproducible. This pipeline
uses the residual SD, the most conservative of the three, and states the
choice in every output.

### Enzyme kinetics

![Kinetics](figures/02_kinetics.png)

Parameters are estimated by non-linear least squares on untransformed rates. 
The double-reciprocal transform inflates the error on the low-substrate points, '
which are the points carrying most of the information about Km.

### Formulation screen

![Formulation screen](figures/03_formulation_screen.png)

Activity is normalised to OD₆₀₀ before comparison. Encapsulated preparations do not
contain the same cell count as the free-cell control, so raw rates would confuse
"this formulation preserves activity" with "this formulation happens to hold more cells".



## Four decisions that shape the output


1. **Working range is defined by recovery, not by R².** A curve can post R² = 0.999
   and still recover its bottom standard at 60% of nominal, because it is dominated
   by high signal points. Recovery is judged per level.

2. **Inverse prediction carries a confidence interval.** The classic Draper & Smith
   standard error includes a term in `(x̂ − x̄)² / Sxx`, so uncertainty is lowered at
   the center of the standards and widens toward both ends.

3. **A model with an unconstrained parameter is rejected, even when AIC prefers it.**
   Substrate inhibition scores a lower AIC than Michaelis–Menten,
   but returns Ki = 584 mM against a highest tested substrate of 40 mM. This inhibition
   is not distinguishable from no inhibition.
   The identifiability check catches this, rather than AIC alone.

5. **Outliers are flagged by median absolute deviation, not mean ± SD.** With three
   replicates, one bad well inflates the SD enough to hide itself. The median stays the same.



## Layout

```
urease-assay-pipeline/
├── src/
│   ├── calibration.py   standard curves, LOD/LOQ, inverse prediction, working range
│   ├── kinetics.py      initial rates, Michaelis-Menten, identifiability, normalisation
│   ├── qc.py            replicate CV, Z'-factor, spike recovery, outlier flagging
│   └── plotting.py      figure generation
├── scripts/
│   ├── run_analysis.py       end-to-end analysis
│   └── make_example_data.py  regenerate the example datasets
├── tests/test_pipeline.py    20 tests
├── data/                     input CSVs
├── figures/                  generated figures
└── results/                  generated summary tables
```

## Running it

```bash
pip install -r requirements.txt
python scripts/run_analysis.py
pytest tests/ -q
```

**`standards.csv`** — one row per well, replicates as separate rows

| assay | nh4_mM | replicate | absorbance |
|---|---|---|---|

Pass every replicate rather than pre-averaging because the scatter defines the
residual SD, and the residual SD sets the detection limits.

**`kinetics.csv`** — progress curves, one row per timepoint

| urea_mM | replicate | time_min | nh3_umol |
|---|---|---|---|

**`formulations.csv`** — one row per bead preparation

| formulation | replicate | od600 | rate_umol_per_min |
|---|---|---|---|

Include a row labelled `Free cells` as the normalisation reference.

## Tests

Fitting routines are validated by simulating from known parameters and confirming
recovery: a Km within 0.02 mM of the value was returned and used to generate the data. 

```
20 passed in 1.32s
```

## References

- Duque, R., Shan, Y., Joya, M., Ravichandran, N., Asi, B., Mobed-Miremadi, M.,
Mulrooney, S., McNeil, M., & Prakash, S. (2018). Effect of artificial cell
miniaturization on urea degradation by immobilized *E. coli* DH5α (pKAU17).
*Artificial Cells, Nanomedicine, and Biotechnology*, 46(sup2), 766–775.
https://doi.org/10.1080/21691401.2018.1469026

## Context

Written alongside ongoing research on microencapsulation of *E. coli* DH5α for urea
degradation.
