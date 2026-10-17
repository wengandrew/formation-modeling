# 2026 SEI parameter tuning

Sequential tuning of the SEI transport parameters that forms the basis of the
paper's main results. Each step fixes the parameters set by the previous steps.

| Step | Data range | Error metric | D_LVDC | D_LEDC | rho_LVDC | rho_LEDC | kappa_LVDC | kappa_LEDC | Script (sweep) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | First charge cycle only | Voltage RMSE | **swept, set 1.22e-18 m²/s** | swept | assumed 1.33 g/cm³ | assumed 1.38 g/cm³ | assumed 4.9e-5 S/m | assumed 8.1e-5 S/m | `dsei_sweep.py` (`dsei_fc`) |
| 2 | Through formation aging (~500 h) | Voltage RMSE | fixed (1) | **swept, set 6.66e-20 m²/s** | assumed 1.33* | assumed 1.38* | assumed 4.9e-5* | assumed 8.1e-5* | `dsei_sweep.py` (`dsei_all`) |
| 3 | Through formation aging (~500 h) | Expansion RMSE | fixed (1) | fixed (2) | **swept, set 1.5 g/cm³** | **swept, set {1.94, 2.25, 1.63} g/cm³** | assumed 4.9e-5 | assumed 8.1e-5 | `density_sweep.py` (`rho_all`) |
| 4 | RPT (HPPC) after formation cycles | Absolute resistance error | fixed (1) | fixed (2) | fixed (3) | fixed (3) | **swept, set 5.1e-5 S/m** | **swept, set {7.6, 8.9, 12.9}e-5 S/m** | `conductivity_sweep.py` (`kappa_hppc1`) |

Per-protocol values are listed as {base, fast, fast+}.

\* Step 2 is defined with the same assumed rho/kappa as step 1, but the cached
step 2 sweep was run with rho_LEDC = 1.29, rho_LVDC = 1.49 g/cm³ and
kappa_LEDC = 9.0e-5, kappa_LVDC = 5.1e-5 S/m (recorded as `cache_overrides` in
`src/tuning.py`). Re-running the D_LVDC = 1.22e-18 column with the assumed
values (`data/dsei_all_column_assumed_20261014.pkl`) leaves the grid minimum at
D_LEDC = 6.66e-20 m²/s for all three protocols and changes the voltage RMSE
near it by at most ~3 mV. Interpolating between grid points, the minimum moves
about 5% lower (6.65e-20 → 6.34e-20, protocol mean), well within the ~31% grid
spacing, so the set value is kept and the later steps were not re-run.

All steps use the same experimental cells: 152064 (base), 152071 (fast) and
152098 (fast+). The original step 1 sweep used the second baseline cell 152074
for the base protocol; its base runs were redone with 152064
(`data/dsei_fc_20261015.pkl`, fast and fast+ unchanged). D_LVDC = 1.22e-18 m²/s
holds with either cell: averaged over the three protocols, the first-charge
voltage RMSE at 1.22e-18 is within 0.4 mV (152074) and 0.1 mV (152064) of the
best D_LVDC, which moves from 4.0e-18 to 6.0e-19. The first-charge voltage
constrains D_LVDC only weakly: D_LVDC values within 2 mV of the best fit span
about two decades. With cell 152064 the base protocol alone is nearly flat in
D_LVDC (45-48 mV from 1e-19 to 1e-17), so D_LVDC is set mainly by the fast and
fast+ data. `figures/dlvdc_profile.svg` shows this directly: the best voltage
RMSE over D_LEDC as a function of D_LVDC has a shallow minimum over the first
charge, and is flat for D_LVDC above ~5e-21 through formation aging, which is
why D_LVDC is set from the first charge.

`density_sweep.py` also plots `rho_fc`, the step 3 sweep repeated over the
first charge only. `model_vs_experiment.py` runs the fully tuned model and
compares it against the voltage, expansion, capacity and resistance data.

Everything the scripts produce stays in this folder: figures in `figures/`,
sweep results in `data/`.

## Fitting settings

| Sweep | Simulated through | Data compared | Voltage RMSE range | Strain |
|---|---|---|---|---|
| `dsei_fc`, `rho_fc` | First charge (`include_flag=0`, no CV hold) | First charge, after the first 0.5 h | 2.0-4.2 V | Kink smoothed |
| `dsei_all` | Formation aging (`include_flag=3`, ~503 h) | 0.5-497.5 h (end of the data) | 3.2-4.2 V | Kink smoothed* |
| `rho_all` | Formation aging (`include_flag=3`, ~503 h) | 0.5-497.5 h (end of the data) | 3.2-4.2 V | Kink smoothed* |
| `kappa_hppc1` | First RPT (`include_flag=2`) | R_sei + 10 mOhm at the first HPPC pulse | - | - |

All sweeps use cells 152064 / 152071 / 152098 and smooth each cell's early
expansion kink over its own window (152064: 0.5-0.75 h, 152071: 0.5-0.6 h,
152098: 0.5-0.55 h). \* Two cached sweeps predate this: `dsei_all` smoothed
152064 over 0.5-0.85 h, and `rho_all` used unsmoothed strain (recorded as
`cache_overrides` in `src/tuning.py`). Neither affects a selected value:
step 2 selects on voltage, and re-running the step 3 selection column with
smoothing (`data/rho_all_column_smoothed_20261017.pkl`) changes the expansion
RMSE by at most 0.0025 µm (rho_LEDC unchanged at 1.94 / 2.25 / 1.63).

"Through formation aging (~500 h)" covers 0.5-497.5 h of data (earlier drafts
called this window "All 400 h"). Voltage RMSE is reported in mV and
expansion RMSE in µm.

## Running

```bash
python notebooks_2026/dsei_sweep.py            # steps 1-2
python notebooks_2026/density_sweep.py         # step 3
python notebooks_2026/conductivity_sweep.py    # step 4
python notebooks_2026/model_vs_experiment.py   # final comparison (a few minutes)
python notebooks_2026/tuning_summary.py        # composite figures (steps 2-4; D/rho voltage+expansion, first charge and ~500 h)
```

The sweep scripts load the cached results below, save voltage and expansion
RMSE heatmaps (first charge and through formation aging) or resistance-error heatmaps to
`figures/<sweep>_<metric>.svg`, and print the selected parameter values
next to the tuned ones. Add `--run <sweep>` to re-run a sweep (hours; uses all
cores, set with `--n-jobs`) and `--show` to display the figures.

The sweep definitions, tuned values, metrics and measured HPPC resistances live
in `src/tuning.py`; heatmaps in `src/tuning_plots.py`. `tests/test_tuning.py`
checks that this code reproduces the cached results (`pytest tests/
--runslow` also re-simulates a grid point of every sweep).

## Cached sweep results

| Sweep | Step | File |
|---|---|---|
| `dsei_fc` | 1 | `data/dsei_fc_20261015.pkl` (original, base cell 152074: `data/simulation_results_ec_tuning_diff_20260421.pkl`) |
| `dsei_all` | 2 | `data/simulation_results_clean_dsei_sweep_20260422.pkl` |
| `rho_fc` | 3 (first charge) | `data/simulation_results_first_charge_density_20260429.pkl` |
| `rho_all` | 3 | `data/simulation_results_density_sweep_20260426.pkl` |
| `kappa_hppc1` | 4 | `data/resistance_optimization_results_20260504.pkl` |
| `dsei_all`, one column | 2 (check) | `data/dsei_all_column_assumed_20261014.pkl` |
| `rho_all`, one column | 3 (check) | `data/rho_all_column_smoothed_20261017.pkl` |

These were produced by the original notebooks, which are in the git history
(commits ffdc89b and 3f1fd69). Re-runs are saved as `data/<sweep>_<date>.pkl`; point the
sweep's `cache` in `src/tuning.py` at the new file to use it.

Measured HPPC resistances and RPT capacities (with their sources) are in
`data/measurements.yaml`. The HPPC values are the median series resistance over
each HPPC sequence from `notebooks/2025_04_16_hppc.ipynb`, compared at the
sequence midpoint; their error bars come from an external source.

The experimental data is not committed; see `data/README.md` at the repo root.
