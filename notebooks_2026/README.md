# 2026 SEI parameter tuning

Sequential tuning of the SEI transport parameters that forms the basis of the
paper's main results. Each step fixes the parameters set by the previous steps.

| Step | Data range | Error metric | D_LVDC | D_LEDC | rho_LVDC | rho_LEDC | kappa_LVDC | kappa_LEDC | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | First charge cycle only | Voltage RMSE | **swept, set 1.22e-18 m²/s** | swept | assumed 1.33 g/cm³ | assumed 1.38 g/cm³ | assumed 4.9e-5 S/m | assumed 8.1e-5 S/m | `2026_04_16_first_charge_dsei_sweep` |
| 2 | All 400 h | Voltage RMSE | fixed (1) | **swept, set 6.66e-20 m²/s** | assumed 1.33 | assumed 1.38 | assumed 4.9e-5 | assumed 8.1e-5 | `2026_04_07_clean_dsei_sweep` |
| 3 | All 400 h | Expansion RMSE | fixed (1) | fixed (2) | **swept, set 1.5 g/cm³** | **swept, set {1.94, 2.25, 1.63} g/cm³** | assumed 4.9e-5 | assumed 8.1e-5 | `2026_04_26_density_sweep` |
| 4 | RPT (HPPC) after formation cycles | Absolute resistance error | fixed (1) | fixed (2) | fixed (3) | fixed (3) | **swept, set 5.1e-5 S/m** | **swept, set {7.6, 8.9, 12.9}e-5 S/m** | `2026_05_04_conductivity_sweep` |

Per-protocol values are listed as {base, fast, fast+}.

Note: the cached step 2 sweep was run with rho_LEDC = 1.29, rho_LVDC = 1.49
g/cm³ and kappa_LEDC = 9.0e-5, kappa_LVDC = 5.1e-5 S/m (stored in the pickle),
not the assumed values in the table.

`2026_05_06_model_vs_experiment` runs the model with the final parameter set
and compares it against the voltage, expansion, capacity and resistance data.

## Cached sweep results

Re-running a sweep takes many hours, so the result each step plots from is kept
in `data/`:

| Step | File |
|---|---|
| 1 | `data/simulation_results_ec_tuning_diff_20260421.pkl` |
| 2 | `data/simulation_results_clean_dsei_sweep_20260422.pkl` |
| 3 | `data/simulation_results_density_sweep_20260426.pkl` |
| 4 | `data/resistance_optimization_results_20260504.pkl` |

The experimental data is not committed; see `data/README.md` at the repo root.
Run everything from the repo root.
