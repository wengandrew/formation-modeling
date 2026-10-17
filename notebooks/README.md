# Exploratory notebooks

Analysis notebooks from 2022 onward, named `YYYY_MM_DD_description.ipynb`.
The analysis behind the paper's SEI parameter tuning moved to scripts in
`notebooks_2026/`; these notebooks are kept as a record.

Figures and pickled results written by the notebooks go to `outputs/`. SVG
figures are tracked in git; PNGs, pickles and CSVs there are local only.

Run notebooks with the repo root as the working directory (each notebook's
first cell changes to it).

## Compatibility with the current `src/` (static check, October 2026)

Every notebook was checked against the current code for imports, `module.function`
references and `run_sim` arguments. This does not execute them: a notebook can
still fail on missing local data or outputs.

| Notebook | Status |
|---|---|
| `archives/2025_04_16_dsei_tuning_temp_debug.ipynb` | Uses an old `run_sim` signature (`gamma_kappa`, `to_simulate_*` flags) and the removed `simutils.plot_diffusivity_heatmaps`; would need porting. |
| `2025_05_20_cycle_life_tuning.ipynb` | Defines its own `run_sim` (an older copy) instead of using `simutils.run_sim`. |
| All others | No references to removed or renamed code. |

`2023_09_21_single_sei_sims.ipynb` imported `plotter` without the `src.`
prefix; fixed.
