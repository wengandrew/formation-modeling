# Formation Modeling

Electrochemical simulation code for lithium-ion battery formation modeling.
Published in the Journal of the Electrochemical Society (December 2023).

## Project Structure

```
src/            Core Python source
  cellsim.py    Cell and Simulation classes (main simulation engine)
  simutils.py   run_sim() (protocols in FORMATION_PROTOCOLS), RMSE, legacy heatmap wrappers
  heatmaps.py   Shared base/fast/fast+ heatmap panels
  modelutils.py Electrochemistry lookup functions (OCP, expansion, resistance)
  dvdq.py       eSOH -> open-circuit voltage helpers (tested; not used by current notebooks)
  ioutils.py    Data loading utilities (cell IDs → CSV paths)
  plotter.py    Matplotlib style initialization
  paths.py      Repo-root paths (data, params, outputs) independent of the working directory
  tuning.py     SEI parameter sweeps: specs, tuned values, metrics, cached results
  tuning_plots.py  Sweep heatmaps
  tuning_cli.py Shared entry point for the notebooks_2026 sweep scripts
params/         YAML configuration files for cell parameters
  default.yaml  Primary parameter set
notebooks/      Jupyter analysis notebooks (YYYY_MM_DD naming)
  outputs/      Figures and pickled results written by the notebooks
notebooks_2026/ Paper parameter-tuning scripts (see its README)
  data/         Cached sweep results; measurements.yaml (measured values + provenance)
  figures/      Figures written by the scripts
tests/          pytest test suite (regression/ holds exact run_sim and figure fingerprints)
.github/        CI: fast tests on every push (Python 3.11)
data/           Experimental data (not committed; download from Google Drive)
```

## Running Simulations

Always run from the repo root (not from inside `src/`):

```python
from src import cellsim, simutils

# Single simulation (formation_type: 'base' | 'fast' | 'fast+')
df = simutils.run_sim('fast', include_flag=2)  # through the first RPT
df = simutils.run_sim('base')   # NOTE: default include_flag=False == 0 stops after the first charge

# Fully tuned model (values from src/tuning.py)
from src import tuning
df = simutils.run_sim('fast', **tuning.tuned_params('fast'), include_flag=3)

# Manual simulation
cell = cellsim.Cell()
cell.load_config('params/default.yaml')
sim = cellsim.Simulation(cell, sim_time_s=1000*3600, dt=1.0)
sim.run_rest(1, rest_time_hrs=0.5)
sim.run_chg_cccv(1, icc=2.5/10, icv=2.5/20, vmax=4.2)
df = sim.get_results()  # returns a pandas DataFrame
```

`include_flag`: 0 = first charge only (ends without a CV hold), 1 = formation
cycling, 2 = + first RPT (HPPC), 3 = + formation aging (~500 h), 4 = + second RPT,
5 = + 101 aging cycles. The protocols are step lists in `simutils.FORMATION_PROTOCOLS`.

## Core Design

```
Cell            Holds cell parameters (loaded from YAML) + OCP/expansion functions
Simulation      Time-stepping engine; holds all state arrays pre-allocated as NaN vectors
  .step()       Single time step (CC or CV mode); updates all electrochemical states
  .run_rest()   Rest protocol wrapper around step()
  .run_chg_cccv() / .run_dch_cccv()  CCCV charge/discharge wrappers
  .get_results() Returns results as a trimmed DataFrame
```

## Key State Variables (in `Simulation`)

| Variable | Description |
|---|---|
| `theta_n / theta_p` | Electrode stoichiometries |
| `ocv_n / ocv_p / ocv / vt` | Half-cell OCPs, full-cell OCV, terminal voltage |
| `i_app / i_int / i_sei` | Applied, intercalation, SEI currents |
| `j_sei1 / j_sei2` | SEI current densities (EC and VC reactions) |
| `delta_sei1 / delta_sei2` | SEI layer thicknesses |
| `R_sei1 / R_sei2 / R_sei` | SEI resistances |
| `c_sei1 / c_sei2` | Bulk solvent concentrations |
| `expansion_rev / expansion_irrev` | Reversible and irreversible cell expansion |
| `boost` | SEI boost factor (strain-driven extra SEI growth during cycling) |

## Step Numbers

| Constant | Value | Meaning |
|---|---|---|
| `STEP_NUM_CHARGE_CC` | 0 | CC charge |
| `STEP_NUM_CHARGE_CV` | 1 | CV hold during charge |
| `STEP_NUM_DISCHARGE_CC` | 2 | CC discharge |
| `STEP_NUM_DISCHARGE_CV` | 3 | CV hold during discharge |
| `STEP_NUM_REST` | 4 | Rest |

## Parameters (YAML)

Cell parameters are loaded from YAML via `cell.load_config(path)`. Key SEI parameters
(reaction/solvent 1 = EC, forming LEDC; 2 = VC, forming LVDC):
- `D_SEIij` — diffusivity of solvent i through SEI product j. The effective
  diffusivity of each solvent is the harmonic mean over products, weighted by the
  SEI composition. `run_sim(diff_ec=D_LEDC, diff_vc=D_LVDC)` sets
  `D_SEI11 = D_SEI21 = D_LEDC` and `D_SEI12 = D_SEI22 = D_LVDC`, i.e. each is a
  property of the product layer, shared by both solvents.
- `k_SEI1 / k_SEI2` — kinetic rate constants
- `U_SEI1 / U_SEI2` — equilibrium reaction potentials (set to -1000 to disable a reaction)
- `kappa_SEI1 / kappa_SEI2` — SEI conductivities
- `rho_SEI1 / rho_SEI2` — SEI densities

## Development Guidelines

- Do **not** commit raw data files (use Google Drive; see `data/README.md`).
- Track SVG figures; PNGs, pickles and CSVs are not committed, except the cached
  sweeps in `notebooks_2026/data/` (local caches go in `notebooks_2026/data/cache/`).
- Measured values compared against the model (HPPC resistance, RPT capacity) live in
  `notebooks_2026/data/measurements.yaml`, loaded by `src/tuning.py`.
- One figure per notebook; name notebooks `YYYY_MM_DD_description.ipynb`.
- Prototype in notebooks, then refactor proven code into `src/`.
- Python 3.12 (pyenv env `formation-modeling-env`, see `.python-version`); notebooks use 3.10+ syntax such as `match`.
- Run tests with `pytest tests/`. Add `--runslow` to re-simulate cached sweep points
  and the multi-cycle run_sim regression cases (needs the experimental data; slow).
- `tests/regression/` holds exact fingerprints of run_sim output and of every figure
  function. After changing simulation or plotting code, check with
  `python -m tests.regression.make_references --check` (parallel); re-record with
  `python -m tests.regression.make_references --only <prefix>` only when a change
  is intended.
- Tuned SEI parameters live in `src/tuning.py` (`TUNED`, `tuned_params()`); don't
  hard-code them in scripts.

## Usage Examples

| Script / notebook | Shows |
|---|---|
| `notebooks_2026/dsei_sweep.py` (and `density_sweep.py`, `conductivity_sweep.py`) | Paper tuning steps: load cached sweep, heatmaps, selected values |
| `notebooks_2026/tuning_summary.py` | Composite multi-row heatmap figures |
| `notebooks_2026/model_vs_experiment.py` | Fully tuned model vs voltage, expansion, capacity, resistance |
| `2025_06_13_simulate_first_charge_cycle.ipynb` | Canonical single-charge sim + plot |
| `2026_02_25_ec_tuning.ipynb` | Parameter sweep with `plot_heatmaps_diff` |
| `2025_05_20_cycle_life_tuning.ipynb` | Multi-cycle aging + capacity fade |
| `2023_12_21_build_electrode_resistance_curves.ipynb` | `modelutils.decompose_resistance_curve` usage |
