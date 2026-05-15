# Formation Modeling

Electrochemical simulation code for lithium-ion battery formation modeling.
Published in the Journal of the Electrochemical Society (December 2023).

## Project Structure

```
src/            Core Python source
  cellsim.py    Cell and Simulation classes (main simulation engine)
  simutils.py   run_sim() orchestration, RMSE, heatmap plotting
  modelutils.py Electrochemistry lookup functions (OCP, expansion, resistance)
  dvdq.py       dV/dQ analysis tools
  ioutils.py    Data loading utilities (cell IDs → CSV paths)
  plotter.py    Matplotlib style initialization
params/         YAML configuration files for cell parameters
  default.yaml  Primary parameter set
notebooks/      Jupyter analysis notebooks (YYYY_MM_DD naming)
tests/          pytest test suite
data/           Experimental data (not committed; download from Google Drive)
outputs/        Figures and pickled simulation results
```

## Running Simulations

Always run from the repo root (not from inside `src/`):

```python
from src import cellsim, simutils

# Single simulation
df = simutils.run_sim('base')          # formation_type: 'base' | 'fast' | 'fast+'
df = simutils.run_sim('fast', include_flag=2)  # simulate through first RPT

# Manual simulation
cell = cellsim.Cell()
cell.load_config('params/default.yaml')
sim = cellsim.Simulation(cell, sim_time_s=1000*3600, dt=1.0)
sim.run_rest(1, rest_time_hrs=0.5)
sim.run_chg_cccv(1, icc=2.5/10, icv=2.5/20, vmax=4.2)
df = sim.get_results()  # returns a pandas DataFrame
```

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

Cell parameters are loaded from YAML via `cell.load_config(path)`. Key SEI parameters:
- `D_SEI11 / D_SEI12` — EC diffusivity (inner/outer layer)
- `D_SEI21 / D_SEI22` — VC diffusivity (inner/outer layer)
- `k_SEI1 / k_SEI2` — kinetic rate constants
- `U_SEI1 / U_SEI2` — equilibrium reaction potentials (set to -1000 to disable a reaction)
- `kappa_SEI1 / kappa_SEI2` — SEI conductivities
- `rho_SEI1 / rho_SEI2` — SEI densities

## Development Guidelines

- Do **not** commit raw data files (use Google Drive; see `data/README.md`).
- One figure per notebook; name notebooks `YYYY_MM_DD_description.ipynb`.
- Prototype in notebooks, then refactor proven code into `src/`.
- Python 3.8.8 compatible.
- Run tests with `pytest tests/`.

## Notebook Usage Examples

| Notebook | Shows |
|---|---|
| `2025_06_13_simulate_first_charge_cycle.ipynb` | Canonical single-charge sim + plot |
| `2026_02_25_ec_tuning.ipynb` | Parameter sweep with `plot_heatmaps_diff` |
| `2025_05_20_cycle_life_tuning.ipynb` | Multi-cycle aging + capacity fade |
| `2023_12_12_dynamic_dvdq.ipynb` | dV/dQ analysis via `dvdq.py` |
| `2023_12_21_build_electrode_resistance_curves.ipynb` | `modelutils.decompose_resistance_curve` usage |
