"""SEI parameter tuning: sweep definitions, metrics, and cached results.

The paper's SEI parameters are tuned in four sequential steps (see
notebooks_2026/README.md). Each step sweeps two parameters on a grid, runs the
model for the three formation protocols at every grid point, and scores the
result against experiment:

    D_LEDC x D_LVDC    voltage and expansion RMSE    first charge / all ~400 h
    rho_LEDC x rho_LVDC voltage and expansion RMSE   first charge / all ~400 h
    kappa_LEDC x kappa_LVDC  |R_model - R_meas|      first RPT (HPPC)

A sweep is described by a SweepSpec. run_sweep() evaluates it, and
load_sweep() reads the cached result in notebooks_2026/data/ (either the
original notebook pickles or files written by save_sweep()).
"""

import functools
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import yaml
from scipy.interpolate import interp1d

from src import ioutils, paths, simutils

PROTOCOLS = ('base', 'fast', 'fast+')

# Experimental cell for each protocol (152074 is a second, unused baseline cell)
EXPT_CELLS = {'base': 152064, 'fast': 152071, 'fast+': 152098}

# Time window (hours) of the early expansion kink that is smoothed out
KINK_WINDOWS = {
    152064: (0.5, 0.75),
    152074: (0.5, 0.85),
    152071: (0.5, 0.6),
    152098: (0.5, 0.55),
}

# Experimental data is loaded up to this many hours
EXPT_MAX_HOURS = 600

# The first 30 minutes (pre-formation rest) are excluded from every RMSE
RMSE_IGNORE_TIME_S = (0, 30 * 60)

# Values assumed before a parameter is tuned (the run_sim defaults)
ASSUMED = dict(rho_ec=1.38, rho_vc=1.33, kappa_ec=0.081e-3, kappa_vc=0.049e-3)

# Values set by each tuning step. Per-protocol values are keyed by protocol.
TUNED = {
    'diff_vc': 1.22e-18,                                        # step 1
    'diff_ec': 6.66e-20,                                        # step 2
    'rho_vc': 1.5,                                              # step 3
    'rho_ec': {'base': 1.94, 'fast': 2.25, 'fast+': 1.63},      # step 3
    'kappa_vc': 0.0511e-3,                                      # step 4
    'kappa_ec': {'base': 0.0765e-3, 'fast': 0.0897e-3, 'fast+': 0.129e-3},  # step 4
}

# Resistance not captured by R_sei (positive electrode charge transfer, R0p)
R_OFFSET = 0.01


@dataclass(frozen=True)
class Measurement:
    t_s: float      # time in the model's time base [s]
    r_ohm: float    # measured resistance [Ohm]
    r_std: float    # error bar [Ohm]


@dataclass(frozen=True)
class CapacityPoint:
    t_s: float      # time in the model's time base [s]
    q_ah: float     # measured discharge capacity [Ah]
    q_std: float    # error bar [Ah]


# Measured values and their provenance live in this file
MEASUREMENTS_FILE = paths.repo_path('notebooks_2026', 'data', 'measurements.yaml')


def load_measurements(path=MEASUREMENTS_FILE):
    """(HPPC1, HPPC2, QD_RPT1, QD_RPT2), each keyed by protocol.

    HPPC resistance is compared at the midpoint of each HPPC sequence
    (sequence start + offset); see the file for sources.
    """
    with open(path) as f:
        m = yaml.safe_load(f)

    def hppc(key):
        return {p: Measurement(v['t_start_h'] * 3600 + v['offset_s'], v['r_ohm'], v['r_err_ohm'])
                for p, v in m['hppc_resistance'][key].items()}

    def capacity(key):
        return {p: CapacityPoint(v['t_h'] * 3600, v['q_ah'], v['q_err_ah'])
                for p, v in m['discharge_capacity'][key].items()}

    return hppc('hppc1'), hppc('hppc2'), capacity('rpt1'), capacity('rpt2')


# HPPC resistance after formation (HPPC1) and after formation aging (HPPC2);
# discharge capacity at the RPTs after formation and after formation aging
HPPC1, HPPC2, QD_RPT1, QD_RPT2 = load_measurements()


def _check_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown protocol {protocol!r}; expected one of {PROTOCOLS}')


def per_protocol(value, protocol):
    """Pick the per-protocol entry of a value that may be keyed by protocol."""
    return value[protocol] if isinstance(value, dict) else value


def tuned_params(protocol):
    """run_sim keyword arguments for the fully tuned model."""
    _check_protocol(protocol)
    return {key: per_protocol(TUNED[key], protocol) for key in
            ('diff_ec', 'diff_vc', 'rho_ec', 'rho_vc', 'kappa_ec', 'kappa_vc')}


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

WINDOW_INCLUDE_FLAG = {'fc': 0, 'all': 3, 'hppc1': 2}
WINDOW_LABELS = {
    'fc': 'First charge',
    'all': 'All ~400 h',
    'hppc1': 'First RPT (HPPC)',
}


@dataclass(frozen=True, eq=False)
class SweepSpec:
    """A two-parameter grid sweep over the three formation protocols.

    Matrices are indexed [row, col], with rows over row_param and columns
    over col_param. `fixed` holds the other run_sim arguments; a value may be
    a dict keyed by protocol. If the cached result was run with different
    fixed arguments, they are recorded in `cache_fixed` (see as_cached()).

    window: 'fc'    first charge only (include_flag=0), voltage + expansion RMSE
            'all'   through formation aging (include_flag=3), voltage + expansion RMSE
            'hppc1' through the first RPT (include_flag=2), resistance error
    """
    name: str
    row_param: str
    row_vec: np.ndarray
    col_param: str
    col_vec: np.ndarray
    window: str
    fixed: dict
    cache: str
    base_cell: int = EXPT_CELLS['base']
    v_range: tuple = (3.2, 4.2)
    smooth_strain: bool = True
    kink_overrides: dict = field(default_factory=dict)
    cache_fixed: dict = None

    @property
    def include_flag(self):
        return WINDOW_INCLUDE_FLAG[self.window]

    @property
    def metrics(self):
        return ('r_err',) if self.window == 'hppc1' else ('v_rmse', 'e_rmse')

    @property
    def shape(self):
        return (len(self.row_vec), len(self.col_vec))

    def sim_kwargs(self, protocol, row_value, col_value):
        kwargs = {key: per_protocol(value, protocol) for key, value in self.fixed.items()}
        kwargs[self.row_param] = row_value
        kwargs[self.col_param] = col_value
        kwargs.update(dt=1, include_flag=self.include_flag)
        return kwargs

    def expt_cell(self, protocol):
        _check_protocol(protocol)
        return self.base_cell if protocol == 'base' else EXPT_CELLS[protocol]

    def kink_window(self, protocol):
        return self.kink_overrides.get(protocol, KINK_WINDOWS[self.expt_cell(protocol)])

    def as_cached(self):
        """This spec with the fixed arguments its cached result was run with."""
        return self if self.cache_fixed is None else replace(self, fixed=self.cache_fixed)


DATA_DIR = paths.repo_path('notebooks_2026', 'data')

_D_TUNED = dict(diff_ec=TUNED['diff_ec'], diff_vc=TUNED['diff_vc'])

SPECS = {spec.name: spec for spec in [
    # Step 1: set D_LVDC from the first charge. The original sweep
    # (simulation_results_ec_tuning_diff_20260421.pkl) used baseline cell
    # 152074; the base protocol was re-run with 152064 so every step uses the
    # same cells. D_LVDC = 1.22e-18 is within 0.5 mV (mean voltage RMSE) of the
    # best D_LVDC with either cell; the valley is flat over ~2 decades.
    SweepSpec(
        name='dsei_fc',
        row_param='diff_ec', row_vec=np.logspace(-20, -16, 55),
        col_param='diff_vc', col_vec=np.logspace(-21, -12, 56),
        window='fc', fixed={}, v_range=(2.0, 4.2),
        cache=f'{DATA_DIR}/dsei_fc_20261015.pkl',
    ),
    # Step 2: set D_LEDC from all ~400 h, with rho/kappa at the ASSUMED values.
    # The cached sweep was run with the cache_fixed rho/kappa instead. Re-running
    # the D_LVDC = 1.22e-18 column with the ASSUMED values
    # (data/dsei_all_column_assumed_20261014.pkl) leaves the grid minimum at
    # D_LEDC = 6.66e-20 for all protocols; interpolating between grid points
    # moves it about 5% lower (~6.3e-20).
    SweepSpec(
        name='dsei_all',
        row_param='diff_ec', row_vec=np.logspace(-20, -16, 35),
        col_param='diff_vc', col_vec=np.logspace(-21, -12, 36),
        window='all',
        fixed={},
        cache_fixed=dict(rho_ec=1.29, rho_vc=1.49, kappa_ec=0.090e-3, kappa_vc=0.051e-3),
        kink_overrides={'base': (0.5, 0.85)},
        cache=f'{DATA_DIR}/simulation_results_clean_dsei_sweep_20260422.pkl',
    ),
    # Step 3 (first-charge counterpart)
    SweepSpec(
        name='rho_fc',
        row_param='rho_ec', row_vec=np.linspace(0.5, 4, 55),
        col_param='rho_vc', col_vec=np.linspace(0.5, 4, 56),
        window='fc', fixed=_D_TUNED, v_range=(2.0, 4.2),
        cache=f'{DATA_DIR}/simulation_results_first_charge_density_20260429.pkl',
    ),
    # Step 3: set rho_LVDC and rho_LEDC from the all ~400 h expansion
    SweepSpec(
        name='rho_all',
        row_param='rho_ec', row_vec=np.linspace(0.5, 4, 35),
        col_param='rho_vc', col_vec=np.linspace(0.5, 4, 36),
        window='all', fixed=_D_TUNED, smooth_strain=False,
        cache=f'{DATA_DIR}/simulation_results_density_sweep_20260426.pkl',
    ),
    # Step 4: set kappa_LVDC and kappa_LEDC from the first HPPC resistance
    SweepSpec(
        name='kappa_hppc1',
        row_param='kappa_ec', row_vec=np.linspace(5e-5, 5e-4, 35),
        col_param='kappa_vc', col_vec=np.linspace(1e-5, 1e-4, 36),
        window='hppc1',
        fixed=dict(_D_TUNED, rho_ec=TUNED['rho_ec'], rho_vc=TUNED['rho_vc']),
        cache=f'{DATA_DIR}/resistance_optimization_results_20260504.pkl',
    ),
]}


# ---------------------------------------------------------------------------
# Experimental data
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _fetch(cellid):
    expt, _ = ioutils.fetch_experimental_data(cellid, EXPT_MAX_HOURS)
    return expt


def load_experiments(spec):
    """Experimental data for each protocol, keyed by protocol.

    'daq_strain' is the strain the model is scored against: smoothed over the
    early kink when spec.smooth_strain is set. The unsmoothed strain is kept
    as 'daq_strain_raw'.
    """
    experiments = {}
    for protocol in PROTOCOLS:
        expt = dict(_fetch(spec.expt_cell(protocol)))
        expt['daq_strain_raw'] = expt['daq_strain']
        if spec.smooth_strain:
            daq_time = np.array(expt['daq_time'])
            time_hours = (daq_time - daq_time[0]) / 3600.0
            expt['daq_strain'] = ioutils.smooth_kink_region(
                time_hours, np.array(expt['daq_strain_raw']), *spec.kink_window(protocol))
        experiments[protocol] = expt
    return experiments


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse_metrics(df, expt, v_range):
    """Voltage RMSE [V] and expansion RMSE [um] of one simulation."""
    v_rmse, _, _ = simutils.calculate_rmse(
        expt['arbin_time'], expt['arbin_voltage'], df['t'], df['vt'],
        ignore_time_range=RMSE_IGNORE_TIME_S, y_range=v_range)
    e_rmse, _, _ = simutils.calculate_rmse(
        expt['daq_time'], expt['daq_strain'], df['t'], df['expansion'] * 1e6,
        ignore_time_range=RMSE_IGNORE_TIME_S)
    return {'v_rmse': v_rmse, 'e_rmse': e_rmse}


def model_resistance(df, t_s):
    """Modeled cell resistance (R_sei + R_OFFSET) at time t_s [Ohm]."""
    f = interp1d(df['t'], df['R_sei'] + R_OFFSET, bounds_error=False, fill_value='extrapolate')
    return float(f(t_s))


def resistance_error(df, t_s, r_meas):
    """|R_model - R_meas| at time t_s [Ohm]."""
    return abs(model_resistance(df, t_s) - r_meas)


def evaluate_point(spec, protocol, row_value, col_value, expt=None):
    """Run one simulation and return its metrics as {metric: value}."""
    df = simutils.run_sim(protocol, **spec.sim_kwargs(protocol, row_value, col_value))
    if spec.window == 'hppc1':
        m = HPPC1[protocol]
        return {'r_err': resistance_error(df, m.t_s, m.r_ohm)}
    return rmse_metrics(df, expt, spec.v_range)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SweepResult:
    spec: SweepSpec
    row_vec: np.ndarray
    col_vec: np.ndarray
    metrics: dict   # {metric: {protocol: matrix}}


# State for worker processes (and the serial path), set by _init_worker
_WORKER = {}


def _init_worker(spec, experiments, evaluate):
    _WORKER.update(spec=spec, experiments=experiments, evaluate=evaluate)


def _evaluate_task(task):
    i, j, protocol = task
    spec = _WORKER['spec']
    try:
        out = _WORKER['evaluate'](spec, protocol, spec.row_vec[i], spec.col_vec[j],
                                  _WORKER['experiments'].get(protocol))
    except Exception:
        out = None
    return i, j, protocol, out


def run_sweep(spec, n_jobs=1, experiments=None, progress=True, evaluate=evaluate_point):
    """Evaluate every grid point of a sweep. Failed simulations are left as NaN."""
    if experiments is None:
        experiments = {} if spec.window == 'hppc1' else load_experiments(spec)

    metrics = {m: {p: np.full(spec.shape, np.nan) for p in PROTOCOLS} for m in spec.metrics}
    tasks = [(i, j, p) for i in range(spec.shape[0]) for j in range(spec.shape[1])
             for p in PROTOCOLS]

    if n_jobs == 1:
        _init_worker(spec, experiments, evaluate)
        outputs = map(_evaluate_task, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(n_jobs, initializer=_init_worker,
                                   initargs=(spec, experiments, evaluate))
        outputs = pool.map(_evaluate_task, tasks, chunksize=max(1, len(tasks) // (n_jobs * 50)))

    n_failed = 0
    t_start = time.time()
    try:
        for count, (i, j, protocol, out) in enumerate(outputs, start=1):
            if out is None:
                n_failed += 1
            else:
                for metric, value in out.items():
                    metrics[metric][protocol][i, j] = value
            if progress and (count % 30 == 0 or count == len(tasks)):
                elapsed = time.time() - t_start
                print(f'\r[{spec.name}] {count}/{len(tasks)} simulations, '
                      f'{elapsed / 60:.1f} min', end='', file=sys.stderr)
    finally:
        if pool is not None:
            pool.shutdown()
    if progress:
        print(file=sys.stderr)
    if n_failed:
        print(f'[{spec.name}] {n_failed} simulations failed (left as NaN)', file=sys.stderr)

    return SweepResult(spec, spec.row_vec, spec.col_vec, metrics)


# ---------------------------------------------------------------------------
# Cached results
# ---------------------------------------------------------------------------

_FORMAT = 'sweep-v1'

# Key patterns in the original notebook pickles; {n} is 1/2/3 and {p} is
# base/fast/fp for base/fast/fast+.
_LEGACY_KEYS = {
    'dsei_fc': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'dsei_all': {'v_rmse': 'all_{p}', 'e_rmse': 'all_{p}_e'},
    'rho_fc': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'rho_all': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'kappa_hppc1': {'r_err': 'rmse{n}_e_mat'},
}
_LEGACY_PROTOCOL = {'base': 'base', 'fast': 'fast', 'fast+': 'fp'}


def save_sweep(result, path):
    spec = result.spec
    data = {
        'format': _FORMAT,
        'spec': spec.name,
        'row_param': spec.row_param, 'row_vec': result.row_vec,
        'col_param': spec.col_param, 'col_vec': result.col_vec,
        'window': spec.window,
        'fixed': spec.fixed,
        'metrics': result.metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_sweep(spec, path=None):
    """Load a cached sweep (spec.cache by default)."""
    with open(path or spec.cache, 'rb') as f:
        raw = pickle.load(f)

    if raw.get('format') == _FORMAT:
        if raw['spec'] != spec.name:
            raise ValueError(f'{path} holds sweep {raw["spec"]!r}, not {spec.name!r}')
        return SweepResult(spec, raw['row_vec'], raw['col_vec'], raw['metrics'])

    if spec.name not in _LEGACY_KEYS:
        raise ValueError(f'No legacy format known for sweep {spec.name!r}')
    metrics = {}
    for metric, pattern in _LEGACY_KEYS[spec.name].items():
        metrics[metric] = {
            p: raw[pattern.format(n=n, p=_LEGACY_PROTOCOL[p])]
            for n, p in enumerate(PROTOCOLS, start=1)
        }
    return SweepResult(spec, spec.row_vec, spec.col_vec, metrics)


# ---------------------------------------------------------------------------
# Selecting the tuned value
# ---------------------------------------------------------------------------

def nearest_index(vec, value):
    return int(np.argmin(np.abs(np.asarray(vec) - value)))


def select_row(result, metric, pin_col):
    """Row-parameter value minimizing `metric` in the column nearest pin_col.

    Returns {protocol: row value}.
    """
    j = nearest_index(result.col_vec, pin_col)
    return {p: result.row_vec[int(np.nanargmin(result.metrics[metric][p][:, j]))]
            for p in PROTOCOLS}
