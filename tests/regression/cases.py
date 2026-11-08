"""Regression cases for run_sim() and the plotting code.

Each case is a name and a zero-argument function returning a fingerprint.
make_references.py records the fingerprints from the current code; the
tests in test_regression.py compare against them.
"""

import numpy as np
from matplotlib import pyplot as plt

from src import cellsim, paths, plotter, simutils, tuning
from tests.regression.fingerprint import dataframe_fingerprint, figure_fingerprint

PROTOCOLS = ('base', 'fast', 'fast+')

# ---------------------------------------------------------------------------
# run_sim
# ---------------------------------------------------------------------------


def _run(protocol, **kwargs):
    return lambda: dataframe_fingerprint(simutils.run_sim(protocol, **kwargs))


def _follow_current(protocol):
    current = np.concatenate([np.zeros(360), np.full(1440, 0.25), np.full(360, -0.25)])
    return _run(protocol, dt=5, include_flag=1, to_follow_current=True, current_vec=current)


RUN_SIM_CASES = {}
for _p in PROTOCOLS:
    _tuned = tuning.tuned_params(_p)
    for _flag in (0, 1, 2, 3, 4):
        RUN_SIM_CASES[f'run_sim/{_p}/flag{_flag}'] = _run(_p, **_tuned, dt=1, include_flag=_flag)
    RUN_SIM_CASES[f'run_sim/{_p}/flag5_derated'] = _run(
        _p, **_tuned, dt=1, include_flag=5, dsei_derating_cycling=0.1)
    RUN_SIM_CASES[f'run_sim/{_p}/flag2_no_boost'] = _run(_p, **_tuned, dt=1, include_flag=2, to_boost=False)
    RUN_SIM_CASES[f'run_sim/{_p}/follow_current'] = _follow_current(_p)
RUN_SIM_CASES['run_sim/base/defaults'] = _run('base')
RUN_SIM_CASES['run_sim/base/dt5_flag2'] = _run('base', dt=5, include_flag=2)
RUN_SIM_CASES['run_sim/fast+/sei1_only'] = _run('fast+', dt=1, include_flag=2, sei_reactions='sei1')
RUN_SIM_CASES['run_sim/fast+/sei2_only'] = _run('fast+', dt=1, include_flag=2, sei_reactions='sei2')

# Only first-charge and follow-current runs are quick; the rest need --runslow
SLOW_RUN_SIM_CASES = {k for k in RUN_SIM_CASES
                      if not (k.endswith('/flag0') or k.endswith('/follow_current')
                              or k.endswith('/defaults'))}

# ---------------------------------------------------------------------------
# Simulation plotting methods
# ---------------------------------------------------------------------------


def _short_sim():
    cell = cellsim.Cell()
    cell.load_config(paths.repo_path('params', 'default.yaml'))
    sim = cellsim.Simulation(cell, sim_time_s=8 * 3600, dt=5.0)
    sim.run_rest(1, rest_time_hrs=0.5)
    sim.run_chg_cccv(1, 2.5 / 2, 2.5 / 20, 4.2)
    sim.run_rest(1, rest_time_hrs=0.5)
    sim.run_dch_cccv(2, -2.5 / 2, -2.5 / 2, 3.0)
    return sim


def _plot(method, **kwargs):
    def case():
        plotter.initialize(plt)
        sim = _short_sim()
        plt.close('all')
        getattr(sim, method)(**kwargs)
        fp = figure_fingerprint(plt.gcf())
        plt.close('all')
        return fp
    return case


PLOT_CASES = {
    'plot/plot': _plot('plot', to_save=False),
    'plot/plot_xlims': _plot('plot', to_save=False, xlims=(0, 3)),
    'plot/view_1': _plot('plot_view_1', xlims=(0, 4)),
    'plot/view_2': _plot('plot_view_2', xlims=(0, 3)),
    'plot/view_3': _plot('plot_view_3', xlims=(0, 5)),
    'plot/view_4': _plot('plot_view_4', xlims=(0.4, 2)),
    'plot/view_4_options': _plot('plot_view_4', to_annotate=False, ylim_current=(0, 1)),
}

# ---------------------------------------------------------------------------
# Legacy heatmap functions in simutils
# ---------------------------------------------------------------------------


def _mats(shape, seed):
    rng = np.random.default_rng(seed)
    return [rng.random(shape) for _ in range(3)]


def _heatmap(func, x_vec, y_vec, seed, **kwargs):
    def case():
        plotter.initialize(plt)
        plt.close('all')
        mats = _mats((len(x_vec), len(y_vec)), seed)
        getattr(simutils, func)(x_vec, y_vec, *mats, 'Label', **kwargs)
        fp = figure_fingerprint(plt.gcf())
        plt.close('all')
        return fp
    return case


_RHO = (np.linspace(0.5, 4, 8), np.linspace(0.5, 4, 9))
_DIFF = (np.logspace(-20, -16, 8), np.logspace(-21, -12, 9))
_KAPPA = (np.linspace(5e-5, 5e-4, 8) * 1e3, np.linspace(1e-5, 1e-4, 9) * 1e3)

HEATMAP_CASES = {
    'heatmap/rho_plain': _heatmap('plot_heatmaps_rho', *_RHO, 1, zmin=0, zmax=1),
    'heatmap/rho_annotated': _heatmap('plot_heatmaps_rho', *_RHO, 2, to_annotate=True),
    'heatmap/rho_horizontal': _heatmap('plot_heatmaps_rho', *_RHO, 3, to_annotate=True,
                                       annotation_orientation='horizontal', annotation_coord=2.0),
    'heatmap/diff_vertical': _heatmap('plot_heatmaps_diff', *_DIFF, 4, to_annotate=True,
                                      annotation_coord=1.22e-18, zmin=0.04, zmax=0.2),
    'heatmap/diff_line_only': _heatmap('plot_heatmaps_diff', *_DIFF, 5, to_annotate=True,
                                       annotation_coord=1.22e-18, annotation_line_only=True),
    'heatmap/diff_horizontal_per_panel': _heatmap(
        'plot_heatmaps_diff', *_DIFF, 6, to_annotate=True, annotation_orientation='horizontal',
        annotation_coord=[1e-19, 1e-18, 1e-17]),
    'heatmap/diff_minimum': _heatmap('plot_heatmaps_diff', *_DIFF, 7, to_annotate=True,
                                     annotation_orientation='minimum'),
    'heatmap/kappa_vertical': _heatmap('plot_heatmaps_kappa', *_KAPPA, 8, to_annotate=True,
                                       annotation_coord=0.05, zmin=0, zmax=10),
    'heatmap/kappa_plain': _heatmap('plot_heatmaps_kappa', *_KAPPA, 9),
}

# ---------------------------------------------------------------------------
# Sweep heatmaps (src/tuning_plots.py) from the cached sweep results
# ---------------------------------------------------------------------------

# First-charge sweeps only mark the pinned value (they don't set the row parameter)
_LINE_ONLY = {'dsei_fc', 'rho_fc'}

_SWEEP_PINS = {'dsei_fc': 'diff_vc', 'dsei_all': 'diff_vc', 'rho_fc': 'rho_vc',
               'rho_all': 'rho_vc', 'kappa_hppc1': 'kappa_vc'}


def _sweep(name, metric):
    def case():
        from src import tuning_plots
        plotter.initialize(plt)
        plt.close('all')
        result = tuning.load_sweep(tuning.SPECS[name])
        fig = tuning_plots.plot_heatmap_trio(result, metric, pin_col=tuning.TUNED[_SWEEP_PINS[name]],
                                             title='t', mark_minimum=name not in _LINE_ONLY)
        fp = figure_fingerprint(fig)
        plt.close('all')
        return fp
    return case


SWEEP_CASES = {f'sweep/{name}/{metric}': _sweep(name, metric)
               for name, spec in tuning.SPECS.items() for metric in spec.metrics}

ALL_CASES = {**RUN_SIM_CASES, **PLOT_CASES, **HEATMAP_CASES, **SWEEP_CASES}
