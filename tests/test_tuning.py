"""Tests for src/tuning.py

The cached sweep pickles in notebooks_2026/data/ were produced by the original
2026 notebooks. These tests check that the refactored sweep code reproduces
them: the sweep specs carry the same settings as the notebooks, the cached
matrices load unchanged, the selection rules recover the tuned values in the
paper table, and (with --runslow) re-simulated grid points match the cache.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from src import simutils, tuning


DATA_DIR = 'notebooks_2026/data'

LEGACY_FILES = {
    'dsei_fc': 'simulation_results_ec_tuning_diff_20260421.pkl',
    'dsei_all': 'simulation_results_clean_dsei_sweep_20260422.pkl',
    'rho_fc': 'simulation_results_first_charge_density_20260429.pkl',
    'rho_all': 'simulation_results_density_sweep_20260426.pkl',
    'kappa_hppc1': 'resistance_optimization_results_20260504.pkl',
}

# Current caches; dsei_fc was re-run with baseline cell 152064 (base protocol only)
CACHE_FILES = dict(LEGACY_FILES, dsei_fc='dsei_fc_20261015.pkl')

HAS_EXPT_DATA = os.path.isdir('data/raw/from-voltaiq')
needs_expt_data = pytest.mark.skipif(not HAS_EXPT_DATA, reason='experimental data not downloaded')


def _raw(name):
    with open(os.path.join(DATA_DIR, LEGACY_FILES[name]), 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Specs carry the same settings as the original notebooks
# ---------------------------------------------------------------------------

class TestSpecsMatchNotebooks:

    def test_all_steps_defined(self):
        assert set(tuning.SPECS) == set(LEGACY_FILES)

    def test_cache_paths(self):
        for name, fname in CACHE_FILES.items():
            assert tuning.SPECS[name].cache == os.path.join(DATA_DIR, fname)

    def test_dsei_fc(self):
        s = tuning.SPECS['dsei_fc']
        assert (s.row_param, s.col_param) == ('diff_ec', 'diff_vc')
        np.testing.assert_array_equal(s.row_vec, np.logspace(-20, -16, 55))
        np.testing.assert_array_equal(s.col_vec, np.logspace(-21, -12, 56))
        assert s.window == 'fc' and s.include_flag == 0
        assert s.expt_cell('base') == 152064
        assert s.kink_window('base') == (0.5, 0.75)
        assert s.v_range == (2.0, 4.2)
        assert s.smooth_strain
        assert s.sim_kwargs('fast', 1e-19, 1e-18) == dict(
            diff_ec=1e-19, diff_vc=1e-18, dt=1, include_flag=0)

    def test_dsei_all(self):
        s = tuning.SPECS['dsei_all']
        raw = _raw('dsei_all')
        np.testing.assert_array_equal(s.row_vec, raw['diff_ec_vec'])
        np.testing.assert_array_equal(s.col_vec, raw['diff_vc_vec'])
        assert s.window == 'all' and s.include_flag == 3
        assert s.expt_cell('base') == 152064
        assert s.kink_window('base') == (0.5, 0.85)
        assert s.v_range == (3.2, 4.2)
        assert s.smooth_strain
        # Step 2 uses the assumed rho/kappa, like step 1 ...
        assert s.sim_kwargs('base', 1e-19, 1e-18) == dict(
            diff_ec=1e-19, diff_vc=1e-18, dt=1, include_flag=3)
        # ... but the cached sweep was run with the values stored in the pickle
        assert s.as_cached().sim_kwargs('base', 1e-19, 1e-18) == dict(
            diff_ec=1e-19, diff_vc=1e-18,
            rho_ec=raw['rho_ec'], rho_vc=raw['rho_vc'],
            kappa_ec=raw['kappa_ec'], kappa_vc=raw['kappa_vc'],
            dt=1, include_flag=3)

    def test_rho_fc(self):
        s = tuning.SPECS['rho_fc']
        assert (s.row_param, s.col_param) == ('rho_ec', 'rho_vc')
        np.testing.assert_array_equal(s.row_vec, np.linspace(0.5, 4, 55))
        np.testing.assert_array_equal(s.col_vec, np.linspace(0.5, 4, 56))
        assert s.window == 'fc' and s.include_flag == 0
        assert s.expt_cell('base') == 152064
        assert s.kink_window('base') == (0.5, 0.75)
        assert s.v_range == (2.0, 4.2)
        assert s.smooth_strain
        assert s.sim_kwargs('fast+', 1.0, 2.0) == dict(
            diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=1.0, rho_vc=2.0,
            dt=1, include_flag=0)

    def test_rho_all(self):
        s = tuning.SPECS['rho_all']
        np.testing.assert_array_equal(s.row_vec, np.linspace(0.5, 4, 35))
        np.testing.assert_array_equal(s.col_vec, np.linspace(0.5, 4, 36))
        assert s.window == 'all' and s.include_flag == 3
        assert s.expt_cell('base') == 152064
        assert s.v_range == (3.2, 4.2)
        assert not s.smooth_strain
        assert s.sim_kwargs('base', 1.0, 2.0) == dict(
            diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=1.0, rho_vc=2.0,
            dt=1, include_flag=3)

    def test_kappa_hppc1(self):
        s = tuning.SPECS['kappa_hppc1']
        assert (s.row_param, s.col_param) == ('kappa_ec', 'kappa_vc')
        np.testing.assert_array_equal(s.row_vec, np.linspace(5e-5, 5e-4, 35))
        np.testing.assert_array_equal(s.col_vec, np.linspace(1e-5, 1e-4, 36))
        assert s.window == 'hppc1' and s.include_flag == 2
        assert s.metrics == ('r_err',)
        for protocol, rho_ec in zip(tuning.PROTOCOLS, (1.94, 2.25, 1.63)):
            assert s.sim_kwargs(protocol, 1e-4, 5e-5) == dict(
                diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=rho_ec, rho_vc=1.5,
                kappa_ec=1e-4, kappa_vc=5e-5, dt=1, include_flag=2)

    def test_expt_cells_for_fast_protocols(self):
        for s in tuning.SPECS.values():
            assert s.expt_cell('fast') == 152071
            assert s.expt_cell('fast+') == 152098


# ---------------------------------------------------------------------------
# Tuned parameter registry (paper table)
# ---------------------------------------------------------------------------

class TestTunedParams:

    def test_values_used_in_model_vs_experiment(self):
        expected = {
            'base':  dict(diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=1.94, rho_vc=1.5,
                          kappa_ec=0.0765e-3, kappa_vc=0.0511e-3),
            'fast':  dict(diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=2.25, rho_vc=1.5,
                          kappa_ec=0.0897e-3, kappa_vc=0.0511e-3),
            'fast+': dict(diff_ec=6.66e-20, diff_vc=1.22e-18, rho_ec=1.63, rho_vc=1.5,
                          kappa_ec=0.129e-3, kappa_vc=0.0511e-3),
        }
        for protocol, params in expected.items():
            assert tuning.tuned_params(protocol) == params

    def test_assumed_values_are_run_sim_defaults(self):
        import inspect
        defaults = {k: v.default for k, v in
                    inspect.signature(simutils.run_sim).parameters.items()}
        for key, value in tuning.ASSUMED.items():
            assert defaults[key] == value

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError):
            tuning.tuned_params('slow')


# ---------------------------------------------------------------------------
# Cached results load unchanged
# ---------------------------------------------------------------------------

LEGACY_KEYS = {
    'dsei_fc': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'rho_fc': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'rho_all': {'v_rmse': 'rmse{n}_v_mat', 'e_rmse': 'rmse{n}_e_mat'},
    'dsei_all': {'v_rmse': 'all_{p}', 'e_rmse': 'all_{p}_e'},
    'kappa_hppc1': {'r_err': 'rmse{n}_e_mat'},
}


@pytest.mark.parametrize('name', sorted(set(LEGACY_FILES) - {'dsei_fc'}))
def test_load_cached_matches_raw_pickle(name):
    spec = tuning.SPECS[name]
    result = tuning.load_sweep(spec)
    raw = _raw(name)

    np.testing.assert_array_equal(result.row_vec, spec.row_vec)
    np.testing.assert_array_equal(result.col_vec, spec.col_vec)
    assert set(result.metrics) == set(spec.metrics)

    legacy_protocol = {'base': 'base', 'fast': 'fast', 'fast+': 'fp'}
    for metric, pattern in LEGACY_KEYS[name].items():
        for n, protocol in enumerate(tuning.PROTOCOLS, start=1):
            key = pattern.format(n=n, p=legacy_protocol[protocol])
            mat = result.metrics[metric][protocol]
            assert mat.shape == (len(spec.row_vec), len(spec.col_vec))
            np.testing.assert_array_equal(mat, raw[key])


def test_dsei_fc_cache_rerun_base_only():
    # Fast and fast+ are the original results; only the base protocol (now cell
    # 152064 instead of 152074) was re-run.
    result = tuning.load_sweep(tuning.SPECS['dsei_fc'])
    raw = _raw('dsei_fc')
    for metric, key in (('v_rmse', 'rmse{n}_v_mat'), ('e_rmse', 'rmse{n}_e_mat')):
        for n, protocol in ((2, 'fast'), (3, 'fast+')):
            np.testing.assert_array_equal(result.metrics[metric][protocol], raw[key.format(n=n)])
        base = result.metrics[metric]['base']
        assert not np.isnan(base).any()
        assert not np.array_equal(base, raw[key.format(n=1)])


def test_save_load_roundtrip(tmp_path):
    spec = tuning.SPECS['kappa_hppc1']
    result = tuning.load_sweep(spec)
    path = tmp_path / 'sweep.pkl'
    tuning.save_sweep(result, path)
    reloaded = tuning.load_sweep(spec, path)
    for protocol in tuning.PROTOCOLS:
        np.testing.assert_array_equal(reloaded.metrics['r_err'][protocol],
                                      result.metrics['r_err'][protocol])


def test_load_rejects_mismatched_spec(tmp_path):
    result = tuning.load_sweep(tuning.SPECS['kappa_hppc1'])
    path = tmp_path / 'sweep.pkl'
    tuning.save_sweep(result, path)
    with pytest.raises(ValueError):
        tuning.load_sweep(tuning.SPECS['rho_all'], path)


# ---------------------------------------------------------------------------
# Selection rules recover the tuned values
# ---------------------------------------------------------------------------

class TestSelection:

    def test_step2_sets_d_ledc(self):
        result = tuning.load_sweep(tuning.SPECS['dsei_all'])
        sel = tuning.select_row(result, 'v_rmse', pin_col=tuning.TUNED['diff_vc'])
        for protocol in tuning.PROTOCOLS:
            assert sel[protocol] == pytest.approx(tuning.TUNED['diff_ec'], rel=1e-3)

    def test_step3_sets_rho_ledc(self):
        result = tuning.load_sweep(tuning.SPECS['rho_all'])
        sel = tuning.select_row(result, 'e_rmse', pin_col=tuning.TUNED['rho_vc'])
        for protocol in tuning.PROTOCOLS:
            assert sel[protocol] == pytest.approx(tuning.TUNED['rho_ec'][protocol], abs=0.005)

    def test_step4_sets_kappa_ledc(self):
        result = tuning.load_sweep(tuning.SPECS['kappa_hppc1'])
        sel = tuning.select_row(result, 'r_err', pin_col=tuning.TUNED['kappa_vc'])
        for protocol in tuning.PROTOCOLS:
            assert sel[protocol] == pytest.approx(tuning.TUNED['kappa_ec'][protocol], rel=5e-3)

    def test_pinned_value_snaps_to_nearest_column(self):
        result = tuning.load_sweep(tuning.SPECS['dsei_all'])
        assert tuning.nearest_index(result.col_vec, 1.22e-18) == 12


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _synthetic_run():
    t = np.arange(0, 7200.0, 10.0)
    return pd.DataFrame({
        't': t,
        'vt': 3.0 + t / 7200,
        'expansion': 1e-6 * (10 + t / 360),
        'R_sei': 0.005 + t * 1e-7,
    })


def _synthetic_expt():
    t = np.arange(0, 7200.0, 7.0)
    return {
        'arbin_time': pd.Series(t),
        'arbin_voltage': pd.Series(3.0 + t / 7200 + 0.01),
        'daq_time': pd.Series(t),
        'daq_strain': pd.Series(10 + t / 360 + 2.0),
    }


def test_rmse_metrics_match_calculate_rmse():
    df, expt = _synthetic_run(), _synthetic_expt()
    out = tuning.rmse_metrics(df, expt, v_range=(3.2, 4.2))
    v, _, _ = simutils.calculate_rmse(expt['arbin_time'], expt['arbin_voltage'], df['t'], df['vt'],
                                      ignore_time_range=(0, 1800), y_range=(3.2, 4.2))
    e, _, _ = simutils.calculate_rmse(expt['daq_time'], expt['daq_strain'], df['t'],
                                      df['expansion'] * 1e6, ignore_time_range=(0, 1800))
    assert out == {'v_rmse': v, 'e_rmse': e}
    assert out['v_rmse'] == pytest.approx(0.01)
    assert out['e_rmse'] == pytest.approx(2.0)


def test_resistance_error_at_hppc1():
    df = _synthetic_run()
    t_meas, r_meas = 3600.0, 0.02
    expected = abs(0.005 + t_meas * 1e-7 + tuning.R_OFFSET - r_meas)
    assert tuning.resistance_error(df, t_meas, r_meas) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _fake_evaluate(spec, protocol, row, col, expt=None):
    if row == 2 and col == 30:
        raise RuntimeError('solver failed')
    offset = tuning.PROTOCOLS.index(protocol)
    return {m: row * 100 + col + offset for m in spec.metrics}


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_run_sweep_assembles_grid(n_jobs):
    spec = tuning.SweepSpec(
        name='toy', row_param='rho_ec', row_vec=np.array([1, 2]),
        col_param='rho_vc', col_vec=np.array([10, 20, 30]),
        window='fc', fixed={}, cache='unused.pkl')
    result = tuning.run_sweep(spec, n_jobs=n_jobs, experiments={}, progress=False,
                              evaluate=_fake_evaluate)

    for metric in ('v_rmse', 'e_rmse'):
        for offset, protocol in enumerate(tuning.PROTOCOLS):
            mat = result.metrics[metric][protocol]
            assert mat.shape == (2, 3)
            assert mat[0, 0] == 110 + offset
            assert mat[1, 1] == 220 + offset
            assert np.isnan(mat[1, 2])


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def test_plot_heatmap_trio_marks_selection(tmp_path):
    from src import tuning_plots

    result = tuning.load_sweep(tuning.SPECS['rho_all'])
    savename = tmp_path / 'trio.svg'
    fig = tuning_plots.plot_heatmap_trio(result, 'e_rmse', pin_col=1.5, savename=savename)

    assert savename.exists()
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title() for ax in panels] == ['Base Formation', 'Fast Formation', 'Fast+ Formation']
    sel = tuning.select_row(result, 'e_rmse', pin_col=1.5)
    for ax, protocol in zip(panels, tuning.PROTOCOLS):
        star = [line for line in ax.get_lines() if line.get_marker() == '*'][0]
        assert star.get_ydata()[0] == pytest.approx(sel[protocol])


# ---------------------------------------------------------------------------
# Re-simulated grid points reproduce the cache (slow)
# ---------------------------------------------------------------------------

SPOT_POINTS = {
    'dsei_fc': (20, 25),
    'rho_fc': (20, 30),
    'dsei_all': (7, 12),
    'rho_all': (10, 10),
    'kappa_hppc1': (3, 16),
}


@pytest.mark.slow
@pytest.mark.parametrize('name', sorted(SPOT_POINTS))
def test_resimulated_point_matches_cache(name):
    spec = tuning.SPECS[name].as_cached()
    if spec.window != 'hppc1' and not HAS_EXPT_DATA:
        pytest.skip('experimental data not downloaded')
    cached = tuning.load_sweep(spec)
    experiments = tuning.load_experiments(spec) if spec.window != 'hppc1' else {}
    i, j = SPOT_POINTS[name]

    for protocol in tuning.PROTOCOLS:
        out = tuning.evaluate_point(spec, protocol, spec.row_vec[i], spec.col_vec[j],
                                    experiments.get(protocol))
        for metric in spec.metrics:
            assert out[metric] == pytest.approx(cached.metrics[metric][protocol][i, j], rel=1e-9), \
                (protocol, metric)


def test_specs_without_cache_fixed_are_unchanged_by_as_cached():
    for spec in tuning.SPECS.values():
        if spec.cache_fixed is None:
            assert spec.as_cached() is spec


def test_step2_uses_same_assumed_values_as_step1():
    s1, s2 = tuning.SPECS['dsei_fc'], tuning.SPECS['dsei_all']
    assert s1.fixed == s2.fixed == {}


def test_plot_heatmap_trio_line_only_marks_exact_pin(tmp_path):
    from src import tuning_plots

    result = tuning.load_sweep(tuning.SPECS['dsei_fc'])
    fig = tuning_plots.plot_heatmap_trio(result, 'v_rmse', pin_col=1.22e-18, mark_minimum=False)
    for ax in [ax for ax in fig.axes if ax.get_title()]:
        assert not [line for line in ax.get_lines() if line.get_marker() == '*']
        (vline,) = ax.get_lines()
        assert vline.get_xdata()[0] == 1.22e-18
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['1.22e-18']


def test_plot_tuning_summary_layout(tmp_path):
    from src import tuning_plots

    rows = [(tuning.load_sweep(tuning.SPECS[name]), metric, tuning.TUNED[pin], None)
            for name, metric, pin in (('dsei_all', 'v_rmse', 'diff_vc'),
                                      ('rho_all', 'e_rmse', 'rho_vc'),
                                      ('kappa_hppc1', 'r_err', 'kappa_vc'))]
    savename = tmp_path / 'summary.svg'
    fig = tuning_plots.plot_tuning_summary(rows, savename=savename)

    assert savename.exists()
    panels = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    assert len(panels) == 9 and len(fig.axes) == 12          # one colorbar per row
    letters = [t.get_text() for ax in panels for t in ax.texts]
    assert letters == [f'({c})' for c in 'abcdefghi']
    assert [ax.get_title() for ax in panels[:3]] == ['Base Formation', 'Fast Formation',
                                                     'Fast+ Formation']
    assert all(ax.get_title() == '' for ax in panels[3:])
    for ax, protocol in zip(panels[3:6], tuning.PROTOCOLS):     # step 3 row: tuned rho_LEDC
        star = [line for line in ax.get_lines() if line.get_marker() == '*'][0]
        assert star.get_ydata()[0] == pytest.approx(tuning.TUNED['rho_ec'][protocol], abs=0.005)


def test_plot_tuning_summary_line_only_rows():
    from src import tuning_plots

    rows = [(tuning.load_sweep(tuning.SPECS['dsei_fc']), 'v_rmse', 1.22e-18, None, False),
            (tuning.load_sweep(tuning.SPECS['rho_fc']), 'e_rmse', 1.5, None, False)]
    fig = tuning_plots.plot_tuning_summary(rows)
    panels = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    assert len(panels) == 6 and len(fig.axes) == 8
    for ax, pin in zip(panels, [1.22e-18] * 3 + [1.5] * 3):
        assert not [line for line in ax.get_lines() if line.get_marker() == '*']
        assert ax.get_lines()[0].get_xdata()[0] == pin
