"""Compare the fully tuned model against experiment.

Runs the model for each formation protocol with the tuned SEI parameters
(src/tuning.py: TUNED) through the aging cycles, then plots voltage,
expansion, capacity, and resistance against the measurements and prints the
first-HPPC resistance error. Replaces 2026_05_06_model_vs_experiment.

The simulations take a few minutes. A normal run simulates at full resolution
and plots from those results into figures/ (the published figures). It also
caches the printed numbers and a 1-minute downsampled copy of the runs in
data/cache/ (not committed). --preview re-plots from that cache in seconds,
into figures/preview/ (not committed), so it never touches the published
figures; the numbers it prints are the full-resolution ones.

    python notebooks_2026/model_vs_experiment.py [--preview] [--show]
"""

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib.patches as patches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # noqa: E402

from src import ioutils, paths, plotter, simutils, tuning  # noqa: E402
from src.tuning_plots import FIGURE_DIR  # noqa: E402

COLORS = {'base': 'b', 'fast': 'g', 'fast+': 'r'}
LONG_COLORS = {'base': 'blue', 'fast': 'green', 'fast+': 'red'}
NAMES = {'base': 'Base', 'fast': 'Fast', 'fast+': 'Fast+'}
TITLES = {'base': 'Baseline Formation', 'fast': 'Fast Formation', 'fast+': 'Fast+ Formation'}

# D_SEI multiplier applied once the aging cycles start
DSEI_DERATING_CYCLING = 0.1


LEGEND = dict(fontsize='small', frameon=True, facecolor=(1, 1, 1, 0), edgecolor='black')


def load_experiments():
    return {p: ioutils.fetch_experimental_data(tuning.EXPT_CELLS[p], to_include_cycling=True)[0]
            for p in tuning.PROTOCOLS}


CACHE_FILE = paths.repo_path('notebooks_2026', 'data', 'cache', 'model_vs_experiment.pkl')

# Columns kept in the cache for plotting, and the downsampling step [s]
CACHE_COLUMNS = ['t', 'vt', 'expansion', 'q_sei', 'q_sei2', 'R_sei', 'R_sei2', 'rho_sei',
                 'c_sei1', 'c_sei2', 'D_sei1', 'boost']
CACHE_STEP_S = 60

EXPANSION_REPORT_HOURS = 400


def _run_settings():
    """What the cached runs depend on; a mismatch triggers a re-run."""
    return {'params': {p: tuning.tuned_params(p) for p in tuning.PROTOCOLS},
            'include_flag': 5, 'dt': 1, 'dsei_derating_cycling': DSEI_DERATING_CYCLING,
            'measurements': (tuning.HPPC1, tuning.QD_RPT1)}


def run_model():
    sims = {}
    for p in tuning.PROTOCOLS:
        print(f'Simulating {p} formation...')
        sims[p] = simutils.run_sim(p, **tuning.tuned_params(p), dt=1, include_flag=5,
                                   dsei_derating_cycling=DSEI_DERATING_CYCLING)
    return sims


def summarize(sims, hours=EXPANSION_REPORT_HOURS):
    """Numbers reported from the full-resolution runs, keyed by protocol."""
    t_s = hours * 3600
    summary = {'expansion_um': {}, 'q0_ah': {}, 'r_err_ohm': {}}
    for p in tuning.PROTOCOLS:
        df = sims[p]
        summary['expansion_um'][p] = df.loc[(df['t'] - t_s).abs().idxmin(), 'expansion'] * 1e6
        point = tuning.QD_RPT1[p]
        summary['q0_ah'][p] = point.q_ah + df.loc[(df['t'] - point.t_s).abs().idxmin(), 'q_sei']
        m = tuning.HPPC1[p]
        summary['r_err_ohm'][p] = tuning.resistance_error(df, m.t_s, m.r_ohm)
    return summary


def run_and_cache_model():
    """Full-resolution runs and their summary; also refreshes the preview cache."""
    full = run_model()
    summary = summarize(full)
    preview = {p: df.loc[df.index[::int(CACHE_STEP_S)].union([df.index[-1]]), CACHE_COLUMNS]
               .reset_index(drop=True) for p, df in full.items()}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'settings': _run_settings(), 'sims': preview, 'summary': summary}, f)
    return full, summary


def load_preview():
    """(downsampled sims, summary) from the cache; None if missing or out of date."""
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    if cache['settings'] != _run_settings():
        return None
    return cache['sims'], cache['summary']


def _hours(t):
    return np.asarray(t) / 3600


# Where figures go: FIGURE_DIR for full runs, PREVIEW_DIR for --preview
_output_dir = FIGURE_DIR
PREVIEW_DIR = os.path.join(FIGURE_DIR, 'preview')


def _save(fig, name):
    os.makedirs(_output_dir, exist_ok=True)
    fig.savefig(os.path.join(_output_dir, name), format='svg', dpi=300, metadata={'Date': None})


def plot_experiment_overview(expts, xmax=650, strain_lim=(0, 200)):
    """Measured voltage and expansion for the three protocols."""
    fig_v, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True, sharey=True)
    for ax, p in zip(axs, tuning.PROTOCOLS):
        ax.plot(_hours(expts[p]['arbin_time']), expts[p]['arbin_voltage'],
                color=COLORS[p], label=TITLES[p])
        ax.legend(loc='lower center', **LEGEND)
        ax.grid(False)
    axs[2].set_xlim(-5, xmax)
    axs[2].set_ylim(2.8, 4.25)
    axs[2].set_xlabel('Time (hrs)')
    fig_v.align_ylabels(axs)
    fig_v.tight_layout()
    _save(fig_v, f'experiment_voltage_{xmax}h.svg')

    fig_e, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True, sharey=True)
    for ax, p in zip(axs, tuning.PROTOCOLS):
        ax.plot(_hours(expts[p]['daq_time']), expts[p]['daq_strain'],
                color=COLORS[p], label=TITLES[p])
        ax.grid(False)
    axs[2].set_ylim(strain_lim)
    axs[2].set_xlim(-5, xmax)
    axs[2].set_xlabel('Time (hrs)')
    fig_e.align_ylabels(axs)
    fig_e.tight_layout()
    _save(fig_e, f'experiment_strain_{xmax}h.svg')


def _model_vs_experiment_panels(expts, sims, t_key, y_key, model_y, ylabel, ylim,
                                xmax, xlim_inset, ylim_inset, inset_locs, inset_grid):
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True)
    insets = []
    for ax, p in zip(axs, tuning.PROTOCOLS):
        t_expt, y_expt = _hours(expts[p][t_key]), expts[p][y_key]
        t_model, y_model = _hours(sims[p]['t']), model_y(sims[p])
        ax.plot(t_expt, y_expt, label='Experiment', color='k')
        ax.plot(t_model, y_model, label='Model', color=LONG_COLORS[p], ls='--')
        ax.set_ylabel(ylabel)
        ax.set_title(TITLES[p], loc='left', fontsize='medium')
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), **LEGEND)
        ax.grid(False)

        axins = ax.inset_axes([0.50, 0.15, 0.45, 0.6])
        axins.plot(t_expt, y_expt, color='k')
        axins.plot(t_model, y_model, color=LONG_COLORS[p], ls='--')
        axins.set_xlim(xlim_inset)
        axins.set_ylim(ylim_inset)
        axins.tick_params(axis='both', labelsize='small')
        axins.grid(inset_grid)
        mark_inset(ax, axins, loc1=inset_locs[0], loc2=inset_locs[1], fc='0.8', ec='0.8')
        insets.append(axins)

    if ylim is not None:
        axs[2].set_ylim(ylim)
    axs[2].set_xlim(-5, xmax)
    axs[2].set_xlabel('Time (hrs)')
    fig.align_ylabels(axs)
    fig.tight_layout()
    return fig


def plot_model_vs_experiment(expts, sims, xmax=500, xlim_inset=(0, 10),
                             strain_lim=(0, 160), strain_lim_inset=(0, 130)):
    """Per-protocol voltage and expansion, model vs experiment, with early-time insets."""
    fig = _model_vs_experiment_panels(
        expts, sims, 'arbin_time', 'arbin_voltage', lambda df: df['vt'],
        'Potential (V)', None, xmax, xlim_inset, (3, 4.25), (2, 4), False)
    _save(fig, f'experimental_voltage_{xmax}h.svg')

    fig = _model_vs_experiment_panels(
        expts, sims, 'daq_time', 'daq_strain', lambda df: df['expansion'] * 1e6,
        'Expansion (µm)', strain_lim, xmax, xlim_inset, strain_lim_inset, (2, 3), True)
    _save(fig, f'experimental_strain_{xmax}h.svg')


def print_expansion_at(expts, summary, hours=EXPANSION_REPORT_HOURS):
    """Modeled and measured expansion at a given time."""
    t_s = hours * 3600
    print(f'\nExpansion at {hours} h (model / experiment, µm):')
    for p in tuning.PROTOCOLS:
        model, expt = summary['expansion_um'][p], expts[p]
        i_expt = (expt['daq_time'] - t_s).abs().idxmin()
        meas = expt['daq_strain'].loc[i_expt]
        print(f'  {NAMES[p]:6s} {model:7.2f} / {meas:7.2f}  '
              f'(expt t = {expt["daq_time"].loc[i_expt] / 3600:.2f} h)')


def plot_combined_overview(expts, sims, xmax=650):
    """All protocols on shared axes: voltage and expansion, with zoomed panels."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for p in tuning.PROTOCOLS:
        c, n = COLORS[p], NAMES[p]
        t_model = _hours(sims[p]['t'])
        for ax, label in ((axs[0, 0], True), (axs[0, 1], False)):
            ax.plot(_hours(expts[p]['arbin_time']), expts[p]['arbin_voltage'], color=c,
                    label=f'{n} (meas.)' if label else None)
            ax.plot(t_model, sims[p]['vt'], color=c, ls='--',
                    label=f'{n} (model)' if label else None)
        for ax, label in ((axs[1, 0], True), (axs[1, 1], False)):
            ax.plot(_hours(expts[p]['daq_time']), expts[p]['daq_strain'], color=c,
                    label=f'{n} (meas.)' if label else None)
            ax.plot(t_model, sims[p]['expansion'] * 1e6, color=c, ls='--',
                    label=f'{n} (model)' if label else None)

    axs[0, 0].legend(loc='lower right', ncols=2, **LEGEND)
    axs[0, 0].set_ylabel('Potential (V)')
    axs[0, 0].set_xlim(-5, xmax)
    axs[0, 0].set_ylim(2.0, 4.30)
    axs[0, 0].add_patch(patches.Rectangle((0, 3.0), 50, 4.25 - 3.0, linewidth=2,
                                          edgecolor='k', facecolor='none'))
    axs[0, 1].set_xlim(0, 50)
    axs[0, 1].set_ylim(3.0, 4.25)

    axs[1, 0].set_ylabel(r'Expansion (µm)')
    axs[1, 0].set_xlim(-5, xmax)
    axs[1, 0].set_ylim(0, 350)
    axs[1, 0].legend(loc='upper right', ncols=2, **LEGEND)
    axs[1, 0].set_xlabel('Time (hrs)')
    axs[1, 0].add_patch(patches.Rectangle((0, 5), 50, 160 - 5, linewidth=2,
                                          edgecolor='k', facecolor='none'))
    axs[1, 1].set_xlim(0, 50)
    axs[1, 1].set_ylim(0, 160)
    axs[1, 1].set_xlabel('Time (hrs)')

    for ax in axs.flat:
        ax.grid(False)
    fig.align_ylabels(axs)
    fig.tight_layout()
    _save(fig, 'experimental_vs_simulated_combined_global_dsei.svg')


def print_initial_capacity(summary):
    """Q0 per protocol, chosen so the model capacity matches the first RPT exactly."""
    for p in tuning.PROTOCOLS:
        print(f'Q0 = {summary["q0_ah"][p]:.5f} Ah  ({p})')


def plot_capacity_and_resistance(sims, q0, xmax=650):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    for p in tuning.PROTOCOLS:
        c, n, df = COLORS[p], NAMES[p], sims[p]
        axs[0].plot(_hours(df['t']), q0[p] - df['q_sei'], color=c, ls='--', label=f'{n} (model)')
        axs[1].plot(_hours(df['t']), df['R_sei'] * 1000 + tuning.R_OFFSET * 1000, color=c, ls='--',
                    label=rf'{n} (model $R_{{\mathrm{{sei}}}} + R_{{\mathrm{{ct}}}}^+)$')

    for p in tuning.PROTOCOLS:
        c, n = COLORS[p], NAMES[p]
        for k, point in enumerate((tuning.QD_RPT1[p], tuning.QD_RPT2[p])):
            axs[0].plot(_hours(point.t_s), point.q_ah, c + 'o', markersize=8,
                        label=f'{n} (meas.)' if k == 0 else '')
            axs[0].errorbar(_hours(point.t_s), point.q_ah, yerr=point.q_std,
                            fmt='none', ecolor=c, capsize=3)
        for k, m in enumerate((tuning.HPPC1[p], tuning.HPPC2[p])):
            axs[1].plot(_hours(m.t_s), m.r_ohm * 1000, c + 'o', markersize=8,
                        label=rf'{n} (measured $R_s$)' if k == 0 else '')
            axs[1].errorbar(_hours(m.t_s), m.r_ohm * 1000, yerr=m.r_std * 1000,
                            fmt='none', ecolor=c, capsize=3)

    axs[0].set_ylabel('Cell Capacity (Ah)')
    axs[0].legend(loc='upper right', ncols=2, **LEGEND)
    axs[0].set_ylim(2.3, 2.95)
    axs[1].set_xlim(-5, xmax)
    axs[1].legend(loc='upper left', ncols=2, fontsize='small', frameon=False)
    axs[1].set_ylabel(r'Cell Resistance (m$\Omega$)')
    axs[1].set_ylim(10, 45)
    axs[1].set_xlabel('Time (hrs)')
    for ax in axs:
        ax.grid(False)
    fig.align_ylabels(axs)
    fig.tight_layout()
    _save(fig, 'experimental_vs_simulated_resistance.svg')


def print_resistance_errors(summary):
    print('\nFirst HPPC resistance error |R_model - R_meas|:')
    for p in tuning.PROTOCOLS:
        print(f'  {NAMES[p]:6s} {summary["r_err_ohm"][p] * 1e3:.2f} mOhm')


def plot_model_outputs(sims, xmax=80):
    """Internal model states: SEI capacity, resistance, density, composition, solvents, D_SEI."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    legend = dict(fontsize='small', frameon=False)
    t = {p: _hours(sims[p]['t']) for p in tuning.PROTOCOLS}

    for p in tuning.PROTOCOLS:
        df, c, n = sims[p], COLORS[p], NAMES[p]
        axs[0, 0].plot(t[p], df['q_sei'], color=c, label=n)
        axs[1, 0].plot(t[p], df['R_sei'] * 1e3, color=c, label=n)
        axs[0, 1].plot(t[p], df['rho_sei'], color=c, label=n)
        axs[1, 1].plot(t[p], df['q_sei2'] / df['q_sei'] * 100, color=c, label=n)
        axs[0, 2].plot(t[p], df['c_sei1'] * 1e-3, color=c, label=rf'$c_{{\mathrm{{EC}}}}$ {n}')
        axs[1, 2].plot(t[p], df['D_sei1'], color=c, label=n)
    for p in tuning.PROTOCOLS:
        df, c = sims[p], COLORS[p]
        axs[0, 0].plot(t[p], df['q_sei2'], color=c, ls='--', label='LVDC Only')
        axs[1, 0].plot(t[p], df['R_sei2'] * 1e3, color=c, ls='--', label='LVDC Only')
        axs[1, 2].plot(t[p], df['D_sei1'] * (1 + df['boost']), color=c, ls='--', linewidth=1,
                       label=f'{NAMES[p]} (boosted)')

    axs[0, 0].set_ylabel(r'$Q_{\mathrm{sei}}$ (Ah)')
    axs[0, 0].set_ylim(0, 0.7)
    axs[0, 0].legend(loc='upper center', ncols=2, **legend)

    axs[1, 0].set_ylabel(r'$R_{\mathrm{sei}}$ (m$\Omega$)')
    axs[1, 0].set_ylim(0, 18)
    axs[1, 0].legend(loc='upper center', ncols=2, **legend)

    base = tuning.tuned_params('base')
    axs[0, 1].axhline(base['rho_ec'], color='c', linestyle='--', label='LEDC')
    axs[0, 1].axhline(base['rho_vc'], color='m', linestyle='--', label='LVDC')
    axs[0, 1].set_ylabel(r'$\rho_{\mathrm{sei}}$ (g/cm$^3$)')
    axs[0, 1].set_ylim(1.25, 2.00)
    axs[0, 1].legend(loc='center right', **legend)

    axs[1, 1].set_ylabel(r'% LVDC (Ah/Ah)')
    axs[1, 1].legend(loc='upper right', **legend)

    axs[0, 2].set_ylabel(r'$c_{\mathrm{EC}}$ (kmol/m$^3$)')
    axs[0, 2].set_ylim(3.0, 4.5)
    ax_vc = axs[0, 2].twinx()
    for p in tuning.PROTOCOLS:
        ax_vc.plot(t[p], sims[p]['c_sei2'] * 1e-3, color=COLORS[p], ls='--',
                   label=rf'$c_{{\mathrm{{VC}}}}$ {NAMES[p]}')
    ax_vc.set_ylabel(r'$c_{\mathrm{VC}}$ (kmol/m$^3$)')
    ax_vc.set_ylim(0, 0.6)
    ax_vc.grid(False)
    lines1, labels1 = axs[0, 2].get_legend_handles_labels()
    lines2, labels2 = ax_vc.get_legend_handles_labels()
    axs[0, 2].legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncols=2, **legend)

    axs[1, 2].set_yscale('log')
    axs[1, 2].axhline(base['diff_ec'], color='c', linestyle='--')
    axs[1, 2].axhline(base['diff_vc'], color='m', linestyle='--')
    axs[1, 2].set_ylabel(r'$D_{\mathrm{sei}}$ (m$^2$/s)')
    axs[1, 2].legend(loc='upper right', ncols=2, **legend)

    for ax in axs.flat:
        ax.set_xlabel('Time (hrs)')
        ax.set_xlim(-5, xmax)
        ax.grid(False)
    fig.align_ylabels(axs)
    fig.tight_layout()
    _save(fig, f'model_outputs_{xmax}h.svg')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--preview', action='store_true',
                        help='re-plot quickly from the cached, downsampled runs into figures/preview/')
    parser.add_argument('--show', action='store_true', help='show the figures')
    args = parser.parse_args(argv)

    global _output_dir
    plotter.initialize(plt)
    expts = load_experiments()
    if args.preview:
        cached = load_preview()
        if cached is None:
            raise SystemExit('No up-to-date cache; run once without --preview first.')
        sims, summary = cached
        _output_dir = PREVIEW_DIR
        print(f'Preview from cached, downsampled runs ({paths.relative(CACHE_FILE)})')
    else:
        sims, summary = run_and_cache_model()

    plot_experiment_overview(expts)
    plot_model_vs_experiment(expts, sims)
    print_expansion_at(expts, summary)
    plot_combined_overview(expts, sims)
    print_initial_capacity(summary)
    plot_capacity_and_resistance(sims, summary['q0_ah'])
    print_resistance_errors(summary)
    plot_model_outputs(sims)
    print(f'\nFigures saved to {paths.relative(_output_dir)}/')

    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
