"""Compare the fully tuned model against experiment.

Runs the model for each formation protocol with its tuned SEI parameters
(src/tuning.py: TUNED) through the aging cycles, and again with the baseline
protocol's parameters for comparison. Plots voltage, expansion, capacity, and
resistance against the measurements and prints the first-HPPC resistance error.
Replaces 2026_05_06_model_vs_experiment.

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
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # noqa: E402

from src import hppc, ioutils, paths, plotter, simutils, tuning  # noqa: E402
from src.tuning_plots import FIGURE_DIR  # noqa: E402

COLORS = {'base': 'b', 'fast': 'g', 'fast+': 'r'}
LONG_COLORS = {'base': 'blue', 'fast': 'green', 'fast+': 'red'}
NAMES = {'base': 'Base', 'fast': 'Fast', 'fast+': 'Fast+'}
TITLES = {'base': 'Baseline Formation', 'fast': 'Fast Formation', 'fast+': 'Fast+ Formation'}

# D_SEI multiplier applied once the aging cycles start
DSEI_DERATING_CYCLING = 0.1


LEGEND = dict(fontsize='small', frameon=True, facecolor=(1, 1, 1, 0), edgecolor='black')


def load_data():
    """(experiments, HPPC pulses), keyed by protocol.

    HPPC pulses: {1: (SOC %, R Ohm), 2: (...)} for the HPPC after formation (1)
    and after formation aging (2), from the cycler data (src/hppc.py).
    """
    expts, pulses = {}, {}
    for p in tuning.PROTOCOLS:
        cellid = tuning.EXPT_CELLS[p]
        expts[p], df_arbin = ioutils.fetch_experimental_data(cellid, to_include_cycling=True)
        pulses[p] = {n: hppc.hppc_resistance(df_arbin, cellid, n) for n in (1, 2)}
    return expts, pulses


def load_experiments():
    return load_data()[0]


CACHE_FILE = paths.repo_path('notebooks_2026', 'data', 'cache', 'model_vs_experiment.pkl')

# Columns kept in the cache for plotting, and the downsampling step [s]
CACHE_COLUMNS = ['t', 'vt', 'expansion', 'q_sei', 'q_sei2', 'R_sei', 'R_sei2', 'rho_sei',
                 'c_sei1', 'c_sei2', 'D_sei1', 'boost']
CACHE_STEP_S = 60

EXPANSION_REPORT_HOURS = 400


def _run_settings():
    """What the cached runs depend on; a mismatch triggers a re-run."""
    return {'version': 3,
            'params': {p: tuning.tuned_params(p) for p in tuning.PROTOCOLS},
            'include_flag': 5, 'dt': 1, 'dsei_derating_cycling': DSEI_DERATING_CYCLING,
            'measurements': (tuning.HPPC1, tuning.QD_RPT1)}


def run_model():
    sims = {}
    for p in tuning.PROTOCOLS:
        print(f'Simulating {p} formation...')
        sims[p] = simutils.run_sim(p, **tuning.tuned_params(p), dt=1, include_flag=5,
                                   dsei_derating_cycling=DSEI_DERATING_CYCLING)
    return sims


def run_model_with_baseline_params(sims):
    """Each protocol simulated with the baseline protocol's tuned parameters.

    For the baseline protocol this is the same run as in sims, so it is reused.
    """
    base_params = tuning.tuned_params('base')
    out = {}
    for p in tuning.PROTOCOLS:
        if tuning.tuned_params(p) == base_params:
            out[p] = sims[p]
            continue
        print(f'Simulating {p} formation with baseline parameters...')
        out[p] = simutils.run_sim(p, **base_params, dt=1, include_flag=5,
                                  dsei_derating_cycling=DSEI_DERATING_CYCLING)
    return out


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
    # One Q0 for all protocols: the least-squares fit to the three RPT 1 points
    summary['q0_shared_ah'] = float(np.mean(list(summary['q0_ah'].values())))
    return summary


def _downsample(sims):
    return {p: df.loc[df.index[::int(CACHE_STEP_S)].union([df.index[-1]]), CACHE_COLUMNS]
            .reset_index(drop=True) for p, df in sims.items()}


def run_and_cache_model():
    """Full-resolution runs (protocol and baseline parameters) and their summary.

    Also refreshes the preview cache.
    """
    full = run_model()
    full_base = run_model_with_baseline_params(full)
    summary = summarize(full)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'settings': _run_settings(), 'sims': _downsample(full),
                     'sims_base_params': _downsample(full_base), 'summary': summary}, f)
    return full, full_base, summary


def load_preview():
    """(downsampled sims, summary) from the cache; None if missing or out of date."""
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    if cache['settings'] != _run_settings():
        return None
    return cache['sims'], cache['sims_base_params'], cache['summary']


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


# Style of the model run with the baseline protocol's parameters
BASE_PARAMS_STYLE = dict(color='0.45', ls='-.', zorder=1.9)   # drawn under the protocol-param model


def _panel_settings(strain_lim_inset=(0, 130)):
    """(voltage, expansion) settings for _comparison_panel."""
    voltage = dict(t_key='arbin_time', y_key='arbin_voltage', model_y=lambda df: df['vt'],
                   ylabel='Potential (V)', ylim_inset=(3, 4.25), inset_locs=(2, 4),
                   inset_grid=True)
    expansion = dict(t_key='daq_time', y_key='daq_strain',
                     model_y=lambda df: df['expansion'] * 1e6, ylabel='Expansion (µm)',
                     ylim_inset=strain_lim_inset, inset_locs=(2, 3), inset_grid=True)
    return voltage, expansion


def _comparison_panel(ax, expt, sim, sim_base, protocol, t_key, y_key, model_y, ylabel,
                      xlim_inset, ylim_inset, inset_locs, inset_grid, legend=True):
    """Experiment, protocol-parameter model and baseline-parameter model on one
    panel, with an inset zoomed on xlim_inset."""
    t_expt, y_expt = _hours(expt[t_key]), expt[y_key]
    t_model, y_model = _hours(sim['t']), model_y(sim)
    t_base, y_base = _hours(sim_base['t']), model_y(sim_base)
    ax.plot(t_expt, y_expt, label='Experiment', color='k')
    ax.plot(t_model, y_model, label='Model (protocol params)', color=LONG_COLORS[protocol], ls='--')
    ax.plot(t_base, y_base, label='Model (baseline params)', **BASE_PARAMS_STYLE)
    ax.set_ylabel(ylabel)
    ax.set_title(TITLES[protocol], loc='left', fontsize='medium')
    if legend:
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), **LEGEND)
    ax.grid(False)

    axins = ax.inset_axes([0.50, 0.15, 0.45, 0.6])
    axins.plot(t_expt, y_expt, color='k')
    axins.plot(t_model, y_model, color=LONG_COLORS[protocol], ls='--')
    axins.plot(t_base, y_base, **BASE_PARAMS_STYLE)
    axins.set_xlim(xlim_inset)
    axins.set_ylim(ylim_inset)
    axins.tick_params(axis='both', labelsize='small')
    axins.grid(inset_grid)
    mark_inset(ax, axins, loc1=inset_locs[0], loc2=inset_locs[1], fc='0.8', ec='0.8')


def _model_vs_experiment_panels(expts, sims, sims_base, panel, ylim, xmax, xlim_inset):
    """One column of panels (base, fast, fast+) for one quantity."""
    fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True)
    for ax, p in zip(axs, tuning.PROTOCOLS):
        _comparison_panel(ax, expts[p], sims[p], sims_base[p], p, xlim_inset=xlim_inset, **panel)

    if ylim is not None:
        axs[2].set_ylim(ylim)
    axs[2].set_xlim(-5, xmax)
    axs[2].set_xlabel('Time (hrs)')
    fig.align_ylabels(axs)
    fig.tight_layout()
    return fig


def plot_model_vs_experiment(expts, sims, sims_base, xmax=500, xlim_inset=(0, 10),
                             strain_lim=(0, 160), strain_lim_inset=(0, 130)):
    """Per-protocol voltage and expansion, with early-time insets.

    Each panel shows the experiment, the model with that protocol's tuned
    parameters (sims), and the model with the baseline protocol's parameters
    (sims_base; identical to sims for the baseline protocol).
    """
    voltage, expansion = _panel_settings(strain_lim_inset)
    fig = _model_vs_experiment_panels(expts, sims, sims_base, voltage, None, xmax, xlim_inset)
    _save(fig, f'experimental_voltage_{xmax}h.svg')

    fig = _model_vs_experiment_panels(expts, sims, sims_base, expansion, strain_lim, xmax,
                                      xlim_inset)
    _save(fig, f'experimental_strain_{xmax}h.svg')


def plot_voltage_and_expansion(expts, sims, sims_base, xmax=500, xlim_inset=(0, 10),
                               strain_lim=(0, 160), strain_lim_inset=(0, 130)):
    """The voltage (left) and expansion (right) panels in one figure.

    Rows are the base, fast and fast+ protocols; panels are labelled (a)-(f)
    in reading order. One legend above the panels (the protocol-parameter
    model is drawn in each row's color). Saved as SVG, plus a PNG for LaTeX.
    """
    voltage, expansion = _panel_settings(strain_lim_inset)
    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    letters = iter('abcdef')
    for r, p in enumerate(tuning.PROTOCOLS):
        for c, (panel, ylim) in enumerate(((voltage, None), (expansion, strain_lim))):
            ax = axs[r, c]
            if r > 0:
                ax.sharey(axs[0, c])
            _comparison_panel(ax, expts[p], sims[p], sims_base[p], p, xlim_inset=xlim_inset,
                              legend=False, **panel)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.text(-0.17, 1.04, f'({next(letters)})', transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='bottom')

    axs[0, 0].set_xlim(-5, xmax)
    for ax in axs[-1]:
        ax.set_xlabel('Time (hrs)')
    for c in range(2):
        fig.align_ylabels(axs[:, c])
    handles = [Line2D([], [], color='k', label='Experiment'),
               Line2D([], [], color='k', ls='--', label='Model (protocol params)'),
               Line2D([], [], label='Model (baseline params)', **BASE_PARAMS_STYLE)]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=handles, loc='upper center', ncols=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    name = 'model_vs_experiment_voltage_expansion'
    _save(fig, f'{name}.svg')
    fig.savefig(os.path.join(_output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')


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
    """Q0 per protocol, chosen so the model capacity matches the first RPT exactly,
    and the single Q0 fitted to all three RPT 1 points."""
    for p in tuning.PROTOCOLS:
        print(f'Q0 = {summary["q0_ah"][p]:.5f} Ah  ({p})')
    print(f'Q0 = {summary["q0_shared_ah"]:.5f} Ah  (shared, fitted to all RPT 1 points)')


def _capacity_panel(ax, sims, q0, legend=True):
    """Modeled capacity Q0 - Q_sei and the measured RPT capacities."""
    for p in tuning.PROTOCOLS:
        c, n, df = COLORS[p], NAMES[p], sims[p]
        ax.plot(_hours(df['t']), q0[p] - df['q_sei'], color=c, ls='--', label=f'{n} (model)')
    for p in tuning.PROTOCOLS:
        c, n = COLORS[p], NAMES[p]
        for k, point in enumerate((tuning.QD_RPT1[p], tuning.QD_RPT2[p])):
            ax.plot(_hours(point.t_s), point.q_ah, c + 'o', markersize=8,
                    label=f'{n} (meas.)' if k == 0 else '')
            ax.errorbar(_hours(point.t_s), point.q_ah, yerr=point.q_std,
                        fmt='none', ecolor=c, capsize=3)
    ax.set_ylabel('Cell Capacity (Ah)')
    if legend:
        ax.legend(loc='upper right', ncols=2, **LEGEND)
    ax.set_ylim(2.3, 2.95)


def _resistance_panel(ax, sims, legend=True):
    """Modeled R_sei + R_ct+ and the measured HPPC series resistances."""
    for p in tuning.PROTOCOLS:
        c, n, df = COLORS[p], NAMES[p], sims[p]
        ax.plot(_hours(df['t']), df['R_sei'] * 1000 + tuning.R_OFFSET * 1000, color=c, ls='--',
                label=rf'{n} (model $R_{{\mathrm{{sei}}}} + R_{{\mathrm{{ct}}}}^+)$')
    for p in tuning.PROTOCOLS:
        c, n = COLORS[p], NAMES[p]
        for k, m in enumerate((tuning.HPPC1[p], tuning.HPPC2[p])):
            ax.plot(_hours(m.t_s), m.r_ohm * 1000, c + 'o', markersize=8,
                    label=rf'{n} (measured $R_s$)' if k == 0 else '')
            ax.errorbar(_hours(m.t_s), m.r_ohm * 1000, yerr=m.r_std * 1000,
                        fmt='none', ecolor=c, capsize=3)
    if legend:
        ax.legend(loc='upper left', ncols=2, fontsize='small', frameon=False)
    ax.set_ylabel(r'Cell Resistance (m$\Omega$)')
    ax.set_ylim(10, 45)


def plot_capacity_and_resistance(sims, q0, xmax=650):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    _capacity_panel(axs[0], sims, q0)
    _resistance_panel(axs[1], sims)
    axs[1].set_xlim(-5, xmax)
    axs[1].set_xlabel('Time (hrs)')
    for ax in axs:
        ax.grid(False)
    fig.align_ylabels(axs)
    fig.tight_layout()
    _save(fig, 'experimental_vs_simulated_resistance.svg')


# Vertical offsets [V] stacking the voltage traces in the capacity/resistance summary
VOLTAGE_STACK_OFFSET = {'base': 3.2, 'fast': 1.6, 'fast+': 0.0}
ANNOTATION_GRAY = '0.35'


def _panel_label(ax, letter, x=-0.07):
    ax.text(x, 1.02, f'({letter})', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='bottom', ha='right')


def _gap_arrow(ax, x, y_low, y_high, text, text_dx):
    """Double-headed gray arrow from y_low to y_high at x, labelled on the right."""
    ax.annotate('', xy=(x, y_high), xytext=(x, y_low),
                arrowprops=dict(arrowstyle='<->', color=ANNOTATION_GRAY, lw=1.5))
    ax.text(x + text_dx, (y_low + y_high) / 2, text, color=ANNOTATION_GRAY,
            va='center', ha='left', fontsize='medium')


def plot_capacity_resistance_summary(expts, pulses, sims, q0, xmax=650):
    """Time-resolved capacities and resistances, measured versus modeled.

    (a) measured voltage, stacked by protocol, marking the HPPC and RPT checks;
    (b) capacity, modeled as q0 - Q_sei with one q0 for all protocols;
    (c) resistance, modeled as R_sei + R_ct+, vs the median HPPC series
    resistance; (d, e) per-pulse series resistance vs SOC for HPPC 1 and 2,
    with the modeled resistance at that HPPC. Saved as SVG and PNG.
    """
    fig = plt.figure(figsize=(11, 13.5), layout='constrained')
    grid = fig.add_gridspec(4, 2, height_ratios=[1.25, 1, 1, 1.15])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :], sharex=ax_a)
    ax_c = fig.add_subplot(grid[2, :], sharex=ax_a)
    ax_d = fig.add_subplot(grid[3, 0])
    ax_e = fig.add_subplot(grid[3, 1], sharey=ax_d)
    q0_all = {p: q0 for p in tuning.PROTOCOLS}

    # (a) measured voltage, stacked
    for p in tuning.PROTOCOLS:
        offset = VOLTAGE_STACK_OFFSET[p]
        ax_a.plot(_hours(expts[p]['arbin_time']), expts[p]['arbin_voltage'] + offset,
                  color=COLORS[p], lw=1.5)
        ax_a.text(330, offset + 3.8, TITLES[p], ha='center', va='center', fontsize='medium')
    ax_a.set_ylabel('Potential (V)')
    ax_a.set_yticks([])
    ax_a.set_ylim(2.6, 9.0)
    y_tip = VOLTAGE_STACK_OFFSET['base'] + 4.35
    for label, t_s, side in (('HPPC 1', tuning.HPPC1['base'].t_s, -1),
                             ('RPT 1', tuning.QD_RPT1['base'].t_s, +1),
                             ('HPPC 2', tuning.HPPC2['base'].t_s, -1),
                             ('RPT 2', tuning.QD_RPT2['base'].t_s, +1)):
        t_h = _hours(t_s)
        ax_a.annotate(label, xy=(t_h, y_tip), xytext=(t_h + side * 12, y_tip + 0.9),
                      ha='right' if side < 0 else 'left', va='bottom', fontsize='medium',
                      arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))

    # (b) capacity with one Q0
    _capacity_panel(ax_b, sims, q0_all)
    ax_b.text(_hours(tuning.QD_RPT1['fast'].t_s) + 5, 2.63, 'RPT 1', fontsize='medium')
    ax_b.text(_hours(tuning.QD_RPT2['base'].t_s) + 8, 2.52, 'RPT 2', fontsize='medium')

    # (c) resistance
    _resistance_panel(ax_c, sims)
    ax_c.text(_hours(tuning.HPPC1['base'].t_s) - 55, 27, 'HPPC 1', fontsize='medium')
    t2 = tuning.HPPC2['base'].t_s
    ax_c.text(_hours(t2) + 12, 37, 'HPPC 2', fontsize='medium', va='center')
    model_r2 = max(tuning.model_resistance(sims[p], tuning.HPPC2[p].t_s) for p in tuning.PROTOCOLS)
    meas_r2 = min(tuning.HPPC2[p].r_ohm - tuning.HPPC2[p].r_std for p in tuning.PROTOCOLS)
    _gap_arrow(ax_c, _hours(t2), model_r2 * 1e3 + 1.0, meas_r2 * 1e3 - 0.5, 'Cathode?', 12)
    ax_c.set_xlim(-5, xmax)
    ax_c.set_xlabel('Time (hrs)')
    plt.setp(ax_a.get_xticklabels() + ax_b.get_xticklabels(), visible=False)

    # (d, e) per-pulse resistance vs SOC, with the modeled value at that HPPC
    for ax, n, measurements in ((ax_d, 1, tuning.HPPC1), (ax_e, 2, tuning.HPPC2)):
        for p in tuning.PROTOCOLS:
            soc, resistance = pulses[p][n]
            ax.plot(soc, resistance * 1e3, 'o', color=COLORS[p], markersize=3,
                    label=f'{NAMES[p]} (meas.)')
        for p in tuning.PROTOCOLS:
            r_model = tuning.model_resistance(sims[p], measurements[p].t_s) * 1e3
            ax.axhline(r_model, color=COLORS[p], ls='--', lw=1, label=f'{NAMES[p]} (modeled)')
        ax.text(0.05, 0.62 if n == 1 else 0.38, f'HPPC {n}', transform=ax.transAxes,
                fontsize='large')
        ax.set_xlabel('SOC (%)')
        ax.set_ylim(15, 55)
        ax.legend(loc='upper right', ncols=2, fontsize='x-small', frameon=False)
    ax_d.set_ylabel(r'Resistance (m$\Omega$)')
    soc_mid = 57
    model_e = max(tuning.model_resistance(sims[p], tuning.HPPC2[p].t_s) for p in tuning.PROTOCOLS)
    meas_e = min(np.interp(soc_mid, *pulses[p][2]) for p in tuning.PROTOCOLS)
    _gap_arrow(ax_e, soc_mid, model_e * 1e3 + 1.0, meas_e * 1e3 - 1.0, 'Cathode?', 3)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.grid(False)
    for ax, letter, x in ((ax_a, 'a', -0.07), (ax_b, 'b', -0.07), (ax_c, 'c', -0.07),
                          (ax_d, 'd', -0.14), (ax_e, 'e', -0.14)):
        _panel_label(ax, letter, x)
    fig.align_ylabels([ax_a, ax_b, ax_c])

    name = 'capacity_and_resistance_summary'
    _save(fig, f'{name}.svg')
    fig.savefig(os.path.join(_output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')


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
    expts, pulses = load_data()
    if args.preview:
        cached = load_preview()
        if cached is None:
            raise SystemExit('No up-to-date cache; run once without --preview first.')
        sims, sims_base, summary = cached
        _output_dir = PREVIEW_DIR
        print(f'Preview from cached, downsampled runs ({paths.relative(CACHE_FILE)})')
    else:
        sims, sims_base, summary = run_and_cache_model()

    plot_experiment_overview(expts)
    plot_model_vs_experiment(expts, sims, sims_base)
    plot_voltage_and_expansion(expts, sims, sims_base)
    print_expansion_at(expts, summary)
    plot_combined_overview(expts, sims)
    print_initial_capacity(summary)
    plot_capacity_and_resistance(sims, summary['q0_ah'])
    plot_capacity_resistance_summary(expts, pulses, sims, summary['q0_shared_ah'])
    print_resistance_errors(summary)
    plot_model_outputs(sims)
    print(f'\nFigures saved to {paths.relative(_output_dir)}/')

    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
