"""Heatmaps for the SEI parameter sweeps in src/tuning.py."""

import os

import numpy as np
from matplotlib import pyplot as plt

from src import tuning

# Axis label, axis scale, and display multiplier for each swept parameter
PARAM_AXES = {
    'diff_ec': (r'$D_{\mathrm{LEDC}}$ (m$^2$/s)', 'log', 1.0),
    'diff_vc': (r'$D_{\mathrm{LVDC}}$ (m$^2$/s)', 'log', 1.0),
    'rho_ec': (r'$\rho_{\mathrm{LEDC}}$ (g/cm$^3$)', 'linear', 1.0),
    'rho_vc': (r'$\rho_{\mathrm{LVDC}}$ (g/cm$^3$)', 'linear', 1.0),
    'kappa_ec': (r'$\kappa_{\mathrm{LEDC}}$ (mS/m)', 'linear', 1e3),
    'kappa_vc': (r'$\kappa_{\mathrm{LVDC}}$ (mS/m)', 'linear', 1e3),
}

# Colorbar label, display multiplier, and default color limits for each metric
METRIC_STYLE = {
    'v_rmse': ('Voltage RMSE (V)', 1.0, (0.0, 0.25)),
    'e_rmse': (r'Expansion RMSE ($\mu$m)', 1.0, (0.0, 100.0)),
    'r_err': (r'$|R - \hat{R}|$ (m$\Omega$)', 1e3, (0.0, 10.0)),
}

PANEL_TITLES = {'base': 'Base Formation', 'fast': 'Fast Formation', 'fast+': 'Fast+ Formation'}

FIGURE_DIR = 'notebooks_2026/figures'


def plot_heatmap_trio(result, metric, pin_col=None, zlim=None, title=None, savename=None):
    """One heatmap per formation protocol for a sweep metric.

    If pin_col is given, a dotted line marks that column-parameter value and a
    star marks the row value minimizing the metric along it (select_row).
    Returns the figure.
    """
    spec = result.spec
    x_label, x_scale, x_mult = PARAM_AXES[spec.col_param]
    y_label, y_scale, y_mult = PARAM_AXES[spec.row_param]
    z_label, z_mult, z_default = METRIC_STYLE[metric]
    zmin, zmax = zlim if zlim is not None else z_default

    x = np.asarray(result.col_vec) * x_mult
    y = np.asarray(result.row_vec) * y_mult
    X, Y = np.meshgrid(x, y)
    cmap = plt.get_cmap('viridis_r', 20)

    if pin_col is not None:
        selected = tuning.select_row(result, metric, pin_col)
        x_pin = result.col_vec[tuning.nearest_index(result.col_vec, pin_col)] * x_mult

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, protocol in zip(axes, tuning.PROTOCOLS):
        mat = result.metrics[metric][protocol] * z_mult
        pcm = ax.pcolormesh(X, Y, mat, shading='auto', cmap=cmap, vmin=zmin, vmax=zmax)
        fig.colorbar(pcm, ax=ax, label=z_label, orientation='horizontal', pad=0.2)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(PANEL_TITLES[protocol])
        ax.grid(False)

        if pin_col is not None:
            y_sel = selected[protocol] * y_mult
            ax.axvline(x_pin, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.plot(x_pin, y_sel, 'r*', markersize=10, label=f'({x_pin:.3g}, {y_sel:.3g})')
            ax.legend(facecolor='lightgray', framealpha=1.0, frameon=True, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if savename:
        fig.savefig(savename, bbox_inches='tight', format='svg')
    return fig


def plot_sweep_heatmaps(result, pin_col=None, zlims=None, figure_dir=FIGURE_DIR):
    """Heatmaps for every metric of a sweep, saved as {spec.name}_{metric}.svg.

    zlims maps metric -> (zmin, zmax). Returns {metric: figure}.
    """
    zlims = zlims or {}
    os.makedirs(figure_dir, exist_ok=True)
    window = tuning.WINDOW_LABELS[result.spec.window]
    figs = {}
    for metric in result.spec.metrics:
        figs[metric] = plot_heatmap_trio(
            result, metric, pin_col=pin_col, zlim=zlims.get(metric),
            title=f'{window}: {METRIC_STYLE[metric][0]}',
            savename=os.path.join(figure_dir, f'{result.spec.name}_{metric}.svg'),
        )
    return figs
