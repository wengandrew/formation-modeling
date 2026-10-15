"""Heatmaps for the SEI parameter sweeps in src/tuning.py."""

import os

import numpy as np

from src import heatmaps, tuning

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

    x = np.asarray(result.col_vec) * x_mult
    y = np.asarray(result.row_vec) * y_mult
    mats = [result.metrics[metric][p] * z_mult for p in tuning.PROTOCOLS]
    fig, axes = heatmaps.heatmap_trio(x, y, mats, z_label, x_label, y_label,
                                      zlim=zlim if zlim is not None else z_default,
                                      xscale=x_scale, yscale=y_scale)

    if pin_col is not None:
        x_pin = result.col_vec[tuning.nearest_index(result.col_vec, pin_col)] * x_mult
        heatmaps.annotate_panels(axes, mats, x, y, x_pin)
        heatmaps.add_legends(axes)

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
