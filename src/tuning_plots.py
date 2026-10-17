"""Heatmaps for the SEI parameter sweeps in src/tuning.py."""

import os

import numpy as np
from matplotlib import pyplot as plt

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
    'v_rmse': ('Voltage RMSE (mV)', 1e3, (0.0, 250.0)),
    'e_rmse': (r'Expansion RMSE ($\mu$m)', 1.0, (0.0, 100.0)),
    'r_err': (r'$|R - \hat{R}|$ (m$\Omega$)', 1e3, (0.0, 10.0)),
}

FIGURE_DIR = 'notebooks_2026/figures'


def plot_heatmap_trio(result, metric, pin_col=None, zlim=None, title=None, savename=None,
                      mark_minimum=True):
    """One heatmap per formation protocol for a sweep metric.

    If pin_col is given, a dotted line marks that column-parameter value. With
    mark_minimum, the line is drawn at the nearest grid column and a star marks
    the row value minimizing the metric along it (select_row); otherwise the
    line is drawn at exactly pin_col and labelled with it.
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
        if mark_minimum:
            x_pin = result.col_vec[tuning.nearest_index(result.col_vec, pin_col)] * x_mult
            heatmaps.annotate_panels(axes, mats, x, y, x_pin)
        else:
            heatmaps.annotate_panels(axes, mats, x, y, pin_col * x_mult,
                                     line_only=True, label_line=True)
        heatmaps.add_legends(axes)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if savename:
        fig.savefig(savename, bbox_inches='tight', format='svg', metadata={'Date': None})
    return fig


def plot_sweep_heatmaps(result, pin_col=None, zlims=None, figure_dir=FIGURE_DIR,
                        mark_minimum=True):
    """Heatmaps for every metric of a sweep, saved as {spec.name}_{metric}.svg.

    zlims maps metric -> (zmin, zmax). See plot_heatmap_trio for mark_minimum.
    Returns {metric: figure}.
    """
    zlims = zlims or {}
    os.makedirs(figure_dir, exist_ok=True)
    window = tuning.WINDOW_LABELS[result.spec.window]
    figs = {}
    for metric in result.spec.metrics:
        figs[metric] = plot_heatmap_trio(
            result, metric, pin_col=pin_col, zlim=zlims.get(metric), mark_minimum=mark_minimum,
            title=f'{window}: {METRIC_STYLE[metric][0]}',
            savename=os.path.join(figure_dir, f'{result.spec.name}_{metric}.svg'),
        )
    return figs


def plot_tuning_summary(rows, savename=None, figsize=None):
    """Composite of several sweeps: one row per sweep, one column per protocol.

    rows: list of (result, metric, pin_col, zlim) or
          (result, metric, pin_col, zlim, mark_minimum). Each row gets one
          colorbar on the right and a dotted line at pin_col. With mark_minimum
          (default True) the line is at the nearest grid column with a star at
          the minimum along it; otherwise it is at exactly pin_col, labelled.
    Panels are labelled (a), (b), ... in reading order and the protocol names
    appear once, above the first row. Returns the figure.
    """
    if figsize is None:
        figsize = (13, 0.4 + 3.7 * len(rows))
    fig, axs = plt.subplots(len(rows), 3, figsize=figsize, layout='constrained')
    fig.get_layout_engine().set(hspace=0.06)    # extra space between rows
    axs = np.atleast_2d(axs)
    letters = iter('abcdefghijklmnopqrstuvwxyz')

    for r, (result, metric, pin_col, zlim, *rest) in enumerate(rows):
        mark_minimum = rest[0] if rest else True
        spec = result.spec
        x_label, x_scale, x_mult = PARAM_AXES[spec.col_param]
        y_label, y_scale, y_mult = PARAM_AXES[spec.row_param]
        z_label, z_mult, z_default = METRIC_STYLE[metric]
        zlim = zlim if zlim is not None else z_default

        x = np.asarray(result.col_vec) * x_mult
        y = np.asarray(result.row_vec) * y_mult
        mats = [result.metrics[metric][p] * z_mult for p in tuning.PROTOCOLS]

        for c, (ax, mat) in enumerate(zip(axs[r], mats)):
            pcm = heatmaps.heatmap_panel(ax, x, y, mat, zlim, x_scale, y_scale,
                                         x_label=x_label, y_label=y_label if c == 0 else None)
            if r == 0:
                ax.set_title(heatmaps.PANEL_TITLES[c])
            ax.text(0.03, 0.96, f'({next(letters)})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=18, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=4))

        if mark_minimum:
            x_pin = result.col_vec[tuning.nearest_index(result.col_vec, pin_col)] * x_mult
            heatmaps.annotate_panels(axs[r], mats, x, y, x_pin)
        else:
            heatmaps.annotate_panels(axs[r], mats, x, y, pin_col * x_mult,
                                     line_only=True, label_line=True)
        heatmaps.add_legends(axs[r], loc='upper right', fontsize=11, facecolor='white',
                             edgecolor='black')
        fig.colorbar(pcm, ax=axs[r, :], location='right', label=z_label, aspect=15)

    if savename:
        fig.savefig(savename, bbox_inches='tight', metadata={'Date': None})
    return fig
