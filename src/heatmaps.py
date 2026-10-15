"""Heatmap trios: one panel per formation protocol (base, fast, fast+).

Shared by the sweep heatmaps in tuning_plots.py and the older
plot_heatmaps_* functions in simutils.py.
"""

import numpy as np
from matplotlib import pyplot as plt

PANEL_TITLES = ('Base Formation', 'Fast Formation', 'Fast+ Formation')


def heatmap_trio(x_vec, y_vec, mats, z_label, x_label, y_label,
                 zlim=(None, None), xscale='linear', yscale='linear'):
    """Three heatmaps side by side with horizontal colorbars.

    mats holds one matrix per panel, indexed [y, x]. Returns (fig, axes).
    """
    X, Y = np.meshgrid(x_vec, y_vec)
    cmap = plt.get_cmap('viridis_r', 20)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, mat, title in zip(axes, mats, PANEL_TITLES):
        pcm = ax.pcolormesh(X, Y, mat, shading='auto', cmap=cmap, vmin=zlim[0], vmax=zlim[1])
        fig.colorbar(pcm, ax=ax, label=z_label, orientation='horizontal', pad=0.2)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(np.min(x_vec), np.max(x_vec))
        ax.set_ylim(np.min(y_vec), np.max(y_vec))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(False)
    return fig, axes


def annotate_panels(axes, mats, x_vec, y_vec, coords, orientation='vertical',
                    line_only=False, line_color='darkred', label_line=False):
    """Mark a reference line on each panel and a star at the minimum along it.

    coords: one value, or one per panel.
    orientation: 'vertical'   line at x = coord, star at the minimum over y
                 'horizontal' line at y = coord, star at the minimum over x
                 'minimum'    no line, star at the global minimum
    line_only: draw the line but no star.
    label_line: give the line a legend entry (its value) when line_only is set.
    """
    if np.ndim(coords) == 0:
        coords = [coords] * len(axes)
    x_vec, y_vec = np.asarray(x_vec), np.asarray(y_vec)

    for ax, mat, coord in zip(axes, mats, coords):
        line_kw = dict(color=line_color, linestyle=':', linewidth=1.5, alpha=0.7)
        if label_line:
            line_kw['label'] = f'{coord:.3g}' if line_only else None

        if orientation == 'minimum':
            i, j = np.unravel_index(np.nanargmin(mat), mat.shape)
        elif orientation == 'horizontal':
            i = np.argmin(np.abs(y_vec - coord))
            j = np.nanargmin(mat[i, :])
            ax.axhline(y=coord, **line_kw)
        else:
            j = np.argmin(np.abs(x_vec - coord))
            i = np.nanargmin(mat[:, j])
            ax.axvline(x=coord, **line_kw)

        if not line_only:
            ax.plot(x_vec[j], y_vec[i], 'r*', markersize=10,
                    label=f'({x_vec[j]:.3g}, {y_vec[i]:.3g})')


def add_legends(axes):
    for ax in axes:
        ax.legend(facecolor='lightgray', framealpha=1.0, frameon=True, fontsize=10)
