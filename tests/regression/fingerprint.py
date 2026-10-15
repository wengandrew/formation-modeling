"""Deterministic fingerprints of simulation results and matplotlib figures.

Used to check that refactors of run_sim() and of the plotting code leave
results and figures unchanged. References are recorded from the code before a
refactor by make_references.py and compared in test_regression.py.
"""

import hashlib

import numpy as np
from matplotlib import colors as mcolors


def _hash_array(values):
    arr = np.ascontiguousarray(np.asarray(values, dtype=float))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def dataframe_fingerprint(df):
    """{column: hash} for every numeric column, plus the row count."""
    out = {'__rows__': int(len(df))}
    for col in df.columns:
        try:
            out[col] = _hash_array(df[col].to_numpy())
        except (TypeError, ValueError):
            continue   # object columns (e.g. the Cell instance)
    return out


def _round(x):
    return [None if v is None else float(np.round(v, 10)) for v in x]


def _color(c):
    return [round(v, 4) for v in mcolors.to_rgba(c)]


def _axes_fingerprint(ax):
    legend = ax.get_legend()
    return {
        'title': ax.get_title(),
        'xlabel': ax.get_xlabel(),
        'ylabel': ax.get_ylabel(),
        'xscale': ax.get_xscale(),
        'yscale': ax.get_yscale(),
        'xlim': _round(ax.get_xlim()),
        'ylim': _round(ax.get_ylim()),
        'lines': [
            {
                'label': line.get_label(),
                'color': _color(line.get_color()),
                'ls': line.get_linestyle(),
                'lw': float(line.get_linewidth()),
                'marker': str(line.get_marker()),
                'x': _hash_array(line.get_xdata()),
                'y': _hash_array(line.get_ydata()),
            }
            for line in ax.get_lines()
        ],
        'texts': [
            {
                'text': t.get_text(),
                'pos': _round(t.get_position()),
                'size': float(t.get_fontsize()),
                'color': _color(t.get_color()),
            }
            for t in ax.texts
        ],
        'legend': [t.get_text() for t in legend.get_texts()] if legend else None,
        'meshes': [
            {
                'array': _hash_array(np.ma.filled(c.get_array(), np.nan)),
                'clim': _round(c.get_clim()),
                'cmap': c.get_cmap().name,
            }
            for c in ax.collections if hasattr(c, 'get_clim')
        ],
        'n_patches': len(ax.patches),
    }


def figure_fingerprint(fig):
    return {
        'size': _round(fig.get_size_inches()),
        'axes': [_axes_fingerprint(ax) for ax in fig.axes],
    }
