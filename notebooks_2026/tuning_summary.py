"""Composite figures of the SEI parameter tuning.

tuning_summary: step 2 voltage RMSE over D_LEDC x D_LVDC and step 3 expansion
    RMSE over rho_LEDC x rho_LVDC (both through formation aging), and step 4
    first-HPPC resistance error over kappa_LEDC x kappa_LVDC.
tuning_summary_first_charge / tuning_summary_full_window: voltage and
    expansion RMSE for the D and rho sweeps, over the first charge or through
    formation aging (~500 h of data). Stars mark the tuned value only on the
    rows used to set it; the other rows mark just the pinned value.

Columns: base, fast, fast+. Built from the cached sweeps; saves
figures/<name>.svg and .png.

    python notebooks_2026/tuning_summary.py [--show]
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from matplotlib import pyplot as plt  # noqa: E402

from src import paths, plotter, tuning, tuning_plots  # noqa: E402

# name: [(sweep, metric, pinned column parameter, color limits, mark minimum)]
FIGURES = {
    'tuning_summary': [
        ('dsei_all', 'v_rmse', 'diff_vc', (40, 200), True),       # step 2 (mV)
        ('rho_all', 'e_rmse', 'rho_vc', (0, 100), True),          # step 3 (um)
        ('kappa_hppc1', 'r_err', 'kappa_vc', (0, 10), True),      # step 4 (mOhm)
    ],
    'tuning_summary_first_charge': [
        ('dsei_fc', 'v_rmse', 'diff_vc', (40, 200), False),       # step 1 (mV)
        ('dsei_fc', 'e_rmse', 'diff_vc', (0, 100), False),        # um
        ('rho_fc', 'v_rmse', 'rho_vc', (40, 200), False),         # mV
        ('rho_fc', 'e_rmse', 'rho_vc', (0, 40), False),           # um
    ],
    'tuning_summary_full_window': [
        ('dsei_all', 'v_rmse', 'diff_vc', (40, 200), True),       # step 2 (mV)
        ('dsei_all', 'e_rmse', 'diff_vc', (0, 100), False),       # um
        ('rho_all', 'v_rmse', 'rho_vc', (40, 200), False),        # mV
        ('rho_all', 'e_rmse', 'rho_vc', (0, 100), True),          # step 3 (um)
    ],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--show', action='store_true', help='show the figures')
    args = parser.parse_args(argv)

    plotter.initialize(plt)
    os.makedirs(tuning_plots.FIGURE_DIR, exist_ok=True)
    for name, spec_rows in FIGURES.items():
        rows = [(tuning.load_sweep(tuning.SPECS[sweep]), metric, tuning.TUNED[pin], zlim, mark)
                for sweep, metric, pin, zlim, mark in spec_rows]
        base = os.path.join(tuning_plots.FIGURE_DIR, name)
        fig = tuning_plots.plot_tuning_summary(rows, savename=base + '.svg')
        fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
        print(f'Saved {paths.relative(base)}.svg and .png')

    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
