"""Composite figure of the SEI parameter tuning (steps 2-4).

Rows: step 2 voltage RMSE over D_LEDC x D_LVDC (all ~400 h), step 3
expansion RMSE over rho_LEDC x rho_LVDC (all ~400 h), and step 4 first-HPPC
resistance error over kappa_LEDC x kappa_LVDC. Columns: base, fast, fast+.
Built from the cached sweeps; saves figures/tuning_summary.svg and .png.

    python notebooks_2026/tuning_summary.py [--show]
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from matplotlib import pyplot as plt  # noqa: E402

from src import plotter, tuning, tuning_plots  # noqa: E402

# (sweep, metric, pinned column parameter, color limits)
ROWS = [
    ('dsei_all', 'v_rmse', 'diff_vc', (40, 200)),       # step 2 (mV)
    ('rho_all', 'e_rmse', 'rho_vc', (0, 100)),          # step 3
    ('kappa_hppc1', 'r_err', 'kappa_vc', (0, 10)),      # step 4
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--show', action='store_true', help='show the figure')
    args = parser.parse_args(argv)

    plotter.initialize(plt)
    rows = [(tuning.load_sweep(tuning.SPECS[name]), metric, tuning.TUNED[pin], zlim)
            for name, metric, pin, zlim in ROWS]
    os.makedirs(tuning_plots.FIGURE_DIR, exist_ok=True)
    base = os.path.join(tuning_plots.FIGURE_DIR, 'tuning_summary')
    fig = tuning_plots.plot_tuning_summary(rows, savename=base + '.svg')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    print(f'Saved {base}.svg and {base}.png')

    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
