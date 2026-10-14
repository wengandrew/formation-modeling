"""Step 4 of the SEI parameter tuning: the conductivities kappa_LVDC and kappa_LEDC.

With the diffusivities and densities fixed from steps 1-3, sweep
kappa_LEDC x kappa_LVDC through the first RPT and score the absolute error
between the modeled resistance (R_sei + R_OFFSET) and the first HPPC
measurement. kappa_LVDC is set to 5.11e-5 S/m and kappa_LEDC, per protocol, to
the minimum error along it.

Plots the resistance-error heatmaps from the cached sweep. Replaces
2026_05_04_conductivity_sweep. Run from anywhere:

    python notebooks_2026/conductivity_sweep.py [--run kappa_hppc1] [--n-jobs 8] [--show]
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src import tuning, tuning_cli  # noqa: E402

if __name__ == '__main__':
    tuning_cli.main(
        description=__doc__.splitlines()[0],
        spec_names=['kappa_hppc1'],
        pins={'kappa_hppc1': tuning.TUNED['kappa_vc']},
        zlims={'r_err': (0, 10)},
        selections=[('kappa_hppc1', 'r_err', 'kappa_ec')],
    )
