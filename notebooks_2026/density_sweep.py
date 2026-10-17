"""Step 3 of the SEI parameter tuning: the densities rho_LVDC and rho_LEDC.

With D_LVDC and D_LEDC fixed from steps 1-2, sweep rho_LEDC x rho_LVDC over
the first charge (rho_fc) and through formation aging (~500 h of data, rho_all).
rho_LVDC is set to 1.5 g/cm^3 and rho_LEDC, per protocol, to the minimum
expansion RMSE through formation aging along it.

Plots voltage and expansion RMSE heatmaps for both windows from the cached
sweeps. Replaces 2026_04_29_first_charge_density_sweep and
2026_04_26_density_sweep. Run from anywhere:

    python notebooks_2026/density_sweep.py [--run rho_fc rho_all] [--n-jobs 8] [--show]
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import tuning, tuning_cli  # noqa: E402

if __name__ == '__main__':
    pin = tuning.TUNED['rho_vc']
    tuning_cli.main(
        description=__doc__.splitlines()[0],
        spec_names=['rho_fc', 'rho_all'],
        pins={'rho_fc': pin, 'rho_all': pin},
        zlims={'v_rmse': (0, 250), 'e_rmse': (0, 100)},
        selections=[('rho_all', 'e_rmse', 'rho_ec')],
    )
