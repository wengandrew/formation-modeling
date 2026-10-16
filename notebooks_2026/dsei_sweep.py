"""Steps 1-2 of the SEI parameter tuning: the diffusivities D_LVDC and D_LEDC.

Step 1 (dsei_fc): sweep D_LEDC x D_LVDC over the first charge only and set
    D_LVDC from the voltage RMSE (set by inspection to 1.22e-18 m^2/s).
Step 2 (dsei_all): sweep through formation aging (~400 h), fix D_LVDC, and set
    D_LEDC as the minimum voltage RMSE along that D_LVDC.

Plots voltage and expansion RMSE heatmaps for both windows from the cached
sweeps. Replaces 2026_04_16_first_charge_dsei_sweep and
2026_04_07_clean_dsei_sweep. Run from anywhere:

    python notebooks_2026/dsei_sweep.py [--run dsei_fc dsei_all] [--n-jobs 8] [--show]
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src import tuning, tuning_cli  # noqa: E402

if __name__ == '__main__':
    pin = tuning.TUNED['diff_vc']
    tuning_cli.main(
        description=__doc__.splitlines()[0],
        spec_names=['dsei_fc', 'dsei_all'],
        pins={'dsei_fc': pin, 'dsei_all': pin},
        zlims={'v_rmse': (0.04, 0.2), 'e_rmse': (0, 100)},
        selections=[('dsei_all', 'v_rmse', 'diff_ec')],
    )
