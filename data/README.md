Download the contents from [here](https://drive.google.com/drive/folders/1Xmzc_Yqts-pJuAU8GAySb3E2Zf5M6oGp?usp=sharing).

You should end up with three sub-folders in this directory:
- `interim`
- `processed`
- `raw`

Make sure you don't end up with something like `data/data`.

`src/ioutils.py` expects (paths relative to the repo root):
- `data/raw/from-voltaiq/UMBL2022FEB_CELL<id>_*.csv`: Arbin cycler exports for
  cells 152064 (base), 152074 (base, repeat), 152071 (fast) and 152098 (fast+)
- `data/raw/from-keyence/keyence_20220901_213928.csv`: expansion (Keyence) data

Run code from the repo root; the paths are resolved against the working directory.

# Description of Files

`data/processed/hppc_1.csv` - this file contains cell resistance as a function of SOC as measured by an HPPC test. The file originates from the `fast-formation` project (`github.com/wengandrew/fast-formation`). The code to run is `notebooks-2021-joule/2021_06_08_build_electrode_stoichiometry_model.ipynb`. The specific cell that was used to obtain the data comes from the Weng2021 dataset (Joule paper), cell number 4. The data represents the HPPC test run at the end of formation (at the beginning of the cycling test.)