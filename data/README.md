Download the contents from [here](https://drive.google.com/drive/folders/1Xmzc_Yqts-pJuAU8GAySb3E2Zf5M6oGp?usp=sharing).

You should end up with three sub-folders in this directory:
- `interim`
- `processed`
- `raw`

Make sure you don't end up with something like `data/data`.

# Description of Files

`data/processed/hppc_1.csv` - this file contains cell resistance as a function of SOC as measured by an HPPC test. The file originates from the `fast-formation` project (`github.com/wengandrew/fast-formation`). The code to run is `notebooks-2021-joule/2021_06_08_build_electrode_stoichiometry_model.ipynb`. The specific cell that was used to obtain the data comes from the Weng2021 dataset (Joule paper), cell number 4. The data represents the HPPC test run at the end of formation (at the beginning of the cycling test.)