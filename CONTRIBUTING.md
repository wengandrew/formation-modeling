# Contributor Guidelines

Please follow the following guidelines when contributing code to this
repository.


## Do not change any files in `data/raw`

The data here is raw data. It is immutable.


## Keep figures in notebooks

- For making publication-ready plots, keep one notebook per figure.
- Don't give figure numbers, since this may change. Instead, you can name each
    notebook as something descriptive, e.g.:
      - `fig_effect_of_pressure.ipynb`
      - `fig_expanded_stoichiometry_model.ipynb`


## One figure, one notebook

Create a separate notebook for each figure. Don't try to put too much in one
figure.


## Keep functions in source code

It's okay to prototype functions in notebooks, but when they are ready,
refactor them into Python source files under `src/`.
