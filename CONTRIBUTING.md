# Contributor Guidelines

Please follow the following guidelines when contributing code to this
repository.


## Do not change any files in `data`

The data here is raw data. It is immutable. Only read from them. Don't write to them.

## Keep figures in notebooks

- For making publication-ready plots, keep one notebook per figure.
- Don't give figure numbers, since this may change. Instead, you can name each
    notebook as something descriptive, e.g.:
      - `2022_11_15_fig_effect_of_pressure.ipynb`
      - `2022_11_15_fig_expanded_stoichiometry_model.ipynb`


## Start notebooks with `yyyy_mm_dd`

- I find this to be the best way to track and sort through notebooks

## One figure, one notebook

Create a separate notebook for each figure. Don't try to put too much in one
figure.


## Keep functions in source code

It's okay to prototype functions in notebooks, but when you think they are 'prime-time',
refactor them into Python source files under `src/`.
