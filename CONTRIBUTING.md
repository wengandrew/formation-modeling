# Contributor Guidelines

When contributing code to this repository, please be mindful of the following guidelines.

# Handling raw data

## Do not commit data files into the repo

Raw data files can be really big, like 250 MB. GitHub is not meant for storing such large files. I've been keeping the raw data files in Google Drive (for the specific path, go to `data/`). Follow the setup guide to unpack the data files into your repository. The source code will assume that your data files have already been unpacked into the `data/` folder.

If you need to import more raw data, let's coordinate to do this so we can keep a clean dataset on Google Drive.

# Writing notebooks

## Keep figures in notebooks

For making publication-ready plots, keep one notebook per figure. I generally don't bother putting figure numbers in the filenames, since figure numbers may change very quickly, especially during the last odd hours leading up to a paper submission. Instead, try to give each notebook a descriptive name, e.g.:
  - `2022_11_15_fig_effect_of_pressure.ipynb`
  - `2022_11_15_fig_expanded_stoichiometry_model.ipynb`

## Use Markdown generously

## Start notebooks with `yyyy_mm_dd...`

I find this to be the best way to track and sort through notebooks as the number of notebooks grow

## One figure, one notebook

Create a separate notebook for each figure. Don't try to put too much in one
figure.

## Keep functions in source code

It's okay to prototype functions in notebooks, but when you think they are 'prime-time',
refactor them into Python source files under `src/`.


# Writing source code

Source code is meant for re-use. The core OCV-R model, for example, is a good example of a code that can be re-used across different applications. So this should be kept as a source code rather than 'hard-coded' inside a notebook. 
