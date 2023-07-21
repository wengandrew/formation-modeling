# Formation Modeling

Half-cell OCV-R modeling for battery formation dynamics.

A pre-print of this work is available [here](https://arxiv.org/abs/2305.18722).

Runs on Python 3.8.8.

# Getting Started

1. Clone this repository.

2. Make sure you have Python 3.8.8 installed. Another version of Python will probably work, but no guarantees!

3. Set up your Python virtual environment
  - Make sure you're in the root directory of the repo
  - Create the virtual environment: `python -m venv venv`
    - You should see a new folder called `venv`

4. Install the relevant packages
  - Make sure your virtual environment is activated
    - Run `source venv/bin/activate`
      - You should see `(venv)` appear on the Terminal prompt
    - Also check that `pip` is pointing to the instance from the venv, not from system:
      `which pip`
  - Install the packages using `pip install -r requirements.txt`


5. Download the data files
  - Follow the link in the `data` folder


# To-Do

## Numerical Improvements

- [ ] Fix oscillations at higher C-rates and during the CV hold
- [ ] Refactor the numerical integration scheme to be standard form (dxdt =
    f(x, t, ...)
- [ ] Reparameterize Rn, Ln, Lp based on the full expansion model
- [x] Fix anode expansion function (there's a kink; the equation from Mohtat2020 may have some typos)
- [x] Fix $\delta_p + \delta_n \neq \delta_{tot}$ issue
- [x] Reparameterize $U_n$ and $U_p$ to align with the measured initial full cell voltage before formation


## Experiments
- [ ] Define simulation output metrics directly in the model itself for
    convenience and modularization
    - [ ] CE, Qd, and Qc for cycles 1, 2, 3
    - [ ] dQ/dV peak positions
- [ ] Run the initial formation charge at different C-rates
- [ ] Run the initial formation charge at different CV hold conditions
- [ ] Run calendar aging experiments at different SOCs following formation
