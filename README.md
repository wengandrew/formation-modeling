# Formation Modeling

Half-cell OCV-R modeling for battery formation dynamics

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

Terminology:
- $\varepsilon$ : strain (usually in micrometers)
- $V$ : voltage (V)
- $I$ : current (A)

I suggest we start a notebook for each of the following views:
- [ ] **Main Result**: A summary plot with V(t) and $\varepsilon(t)$ as horizontally-stacked subplots, and formation types as columns. There should be three columns. The time axis needs to be aligned.
- [ ] **Investigation**: Modified functions for $U_n$ and $U_p$ where $U_n(x=0)$ corresponds to the pre-formation neg. electrode potential, and $U_p(y=1)$ corresponds to the pre-formation positive potential. Create these functions in the source code, and then visualize them inside the notebook. Plot $U_n$ and $U_p$, aligned on the x-axis in a common, full cell capacity basis. Provide references or justifications for picking the specific values of $U_p(y=1)$ and $U_n(x=0)$. Is this from literature or from some internal test data? Summarize the results in a notebook.
- [ ] **Investigation**: A method to estimate the irreversible expansion during formation, by "reverse" extrapolating $\varepsilon(t)$ from the cycling data. Start with the raw $V(t)$ and $\varepsilon(t)$ data. 'Background correct' out the effect of the irreversible expansion for now. Try to develop a mapping between OCV and reversible expansion. Then, go back to the formation data and try to model the OCV at each point in time. Then assign a proportion of the total $\varepsilon(t)$ to the irreversible expansion, and the remainder to the reversible expansion.
- [ ] **Main Result**: Take an existing formation protocol and model the output voltage during formation. Compare this to the experimentally measured data. To do this, we may need to implement the CV hold condition in the source code. Right now, it can only do CC. But we should be able to implement the CV mode.
