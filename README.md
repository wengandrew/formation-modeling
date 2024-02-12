# Formation Modeling

December 2023

Source code for modeling battery formation.

This work is published in the Journal of the Electrochemical Society [here](https://iopscience.iop.org/article/10.1149/1945-7111/aceffe).

The code runs on Python 3.8.8.

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
