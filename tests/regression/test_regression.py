"""run_sim() results and figures must match the recorded references exactly.

The references were recorded with make_references.py before refactoring
run_sim() and the plotting code. Long run_sim cases need --runslow.
"""

import json
import os

import matplotlib
import pytest

matplotlib.use('Agg')

from src import cellsim  # noqa: E402
from tests.regression.cases import (  # noqa: E402
    HEATMAP_CASES, PLOT_CASES, RUN_SIM_CASES, SLOW_RUN_SIM_CASES, SWEEP_CASES)
from tests.regression.make_references import REFERENCE_FILE  # noqa: E402

with open(REFERENCE_FILE) as f:
    REFERENCES = json.load(f)


def _params(cases, slow=()):
    return [pytest.param(name, marks=pytest.mark.slow) if name in slow else name
            for name in cases]


def _check(name, cases):
    assert name in REFERENCES, f'no reference recorded for {name}'
    assert cases[name]() == REFERENCES[name]


@pytest.mark.parametrize('name', _params(RUN_SIM_CASES, SLOW_RUN_SIM_CASES))
def test_run_sim_unchanged(name):
    _check(name, RUN_SIM_CASES)


@pytest.mark.parametrize('name', list(PLOT_CASES))
def test_simulation_plots_unchanged(name, tmp_path, monkeypatch):
    monkeypatch.setattr(cellsim, 'FIGURE_DIR', str(tmp_path))
    _check(name, PLOT_CASES)


@pytest.mark.parametrize('name', list(HEATMAP_CASES))
def test_heatmaps_unchanged(name):
    _check(name, HEATMAP_CASES)


@pytest.mark.parametrize('name', list(SWEEP_CASES))
def test_sweep_heatmaps_unchanged(name):
    _check(name, SWEEP_CASES)


def test_reference_file_is_complete():
    missing = set(RUN_SIM_CASES) | set(PLOT_CASES) | set(HEATMAP_CASES) | set(SWEEP_CASES)
    missing -= set(REFERENCES)
    assert not missing
    assert os.path.getsize(REFERENCE_FILE) > 0
