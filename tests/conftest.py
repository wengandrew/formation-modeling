"""Shared fixtures for the formation-modeling test suite."""

import pytest
from src import cellsim


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False,
                     help='run slow tests that re-simulate cached sweep points')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: re-runs full simulations (enable with --runslow)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='needs --runslow')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def default_cell():
    cell = cellsim.Cell()
    cell.load_config('params/default.yaml')
    return cell


@pytest.fixture
def short_sim(default_cell):
    """A Simulation pre-loaded with a short time array for fast tests."""
    return cellsim.Simulation(default_cell, sim_time_s=3600, dt=5.0)
