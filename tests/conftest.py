"""Shared fixtures for the formation-modeling test suite."""

import pytest
from src import cellsim


@pytest.fixture
def default_cell():
    cell = cellsim.Cell()
    cell.load_config('params/default.yaml')
    return cell


@pytest.fixture
def short_sim(default_cell):
    """A Simulation pre-loaded with a short time array for fast tests."""
    return cellsim.Simulation(default_cell, sim_time_s=3600, dt=5.0)
