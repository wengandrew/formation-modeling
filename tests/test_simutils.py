"""Tests for src/simutils.py"""

import numpy as np
import pytest
from src import simutils


class TestCalculateRmse:
    def _make_sine(self, n=200):
        t = np.linspace(0, 10, n)
        y = np.sin(t)
        return t, y

    def test_perfect_match_gives_zero(self):
        t, y = self._make_sine()
        rmse, _, _ = simutils.calculate_rmse(t, y, t, y)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_rmse_is_non_negative(self):
        t, y = self._make_sine()
        y_noisy = y + np.random.default_rng(42).normal(0, 0.1, len(y))
        rmse, _, _ = simutils.calculate_rmse(t, y, t, y_noisy)
        assert rmse >= 0

    def test_rmse_increases_with_noise(self):
        t, y = self._make_sine()
        rng = np.random.default_rng(0)
        rmse_small, _, _ = simutils.calculate_rmse(t, y, t, y + rng.normal(0, 0.01, len(y)))
        rmse_large, _, _ = simutils.calculate_rmse(t, y, t, y + rng.normal(0, 1.0, len(y)))
        assert rmse_large > rmse_small

    def test_y_range_filter(self):
        t = np.linspace(0, 1, 100)
        y_meas = np.ones(100)
        y_modl = np.ones(100)
        y_modl[50:] = 10.0  # spike outside range
        rmse, _, _ = simutils.calculate_rmse(t, y_meas, t, y_modl, y_range=[0, 5])
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_ignore_time_range(self):
        t = np.linspace(0, 10, 200)
        y_meas = np.zeros(200)
        y_modl = np.zeros(200)
        y_modl[100:120] = 100.0  # big spike in ignored window
        rmse, _, _ = simutils.calculate_rmse(
            t, y_meas, t, y_modl,
            ignore_time_range=[t[100], t[120]]
        )
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_returns_three_values(self):
        t, y = self._make_sine()
        result = simutils.calculate_rmse(t, y, t, y)
        assert len(result) == 3

    def test_different_length_vectors(self):
        t_meas = np.linspace(0, 10, 50)
        y_meas = np.sin(t_meas)
        t_modl = np.linspace(0, 10, 200)
        y_modl = np.sin(t_modl)
        rmse, _, _ = simutils.calculate_rmse(t_meas, y_meas, t_modl, y_modl)
        assert rmse < 0.01  # should be close even with different grids


class TestInterpolateRawData:
    def test_output_on_regular_grid(self):
        t = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        y = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        t_out, y_out = simutils.interpolate_raw_data(t, y, dt=1.0)
        assert np.allclose(np.diff(t_out), 1.0)

    def test_output_within_input_bounds(self):
        t = np.linspace(0, 20, 100)
        y = np.sin(t)
        t_out, y_out = simutils.interpolate_raw_data(t, y, dt=0.5)
        assert t_out.min() >= t.min()
        assert t_out.max() <= t.max()

    def test_identity_on_uniform_data(self):
        t = np.linspace(0, 10, 101)
        y = t * 2.0
        t_out, y_out = simutils.interpolate_raw_data(t, y, dt=0.1)
        assert np.allclose(y_out, t_out * 2.0, atol=1e-10)
