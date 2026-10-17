"""Tests for src/cellsim.py"""

import numpy as np
import pytest
from src import cellsim


class TestCell:
    def test_load_config_sets_attributes(self, default_cell):
        assert hasattr(default_cell, 'Cn')
        assert hasattr(default_cell, 'Cp')
        assert hasattr(default_cell, 'theta_n')
        assert hasattr(default_cell, 'theta_p')

    def test_load_config_values(self, default_cell):
        assert default_cell.Cn == pytest.approx(3.14)
        assert default_cell.Cp == pytest.approx(3.02)

    def test_default_ocp_functions_callable(self, default_cell):
        un = default_cell.Un(0.5)
        up = default_cell.Up(0.5)
        assert np.isfinite(un)
        assert np.isfinite(up)

    def test_get_tag_returns_string(self, default_cell):
        tag = default_cell.get_tag()
        assert isinstance(tag, str)
        assert len(tag) > 0


class TestSimulationInit:
    def test_time_array_length(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100, dt=5.0)
        assert len(sim.t) == 20

    def test_initial_theta_n(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100, dt=5.0)
        assert sim.theta_n[0] == pytest.approx(default_cell.theta_n)

    def test_initial_theta_p(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100, dt=5.0)
        assert sim.theta_p[0] == pytest.approx(default_cell.theta_p)

    def test_initial_vt_is_ocv(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100, dt=5.0)
        expected = default_cell.Up(default_cell.theta_p) - default_cell.Un(default_cell.theta_n)
        assert sim.vt[0] == pytest.approx(expected, rel=1e-5)

    def test_step_vectors_initialized_nan(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100, dt=5.0)
        assert np.all(np.isnan(sim.theta_n[1:]))
        assert np.all(np.isnan(sim.theta_p[1:]))

    def test_curr_k_starts_at_zero(self, short_sim):
        assert short_sim.curr_k == 0


class TestSimulationStep:
    def test_step_advances_theta_n_on_charge(self, short_sim):
        # theta_n[k+1] uses i_int[k], so takes two steps to see movement
        short_sim.step(0, 'cc', icc=1.0)
        short_sim.step(1, 'cc', icc=1.0)
        assert short_sim.theta_n[2] > short_sim.theta_n[0]

    def test_step_decreases_theta_p_on_charge(self, short_sim):
        theta_p_before = short_sim.theta_p[0]
        short_sim.step(0, 'cc', icc=1.0)
        assert short_sim.theta_p[1] < theta_p_before

    def test_step_zero_current_leaves_theta_unchanged(self, short_sim):
        theta_n_before = short_sim.theta_n[0]
        theta_p_before = short_sim.theta_p[0]
        short_sim.step(0, 'cc', icc=0.0)
        assert short_sim.theta_n[1] == pytest.approx(theta_n_before, abs=1e-10)
        assert short_sim.theta_p[1] == pytest.approx(theta_p_before, abs=1e-10)

    def test_current_conservation_approx(self, short_sim):
        # i_app ≈ i_int + i_sei at each step (within SEI reaction tolerance)
        short_sim.step(0, 'cc', icc=1.0)
        i_app = short_sim.i_app[0]
        i_int = short_sim.i_int[1]
        i_sei = short_sim.i_sei[1]
        assert i_app == pytest.approx(i_int + i_sei, rel=1e-5)

    def test_R_sei_positive(self, short_sim):
        short_sim.step(0, 'cc', icc=1.0)
        assert short_sim.R_sei[1] > 0

    def test_delta_sei_non_negative(self, short_sim):
        short_sim.step(0, 'cc', icc=1.0)
        assert short_sim.delta_sei1[1] >= 0
        assert short_sim.delta_sei2[1] >= 0

    def test_vt_finite_after_step(self, short_sim):
        short_sim.step(0, 'cc', icc=1.0)
        assert np.isfinite(short_sim.vt[1])


class TestRunRest:
    def test_advances_curr_k(self, short_sim):
        k_before = short_sim.curr_k
        short_sim.run_rest(1, rest_time_hrs=0.1)
        assert short_sim.curr_k > k_before

    def test_zero_current_during_rest(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.1)
        k_end = short_sim.curr_k
        assert np.all(short_sim.i_app[:k_end] == 0.0)

    def test_step_number_label(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.1)
        k_end = short_sim.curr_k
        assert np.all(short_sim.step_number[1:k_end] == cellsim.STEP_NUM_REST)


class TestRunChgCccv:
    def test_reaches_vmax(self, default_cell):
        # Use a small simulation with a low vmax so the test is fast
        sim = cellsim.Simulation(default_cell, sim_time_s=50000, dt=5.0)
        sim.run_rest(1, rest_time_hrs=0.05)
        sim.run_chg_cccv(1, icc=2.5, icv=0.125, vmax=3.5)
        # After CCCV, terminal voltage should be at or near vmax
        df = sim.get_results()
        assert df['vt'].max() >= 3.49

    def test_cc_then_cv_step_numbers(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=50000, dt=5.0)
        sim.run_rest(1, rest_time_hrs=0.05)
        sim.run_chg_cccv(1, icc=2.5, icv=0.125, vmax=3.5)
        df = sim.get_results()
        chg = df[df['cycle_number'] == 1]
        # Both CC and CV step numbers should appear
        step_nums = set(chg['step_number'].unique())
        assert cellsim.STEP_NUM_CHARGE_CC in step_nums
        assert cellsim.STEP_NUM_CHARGE_CV in step_nums


class TestRunDchCccv:
    def test_reaches_vmin(self, default_cell):
        sim = cellsim.Simulation(default_cell, sim_time_s=100000, dt=5.0)
        sim.run_rest(1, rest_time_hrs=0.05)
        # Charge first so there's capacity to discharge
        sim.run_chg_cccv(1, icc=2.5, icv=0.125, vmax=4.2)
        sim.run_dch_cccv(1, icc=-2.5, icv=-0.125, vmin=3.0)
        df = sim.get_results()
        assert df['vt'].min() <= 3.01


class TestGetResults:
    def test_returns_dataframe(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.1)
        df = short_sim.get_results()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_no_nan_in_vt(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.1)
        df = short_sim.get_results()
        assert not df['vt'].isna().any()

    def test_dq_non_negative(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.1)
        df = short_sim.get_results()
        assert (df['dq'] >= 0).all()

    def test_trimmed_to_curr_k(self, short_sim):
        short_sim.run_rest(1, rest_time_hrs=0.05)
        df = short_sim.get_results()
        assert len(df) == short_sim.curr_k


class TestStepNumConstants:
    def test_all_distinct(self):
        constants = [
            cellsim.STEP_NUM_CHARGE_CC,
            cellsim.STEP_NUM_CHARGE_CV,
            cellsim.STEP_NUM_DISCHARGE_CC,
            cellsim.STEP_NUM_DISCHARGE_CV,
            cellsim.STEP_NUM_REST,
        ]
        assert len(constants) == len(set(constants)), "Step number constants must be unique"
