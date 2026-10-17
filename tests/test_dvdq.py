"""Tests for src/dvdq.py"""

import numpy as np
import pytest
from src import dvdq


class TestOcpFunctions:
    def test_f_pos_ocv_mid_range(self):
        u = dvdq.f_pos_ocv(0.5)
        assert 3.0 < u < 4.5

    def test_f_neg_ocv_mid_range(self):
        u = dvdq.f_neg_ocv(0.5)
        assert 0.0 < u < 0.5

    def test_f_pos_ocv_matches_modelutils_Up(self):
        from src import modelutils as mu
        for sto in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert dvdq.f_pos_ocv(sto) == pytest.approx(mu.Up(sto, adjust=False), rel=1e-8)

    def test_f_neg_ocv_matches_modelutils_Un(self):
        from src import modelutils as mu
        for sto in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert dvdq.f_neg_ocv(sto) == pytest.approx(mu.Un(sto, adjust=False), rel=1e-8)


class TestEsohToVoc:
    # x100=0.9 (neg stoic at 100% SOC), y100=0.27 (pos stoic at 100% SOC)
    # capacity q should range from 0 up to about Cn*(x100-x0) ≈ 2.5 Ah
    _X100, _Y100, _CN, _CP = 0.9, 0.27, 3.14, 3.02
    _Q = np.linspace(0, 2.5, 50)

    def test_returns_three_arrays(self):
        result = dvdq.esoh_to_voc(self._X100, self._Y100, self._CN, self._CP, self._Q)
        assert len(result) == 3

    def test_voc_in_voltage_range(self):
        # Start from q=0.5 Ah to avoid extreme stoichiometries near 0% SOC
        q = np.linspace(0.5, 2.5, 40)
        Voc, _, _ = dvdq.esoh_to_voc(self._X100, self._Y100, self._CN, self._CP, q)
        valid = Voc[np.isfinite(Voc)]
        assert np.all(valid > 2.0)
        assert np.all(valid < 5.0)

    def test_voc_equals_up_minus_un(self):
        q = np.linspace(0, 2.5, 50)
        Voc, Up, Un = dvdq.esoh_to_voc(0.9, 0.27, Cn=3.14, Cp=3.02, q=q)
        assert np.allclose(Voc, Up - Un, equal_nan=True)


class TestDegToVocGraphical:
    def test_returns_four_arrays(self):
        result = dvdq.deg_to_voc_graphical(0.0, 0.0, 0.0, Cp=3.02, Cn=3.14)
        assert len(result) == 4

    def test_capacity_vector_length_consistent(self):
        Voc, Un, Up, cap = dvdq.deg_to_voc_graphical(0.0, 0.0, 0.0, Cp=3.02, Cn=3.14)
        assert len(Voc) == len(cap)
        assert len(Un) == len(cap)
        assert len(Up) == len(cap)
