"""Tests for src/modelutils.py"""

import numpy as np
import pytest
from src import modelutils as mu


class TestInitialize:
    def test_length_matches_time_vec(self):
        t = np.arange(0, 100, 5.0)
        v = mu.initialize(t, 1.0)
        assert len(v) == len(t)

    def test_first_element_is_initial_val(self):
        t = np.arange(0, 100, 5.0)
        v = mu.initialize(t, 3.14)
        assert v[0] == pytest.approx(3.14)

    def test_remaining_elements_are_nan(self):
        t = np.arange(0, 100, 5.0)
        v = mu.initialize(t, 1.0)
        assert np.all(np.isnan(v[1:]))

    def test_default_initial_val_is_nan(self):
        t = np.arange(0, 10, 1.0)
        v = mu.initialize(t)
        assert np.all(np.isnan(v))


class TestUnUp:
    def test_Un_mid_range(self):
        # Graphite at ~50% SOC should be near 0.12 V vs Li/Li+
        u = mu.Un(0.5)
        assert 0.0 < u < 0.5

    def test_Up_mid_range(self):
        # NMC at ~50% SOC should be near 3.7–4.0 V
        u = mu.Up(0.5)
        assert 3.0 < u < 4.5

    def test_Un_nan_returns_nan(self):
        # NaN stoichiometry propagates through the polynomial (no explicit guard in Un)
        result = mu.Un(np.nan, adjust=False)
        assert np.isnan(result)

    def test_Un_raises_stoichiometry_too_small(self):
        with pytest.raises(ValueError):
            mu.Un(-0.5)

    def test_Un_raises_stoichiometry_too_large(self):
        with pytest.raises(ValueError):
            mu.Un(1.5)

    def test_Up_raises_stoichiometry_too_small(self):
        with pytest.raises(ValueError):
            mu.Up(-0.5)

    def test_Up_raises_stoichiometry_too_large(self):
        with pytest.raises(ValueError):
            mu.Up(1.5)

    def test_Un_monotonically_decreasing(self):
        # Graphite OCP should decrease as lithiation increases
        sto = np.linspace(0.05, 0.95, 20)
        u = np.array([mu.Un(s) for s in sto])
        assert np.all(np.diff(u) < 0)

    def test_adjust_false_skips_stoichiometry_shift(self):
        # With adjust=False, Un(0) and Up(0) should not raise even at exact boundaries
        u = mu.Un(0.0, adjust=False)
        assert np.isfinite(u)
        u = mu.Up(0.0, adjust=False)
        assert np.isfinite(u)


class TestEnEp:
    def test_En_zero_at_zero(self):
        assert mu.En(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_En_positive_for_positive_sto(self):
        for sto in [0.1, 0.3, 0.5, 0.8]:
            assert mu.En(sto) > 0

    def test_En_raises_on_nan(self):
        with pytest.raises(ValueError):
            mu.En(np.nan)

    def test_Ep_negative_for_partial_delithiation(self):
        # NMC contracts on delithiation (sto < 1 means some Li removed)
        assert mu.Ep(0.5) < 0

    def test_Ep_zero_when_fully_lithiated(self):
        assert mu.Ep(1.0) == pytest.approx(0.0)


class TestOcv:
    def test_ocv_at_zero_soc(self):
        v = mu.ocv(0)
        assert v > 0

    def test_ocv_at_full_soc(self):
        v = mu.ocv(1)
        assert v > 0

    def test_ocv_monotonically_increasing(self):
        soc = np.linspace(0.01, 0.99, 50)
        v = np.array([mu.ocv(s) for s in soc])
        assert np.all(np.diff(v) > 0)

    def test_ocv_raises_below_zero(self):
        with pytest.raises(ValueError):
            mu.ocv(-0.1)

    def test_ocv_raises_above_one(self):
        with pytest.raises(ValueError):
            mu.ocv(1.1)

    def test_ocv_rejects_non_scalar(self):
        with pytest.raises(AssertionError):
            mu.ocv(np.array([0.5]))


class TestUpdateEsoh:
    def test_returns_five_values(self):
        result = mu.update_esoh(0.5, q_max=3.0, x100=0.9, y100=0.27, Cn=3.14, Cp=3.02)
        assert len(result) == 5

    def test_ocv_in_reasonable_range(self):
        x, y, un, up, ocv = mu.update_esoh(0.5, q_max=3.0, x100=0.9, y100=0.27, Cn=3.14, Cp=3.02)
        assert 3.0 < ocv < 4.5

    def test_ocv_equals_up_minus_un(self):
        x, y, un, up, ocv = mu.update_esoh(0.5, q_max=3.0, x100=0.9, y100=0.27, Cn=3.14, Cp=3.02)
        assert ocv == pytest.approx(up - un)


class TestEnSeiStressSEI:
    def test_stress_at_zero_is_zero(self):
        assert mu.stressSEI(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_EnSei_at_zero_is_zero(self):
        assert mu.EnSei(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_EnSei_at_one_matches_max_strain(self):
        # By construction, EnSei(1) == max_strain
        assert mu.EnSei(1.0) == pytest.approx(0.1318, rel=1e-4)
