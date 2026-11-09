"""Tests for src/hppc.py"""

import os

import numpy as np
import pytest

from src import hppc, ioutils, tuning

HAS_EXPT_DATA = os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                                           'from-voltaiq'))


def test_pulse_resistance_of_synthetic_pulse():
    # 2 A discharge pulse through 25 mOhm, sampled every second
    t = np.arange(10.0)
    current = np.where((t >= 4) & (t < 7), -2.0, 0.0)
    voltage = 3.7 + 0.025 * current
    resistance, idx = hppc.pulse_resistances(voltage, current, t)
    np.testing.assert_allclose(resistance, [0.025])
    assert list(idx) == [3]


def test_ignores_small_steps_and_gaps():
    t = np.array([0, 1, 2, 30, 31, 32], dtype=float)
    current = np.array([0, -1.0, -1.0, -1.0, -3.0, -3.0])    # 1 A step; 2 A step after a gap
    voltage = 3.7 + 0.02 * current
    resistance, _ = hppc.pulse_resistances(voltage, current, t)
    np.testing.assert_allclose(resistance, [0.02])


@pytest.mark.skipif(not HAS_EXPT_DATA, reason='experimental data not downloaded')
def test_medians_match_measurements_file():
    # measurements.yaml records the medians computed by the original notebook
    expected = {1: {'base': 21.43, 'fast': 17.70, 'fast+': 16.63},
                2: {'base': 37.52, 'fast': 36.64, 'fast+': 36.61}}
    for protocol in tuning.PROTOCOLS:
        cellid = tuning.EXPT_CELLS[protocol]
        _, df = ioutils.fetch_experimental_data(cellid, to_include_cycling=True)
        for n in (1, 2):
            soc, resistance = hppc.hppc_resistance(df, cellid, n)
            assert np.median(resistance) * 1e3 == pytest.approx(expected[n][protocol], abs=0.005)
            assert soc[0] == 0 and soc[-1] == 100
