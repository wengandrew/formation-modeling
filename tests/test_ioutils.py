"""Tests for src/ioutils.py"""

import pytest

from src import ioutils

CELL_IDS = (152064, 152074, 152071, 152098)


@pytest.mark.parametrize('cellid', CELL_IDS)
def test_experimental_data_is_not_truncated(cellid):
    # The 2026 tuning sweeps used experimental data through ~497 h; the
    # per-cell 480 h cutoff in fetch_file_list must stay disabled.
    _, _, hr_max = ioutils.fetch_file_list(cellid)
    assert hr_max > 1000


def test_unknown_cell_raises():
    with pytest.raises(ValueError):
        ioutils.fetch_file_list(123)
