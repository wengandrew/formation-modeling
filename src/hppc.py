"""Series resistance from the HPPC pulses in the cycler data.

Ported from notebooks/2025_04_16_hppc.ipynb, which produced the measured HPPC
resistances in notebooks_2026/data/measurements.yaml (the medians of the
per-pulse values returned here).
"""

import numpy as np
import pandas as pd

# Cycler step indices making up the HPPC sequence, per cell
HPPC_STEPS = {
    152064: [12, 13, 14, 15, 16, 17],
    152071: [14, 15, 16, 17, 18, 19],
    152098: [14, 15, 16, 17, 18, 19],
}

# Cycler timestamps bounding the first (after formation) and second (after
# formation aging) HPPC sequences
HPPC_WINDOWS = {
    1: (None, '2022-09-15'),
    2: ('2022-09-15', '2022-10-01'),
}


def pulse_resistances(voltage, current, time_s, min_current_step=1.5, max_dt_s=10):
    """Series resistance at each discharge-pulse onset.

    A pulse onset is a current drop larger than min_current_step [A] between
    consecutive samples less than max_dt_s apart. Returns (R [Ohm], index of the
    sample before each onset).
    """
    voltage, current, time_s = (np.asarray(x, dtype=float) for x in (voltage, current, time_s))
    d_current = -(current[0:-2] - current[1:-1])
    dt = time_s[0:-2] - time_s[1:-1]
    d_voltage = voltage[1:-1] - voltage[0:-2]
    idx = np.where((d_current < -min_current_step) & (np.abs(dt) < max_dt_s))[0]
    return d_voltage[idx] / d_current[idx], idx


def hppc_resistance(df_arbin, cellid, hppc):
    """(SOC [%], R [Ohm]) for each pulse of HPPC sequence `hppc` (1 or 2).

    df_arbin is the cycler data from ioutils.fetch_experimental_data(...,
    to_include_cycling=True). SOC is taken as the pulse's position in time
    through the sequence (0 % at the first pulse, 100 % at the last).
    """
    start, end = HPPC_WINDOWS[hppc]
    mask = df_arbin['Step Index'].isin(HPPC_STEPS[cellid])
    if start is not None:
        mask &= df_arbin['Timestamp'] > pd.to_datetime(start)
    mask &= df_arbin['Timestamp'] < pd.to_datetime(end)
    subset = df_arbin[mask]

    time_s = (subset['Timestamp'] - subset['Timestamp'].iloc[0]).dt.total_seconds().to_numpy()
    resistance, idx = pulse_resistances(subset['Potential (V)'], subset['Current (A)'], time_s)
    t_pulse = time_s[idx] - time_s[idx][0]
    soc = t_pulse / np.max(t_pulse) * 100
    return soc, resistance
