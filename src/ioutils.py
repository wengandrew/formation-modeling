""" Utilities for input/output management."""
import numpy as np
import pandas as pd
import os, sys


def fetch_file_list(cellid):

    target_dir = os.getcwd()

    target_dir_arbin = f'{target_dir}/data/raw/from-voltaiq'


    if cellid == 152064:
        file_arbin_list = [
        f'{target_dir_arbin}/UMBL2022FEB_CELL152064_FORMTAP_2_P45C_5P0PSI_20220901_R1.csv',
        f'{target_dir_arbin}/UMBL2022FEB_CELL152064_FORMBASE_1_P45C_5P0PSI_20220902_R1.csv',
        f'{target_dir_arbin}/UMBL2022FEB_CELL152064_FORMAGING_1_P45C_5P0PSI_20220909_R1.csv',
        f'{target_dir_arbin}/UMBL2022FEB_CELL152064_CYC_1C1CR1_P45C_5P0PSI_20220923_R1.csv'
                        ]

        daq_channel = 'Key_CH1'

        hr_max = 20*24

    elif cellid == 152074: # BASELINE FORMATION (repeat 2)

        file_arbin_list = [
        f'{target_dir_arbin}/UMBL2022FEB_CELL152074_FORMTAP_2_P45C_5P0PSI_20220901_R1.csv',
        f'{target_dir_arbin}/UMBL2022FEB_CELL152074_FORMBASE_1_P45C_5P0PSI_20220902_R1.csv',
        f'{target_dir_arbin}/UMBL2022FEB_CELL152074_FORMAGING_1_P45C_5P0PSI_20220909_R1.csv'
                        ]

        daq_channel = 'Key_CH0'

        hr_max = 20*24

    elif cellid == 152071: # FAST FORMATION

        file_arbin_list = [
            f'{target_dir_arbin}/UMBL2022FEB_CELL152071_FORMTAP_2_P45C_5P0PSI_20220901_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152071_FORMFAST_1_P45C_5P0PSI_20220902_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152071_FORMAGING_1_P45C_5P0PSI_20220909_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152071_CYC_1C1CR1_P45C_5P0PSI_20220923_R1.csv'
        ]

        daq_channel = 'Key_CH3'

        hr_max = 20*24

    elif cellid == 152098: # SUPER FAST FORMATION

        file_arbin_list = [
            f'{target_dir_arbin}/UMBL2022FEB_CELL152098_FORMTAP_2_P45C_5P0PSI_20220901_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152098_FORMFAST_2_P45C_5P0PSI_20220902_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152098_FORMAGING_1_P45C_5P0PSI_20220909_R1.csv',
            f'{target_dir_arbin}/UMBL2022FEB_CELL152098_CYC_1C1CR1_P45C_5P0PSI_20220923_R1.csv'
        ]

        daq_channel = 'Key_CH2'

        hr_max = 20*24

    else:
        raise ValueError(f"Unsupported cellid: {cellid}. "
                         f"Valid IDs are: 152064, 152074, 152071, 152098.")

    return file_arbin_list, daq_channel, hr_max



def fetch_experimental_data(cellid, max_hours=1000000, to_include_cycling=False):

    target_dir = os.getcwd()

    # To make the dQ/dV plot work, be sure to only select the FORM protocol without
    # including the aging file. Otherwise the time vector will be confused.
    if to_include_cycling:
        file_indices_to_include = np.array([1,2,3])
    else:
        file_indices_to_include = np.array([1,2])

    file_arbin_list, daq_channel, hr_max = fetch_file_list(cellid)

    # Load the Arbin Data
    df_arbin_list = []

    for file in np.array(file_arbin_list)[file_indices_to_include]:

        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp']= df['Timestamp'].apply(lambda x : x.tz_localize(None))
        df_arbin_list.append(df)

    df_arbin = pd.concat(df_arbin_list, axis=0)

    # Filter the Arbin Data
    qq = df_arbin['Charge Capacity (Ah)']
    tv = df_arbin['Timestamp'].astype(int)
    tv = (tv - np.min(tv)) / 1e9
    y_voltage = df_arbin['Potential (V)'].copy()
    y_current = df_arbin['Current (A)'].copy()

    mask = (tv/3600 > hr_max)
    y_voltage.loc[mask] = np.nan
    y_current.loc[mask] = np.nan

    # Load the Keyence data
    file_keyence = f'{target_dir}/data/raw/from-keyence/keyence_20220901_213928.csv'
    df_keyence = pd.read_csv(file_keyence)
    df_keyence['computer time'] = pd.to_datetime(df_keyence['computer time'], unit='s')
    df_keyence['computer time'] = df_keyence['computer time'].apply(lambda x : x.tz_localize(None))

    # Filter the Keyence data
    df_keyence = df_keyence.loc[(df_keyence['computer time'] > df_arbin['Timestamp'].iloc[0]) & \
                                (df_keyence['computer time'] < df_arbin['Timestamp'].iloc[-1])]
    y_strain = df_keyence[daq_channel] - df_keyence[daq_channel].iloc[0]
    y_strain[y_strain > 500] = np.nan

    tt = df_keyence['computer time'].astype(int)
    tt = (tt - np.min(tt)) / 1e9

    y_strain.iloc[np.where(tt/3600 > hr_max)] = np.nan

    out = dict()

    idx = np.where(tv < max_hours*3600)[0]
    out['arbin_time'] = tv.iloc[idx]
    out['arbin_voltage'] = y_voltage.iloc[idx]
    out['arbin_current'] = y_current.iloc[idx]

    idx = np.where(tt < max_hours * 3600)[0]
    out['daq_time'] = tt.iloc[idx]
    out['daq_strain'] = y_strain.iloc[idx]

    return out, df_arbin


def smooth_kink_region(time_hours, strain, t_start, t_end, deriv_window=0.05):
    """
    Replace data in [t_start, t_end] with a smooth cubic Hermite curve.
    
    The curve matches:
    - Value at t_start (left endpoint)
    - Value at t_end (right endpoint)  
    - Derivative at t_end (for continuity)
    
    Parameters:
    -----------
    time_hours : array-like
        Time in hours
    strain : array-like
        Strain data
    t_start : float
        Start of affected region (hours)
    t_end : float
        End of affected region (hours)
    deriv_window : float
        Time window after t_end to estimate derivative (hours)
    
    Returns:
    --------
    strain_smoothed : array
        Strain with smoothed region
    """
    time_hours = np.array(time_hours)
    strain = np.array(strain)
    strain_smoothed = strain.copy()
    
    # Find indices for the affected region
    mask_region = (time_hours >= t_start) & (time_hours <= t_end)
    idx_region = np.where(mask_region)[0]
    
    if len(idx_region) == 0:
        return strain_smoothed
    
    # Get endpoint values
    # Left endpoint: closest point at or just before t_start
    idx_left = np.searchsorted(time_hours, t_start)
    if idx_left > 0:
        idx_left -= 1
    t_left = time_hours[idx_left]
    y_left = strain[idx_left]
    
    # Right endpoint: closest point at or just after t_end
    idx_right = np.searchsorted(time_hours, t_end)
    if idx_right >= len(time_hours):
        idx_right = len(time_hours) - 1
    t_right = time_hours[idx_right]
    y_right = strain[idx_right]
    
    # Estimate derivative at right endpoint using data just after t_end
    mask_deriv = (time_hours > t_end) & (time_hours <= t_end + deriv_window)
    if np.sum(mask_deriv) >= 2:
        t_deriv = time_hours[mask_deriv]
        y_deriv = strain[mask_deriv]
        # Linear fit to estimate slope
        dy_right = np.polyfit(t_deriv, y_deriv, 1)[0]
    else:
        # Fallback: use simple difference
        dy_right = (strain[idx_right + 1] - strain[idx_right]) / (time_hours[idx_right + 1] - time_hours[idx_right])
    
    # For left endpoint, we can allow a discontinuity, so set derivative to match the general trend
    # Use a simple linear slope from left to right as a starting guide
    dy_left = (y_right - y_left) / (t_right - t_left)
    
    # Cubic Hermite interpolation
    # h(t) = a*t^3 + b*t^2 + c*t + d
    # where t is normalized to [0, 1]
    # h(0) = y_left, h(1) = y_right
    # h'(0) = dy_left * dt, h'(1) = dy_right * dt
    
    dt = t_right - t_left
    
    # Hermite basis functions for normalized t in [0,1]:
    # h(s) = (2s^3 - 3s^2 + 1)*y0 + (s^3 - 2s^2 + s)*m0 + (-2s^3 + 3s^2)*y1 + (s^3 - s^2)*m1
    # where m0 = dy0 * dt, m1 = dy1 * dt
    
    m0 = dy_left * dt
    m1 = dy_right * dt
    
    # Apply smoothing to points in the region
    t_region = time_hours[mask_region]
    s = (t_region - t_left) / dt  # Normalize to [0, 1]
    
    # Hermite interpolation
    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2
    
    strain_smoothed[mask_region] = h00*y_left + h10*m0 + h01*y_right + h11*m1
    
    return strain_smoothed