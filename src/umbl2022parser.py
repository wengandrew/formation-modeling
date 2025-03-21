"""
Parser utilities for UMBL2022FEB dataset
"""

import numpy as np
import pandas as pd
import pickle

NOM_CAP = 2.5 # Nominal capacity in amp-hours

# Declare constants
STEP_INDEX_RPT_C20_CHARGE = 26
STEP_INDEX_RPT_C20_DISCHARGE = 23

# HPPC step indices (starting and ending)
# TODO: resolve the individual step indices for HPPC
STEP_INDEX_RPT_HPPC_START = 13
STEP_INDEX_RPT_HPPC_END = 19

# Any step indices equalling or above this value belongs to the RPT block and not the CYC block
STEP_INDEX_RPT_THRESHOLD = 10

ROOT_PATH = '/Users/aweng/Documents/PROJ_UMBL2022FEB/pouch/'


def read_pickle(path):
    """ Easily load data from Pickle files"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def fetch_raw_data_dict(device_id):
    """
    Fetches a dictionary of data from a device
    """


    full_path_raw = ROOT_PATH + f'UMBL2022FEB_CELL{device_id}.pkl'

    data_dict = read_pickle(full_path_raw)

    return data_dict


def fetch_raw_data(device_id, filetype):
    """
    Fetch DataFrame from a dictionary of dataframes based on a type

    Args:
    device_id (int): ID for the device
    filetype (str): Type of DataFrame to fetch
       options are 'formation_cycle', 'formation_tap', 'formation_aging', 'cycling', 'aux'

    Returns:
    DataFrame: DataFrame of the specified type

    """

    assert filetype in ['formation_cycle',
                        'formation_tap',
                        'formation_aging',
                        'cycling',
                        'aux'], \
        f"Invalid filetype: {filetype}"

    data_dict = fetch_raw_data_dict(device_id)

    key_list = list(data_dict.keys())

    result_list = []

    if filetype == 'formation_cycle':
        for key in key_list:
            if any([x in key for x in ['FORMBASE', 'FORMFAST']]):
                result_list.append(key)
    if filetype == 'formation_tap':
        for key in key_list:
            if 'FORMTAP' in key:
                result_list.append(key)
    if filetype == 'formation_aging':
        for key in key_list:
            if 'FORMAGING' in key:
                result_list.append(key)
    if filetype == 'cycling':
        # Manually-identified exclusion list based on studying raw data traces
        for key in key_list:
            if key == 'UMBL2022FEB_CELL152016_CYC_1C1CR1_P45C_P5P0PSI_20220804_R1':
                continue
            if key == 'UMBL2022FEB_CELL151803_CYC_1C1CR1_P45C_5P0PSI_20220804':
                continue
            if key == 'UMBL2022FEB_CELL151804_CYC_1C1CR1_P45C_5P0PSI_20230227_R1-01-004':
                continue
            if 'CYC' in key and 'VMONITOR' not in key:
                result_list.append(key)
    if filetype == 'aux':
        for key in key_list:
            if 'AuxDat' in key:
                result_list.append(key)

    if device_id == 151805:
        result_list = ['UMBL2022FEB_CELL151805_CYC_1C1CR1_P45C_5P0PSI_20220804_R1']

    assert len(result_list) >= 1, f"No keys found for device {device_id}."

    df = None

    if len(result_list) == 1:
        key = result_list[0]
        df = data_dict[key]

    # With two dataframes, concatenate in chronological order
    if len(result_list) == 2 and filetype == 'cycling':

        df1 = data_dict[result_list[0]]
        df2 = data_dict[result_list[1]]

        if df1['h_datapoint_datetime'].min() < df2['h_datapoint_datetime'].min():
            df2['i_cycle_num'] = df2['i_cycle_num'] + df1['i_cycle_num'].max()
            df2['h_test_time'] = df2['h_test_time'] + df1['h_test_time'].max()
            # df = pd.concat([df1, df2], axis=0)
            df = df1
        else:
            df1['i_cycle_num'] = df1['i_cycle_num'] + df2['i_cycle_num'].max()
            df1['h_test_time'] = df1['h_test_time'] + df2['h_test_time'].max()
            # df = pd.concat([df2, df1], axis=0)
            df = df2
    return df


def process_cycling_data(df):
    """
    Process cycling data to extract cycle-by-cycle data and RPT data

    Args:
    df (DataFrame): Cycling data

    Returns:
    df_cyc (DataFrame): Cycling data without RPTs
    df_rpt (DataFrame): Cycling data with only RPTs

    """

    assert df is not None, "No DataFrame to process."

    df_agg = df.groupby('i_cycle_num', as_index=False).agg({
                                'h_test_time': 'max',
                                'h_discharge_capacity': 'max',
                                'h_datapoint_datetime': 'max'})

    df_agg['cumulative_capacity'] = np.cumsum(df_agg['h_discharge_capacity'])

    # Clean the data

    CAPACITY_THRESHOLD = 1.0
    df_agg.loc[df_agg['h_discharge_capacity'] < CAPACITY_THRESHOLD, \
                'h_discharge_capacity'] = np.nan
    df_agg.loc[df_agg['h_discharge_capacity'] > 3, \
                'h_discharge_capacity'] = np.nan

    # Split data for cycling vs RPT
    cycle_index_rpt_all = df[(df['h_step_index'] >= 10) |
                         (df['h_step_index'] < 3)]['i_cycle_num'].unique()

    cycle_index_rpt_cap = df[df['h_step_index'] == STEP_INDEX_RPT_C20_DISCHARGE]\
                                ['i_cycle_num'].unique()

    df_cyc = df_agg[~df_agg['i_cycle_num'].isin(cycle_index_rpt_all)]
    df_rpt = df_agg[df_agg['i_cycle_num'].isin(cycle_index_rpt_cap)]

    return df_cyc, df_rpt


def update_summary_table(df, id, df_cyc, df_rpt, group):
    """
    Update fields in the summary table based on cycling data

    Args:
    df (DataFrame): Summary table
    id (int): Device ID
    df_cyc (DataFrame): Cycling data without RPTs
    df_rpt (DataFrame): Cycling data with only RPTs
    group (str): label for group
    """

    df.loc[df['device_id'] == id, 'group'] = group
    df.loc[df['device_id'] == id, 'initial_rpt_capacity_ah'] = df_rpt['h_discharge_capacity'].iloc[0]
    df.loc[df['device_id'] == id, 'initial_cyc_capacity_ah'] = np.nanmean(df_cyc['h_discharge_capacity'].iloc[0:5])
    df.loc[df['device_id'] == id, 'eol_efc'] = df_cyc['cumulative_capacity'].iloc[-1]/NOM_CAP
    df.loc[df['device_id'] == id, 'eol_rpt_capacity_ah'] = np.nanmin(df_rpt['h_discharge_capacity'].values)


