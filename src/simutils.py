""" Utilities for running simulations."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from src import cellsim, heatmaps

PHASE_PRE_FORMATION = 0
PHASE_FIRST_CHARGE = 1
PHASE_FORMATION_CYCLING = 2
PHASE_RPT1 = 3
PHASE_FORMATION_AGING = 4
PHASE_RPT2 = 5
PHASE_CYCLING = 6

VMAX = 4.2
VMIN = 3.0
I_CV = 2.5/20

# Formation protocols as step lists. Each step is one of
#   ('rest', cycle, hours)
#   ('chg', cycle, I_cc, I_cv, V_max)     CCCV charge
#   ('dch', cycle, I_cc, I_cv, V_min)     CCCV discharge (negative currents)
#
# first_charge   runs at the start of the first charge
# fc_end_no_cv   ends the first charge when include_flag == 0 (no CV hold)
# fc_end         ends the first charge otherwise
# cycling        the rest of formation, through the final rest
# last_cycle     cycle number of the last formation cycle
FORMATION_PROTOCOLS = {
    'base': dict(
        first_charge=[],
        fc_end_no_cv=('chg', 1, 2.5/10, 2.5/10, 4.2),
        fc_end=('chg', 2, 2.5/10, 2.5/20, 4.2),
        cycling=[
            ('rest', 2, 1/6), ('dch', 2, -2.5/10, -2.5/10, 3.0), ('rest', 2, 1/6),
            ('chg', 3, 2.5/10, 2.5/20, 4.2), ('rest', 3, 1/6),
            ('dch', 3, -2.5/10, -2.5/10, 3.0), ('rest', 3, 1/6),
            ('chg', 4, 2.5/10, 2.5/20, 4.2), ('rest', 4, 1/6),
            ('dch', 4, -2.5/10, 0.05, 3.0), ('rest', 4, 1/6),
            ('dch', 4, -2.5/10, 0.05, 3.0), ('rest', 4, 6),
        ],
        last_cycle=4,
    ),
    'fast': dict(
        first_charge=[('chg', 2, 2.5, 2.5, 3.9)],
        fc_end_no_cv=('chg', 2, 2.5/5, 2.5/5, 4.2),
        fc_end=('chg', 2, 2.5/5, 0.125, 4.2),
        cycling=[
            ('rest', 2, 1/6), ('dch', 2, -2.5/5, 2.5/5, 3.9), ('rest', 2, 1/6),
            ('chg', 3, 2.5/5, 0.125, 4.2), ('rest', 3, 1/6),
            ('dch', 3, -2.5/5, 2.5/5, 3.9), ('rest', 3, 1/6),
            ('chg', 4, 2.5/5, 0.125, 4.2), ('rest', 4, 1/6),
            ('dch', 4, -2.5/5, 2.5/5, 3.9), ('rest', 4, 1/6),
            ('chg', 5, 2.5/5, 0.125, 4.2), ('rest', 5, 1/6),
            ('dch', 5, -2.5/5, 2.5/5, 3.9), ('rest', 5, 1/6),
            ('dch', 5, -2.5/1, -0.05, 3.0), ('rest', 5, 3),
        ],
        last_cycle=5,
    ),
    'fast+': dict(
        first_charge=[('chg', 2, 5.0, 5.0, 3.9)],
        fc_end_no_cv=('chg', 2, 2.5, 2.5, 4.2),
        fc_end=('chg', 2, 2.5, 0.125, 4.2),
        cycling=[
            ('rest', 2, 1/6), ('dch', 2, -2.5, 2.5, 3.9), ('rest', 2, 1/6),
            ('chg', 3, 2.5, 0.125, 4.2), ('rest', 3, 1/6),
            ('dch', 3, -2.5, 2.5, 3.9), ('rest', 3, 1/6),
            ('chg', 4, 2.5, 0.125, 4.2), ('rest', 4, 1/6),
            ('dch', 4, -2.5, 2.5, 3.9), ('rest', 4, 1/6),
            ('chg', 5, 2.5, 2.5, 4.2), ('rest', 5, 1/6),
            ('dch', 5, -2.5, 2.5, 3.9), ('rest', 5, 1/6),
            ('dch', 5, -5.0, -0.05, 3.0), ('rest', 5, 3),
        ],
        last_cycle=5,
    ),
}

# Formation aging starts at this test time [h] for every protocol
FORMATION_AGING_START_HRS = 174.7799

# Aging cycles simulated after the second RPT
NUM_AGING_CYCLES = 101


def _run_steps(sim, steps):
    for kind, cycle, *args in steps:
        if kind == 'rest':
            sim.run_rest(cycle, rest_time_hrs=args[0])
        elif kind == 'chg':
            sim.run_chg_cccv(cycle, *args)
        elif kind == 'dch':
            sim.run_dch_cccv(cycle, *args)
        else:
            raise ValueError(f'Unknown step kind {kind!r}')


def _set_phase(sim, phase):
    sim.phase[sim.curr_k:] = phase


def make_cell(diff_ec=3.86e-20, diff_vc=1.58e-17,
              rho_ec=1.38, rho_vc=1.33,
              kappa_ec=0.081e-3, kappa_vc=0.049e-3,
              to_boost=True, sei_reactions='all'):
    """Cell from params/default.yaml with the given SEI parameters.

    The EC (LEDC) and VC (LVDC) diffusivities are applied to layer 1 and
    layer 2 of both SEI species, respectively.
    """
    cell = cellsim.Cell()
    cell.load_config('params/default.yaml')

    if not to_boost:
        cell.gamma_boost = 0
        cell.tau_boost = 0
        cell.tau_decay = 0

    if sei_reactions == 'sei1':
        cell.U_SEI2 = -1000
    elif sei_reactions == 'sei2':
        cell.U_SEI1 = -1000
    elif sei_reactions != 'all':
        raise ValueError(f"Invalid value for sei_reactions: {sei_reactions}")

    cell.kappa_SEI1 = kappa_ec
    cell.kappa_SEI2 = kappa_vc
    cell.rho_SEI1 = rho_ec
    cell.rho_SEI2 = rho_vc
    cell.D_SEI11 = diff_ec
    cell.D_SEI21 = diff_ec
    cell.D_SEI12 = diff_vc
    cell.D_SEI22 = diff_vc
    return cell


def run_sim(formation_type,
            diff_ec=3.86e-20,
            diff_vc=1.58e-17,
            rho_ec=1.38,
            rho_vc=1.33,
            kappa_ec=0.081e-3,
            kappa_vc=0.049e-3,
            dt=1.0,
            to_follow_current=False,
            current_vec=np.nan,
            include_flag=False,
            dsei_derating_cycling=1.0,
            to_boost=True,
            sei_reactions='all'):
    """
    Simulate a formation protocol, optionally followed by RPTs, formation
    aging and aging cycles.

    Arguments:
    ----------
    formation_type: 'base', 'fast', 'fast+' (see FORMATION_PROTOCOLS)
    diff_ec: EC (LEDC) diffusivity, e.g., 4.2e-20 [m2/s]
    diff_vc: VC (LVDC) diffusivity, e.g., 6.6e-18 [m2/s]
    rho_ec, rho_vc: SEI densities [g/cm3]
    kappa_ec, kappa_vc: SEI conductivities [S/m]
    dt: time step [s]
    to_follow_current: if True, apply current_vec (one value per time step)
                       instead of the formation protocol; requires include_flag=1
    include_flag: 0 : up to the end of first charge cycle (no CV hold)
                  1 : up to formation cycling
                  2 : up to first RPT
                  3 : up to formation aging
                  4 : up to second RPT
                  5 : up to aging cycles
                  False (default) : same as 0
    dsei_derating_cycling: factor to derate D_SEI during the aging cycles,
                           default is 1.0 (no derating)
    to_boost: if True, boost the SEI diffusivity during cycling, default is True
    sei_reactions: 'all', 'sei1', 'sei2', default is 'all'
                    'all': both SEI reactions are included
                    'sei1': only the first SEI reaction is included
                    'sei2': only the second SEI reaction is included
    Returns:
    ----------

    df_sim: DataFrame with simulation results
    """
    if formation_type not in FORMATION_PROTOCOLS:
        raise ValueError(f'Unknown formation_type {formation_type!r}; '
                         f'expected one of {list(FORMATION_PROTOCOLS)}')

    cell = make_cell(diff_ec, diff_vc, rho_ec, rho_vc, kappa_ec, kappa_vc,
                     to_boost=to_boost, sei_reactions=sei_reactions)
    sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

    if to_follow_current:
        for k, current in enumerate(current_vec):
            sim.step(k, mode='cc', icc=current, to_debug=False)
            sim.curr_k = k
        if include_flag != 1:
            raise ValueError('to_follow_current requires include_flag=1')
        return sim.get_results()

    # Formation
    protocol = FORMATION_PROTOCOLS[formation_type]
    _set_phase(sim, PHASE_PRE_FORMATION)
    sim.run_rest(1, rest_time_hrs=0.5)
    _set_phase(sim, PHASE_FIRST_CHARGE)
    _run_steps(sim, protocol['first_charge'])
    if include_flag == 0:
        _run_steps(sim, [protocol['fc_end_no_cv']])
        return sim.get_results()
    _run_steps(sim, [protocol['fc_end']])
    _set_phase(sim, PHASE_FORMATION_CYCLING)
    _run_steps(sim, protocol['cycling'])
    if include_flag == 1:
        return sim.get_results()

    # RPT after formation
    cn = protocol['last_cycle']
    _set_phase(sim, PHASE_RPT1)
    _run_steps(sim, [
        ('chg', cn + 1, +2.5/30, +2.5/30, 4.2),    # approximate HPPC pulse charge
        ('dch', cn + 1, -2.5/20, +2.5/20, 3.0),
        ('chg', cn + 2, +2.5/20, +2.5/20, 4.2),
        ('rest', cn + 2, 12),
        ('dch', cn + 2, -2.5/2, -2.5/2, 3.0),
    ])
    if include_flag == 2:
        return sim.get_results()

    # Formation aging, starting at the same test time for every protocol
    _set_phase(sim, PHASE_FORMATION_AGING)
    rest_before_aging_hrs = FORMATION_AGING_START_HRS - (sim.curr_k * sim.dt) / 3600
    _run_steps(sim, [
        ('rest', cn + 2, rest_before_aging_hrs),
        ('chg', cn + 3, 2.5/2, 0.05, 4.2),
        ('rest', cn + 3, 13*24),                   # 13 days
        ('dch', cn + 3, -2.5/2, -2.5/2, 3.0),
        ('rest', cn + 3, 12),
    ])
    if include_flag == 3:
        return sim.get_results()

    # RPT after formation aging
    cn = cn + 4
    _set_phase(sim, PHASE_RPT2)
    _run_steps(sim, [
        ('chg', cn, +2.5/30, +2.5/30, VMAX),       # approximate HPPC pulse charge
        ('dch', cn, -2.5/20, I_CV, VMIN),
        ('chg', cn + 1, +2.5/20, I_CV, VMAX),
        ('rest', cn + 1, 12),
    ])
    if include_flag == 4:
        return sim.get_results()

    # Aging cycles
    _set_phase(sim, PHASE_CYCLING)
    cell.D_SEI11 = diff_ec * dsei_derating_cycling
    cell.D_SEI21 = diff_ec * dsei_derating_cycling
    cell.D_SEI12 = diff_vc * dsei_derating_cycling
    cell.D_SEI22 = diff_vc * dsei_derating_cycling
    for i in np.arange(cn + 2, cn + 2 + NUM_AGING_CYCLES):
        _run_steps(sim, [
            ('chg', i, 2.5, 0.125, VMAX),
            ('rest', i, 1/6),
            ('dch', i, -2.5, -2.5, VMIN),
            ('rest', i, 1/6),
        ])

    return sim.get_results()


def calculate_rmse(t_meas, y_meas, 
                   t_modl, y_modl,
                   to_plot=False, 
                   y_range=None,
                   ignore_time_range=None):
    """
    Calculates RMSE error between model and measured data.

    Does not assume measurement vectors are the same size.

    Interpolates the measured data along the model vector

    Parameters:
    ----------
    t_meas: time vector for measured data
    y_meas: measured data
    t_modl: time vector for model data
    y_modl: model data
    to_plot: if True, plots the model and measured data
    y_range: range of y values to include in the RMSE calculation
    ignore_time_range: time range (seconds) to ignore in the RMSE calculation

    Returns:
    ----------
    rmse: root mean square error between model and measured data
    t_modl: filtered/trimmed model time vector (after y_range / ignore_time_range)
    pointwise_error: per-point sqrt(squared_error) array
    """

    # Create interpolation function for voltage data
    f = interp1d(t_meas, y_meas, bounds_error=False, fill_value=np.nan)

    # Interpolate voltage onto model time points
    y_meas_interp = f(t_modl)

    # Exclude data beyond specified y_range
    if y_range is not None:
        mask = (y_modl >= y_range[0]) & (y_modl <= y_range[1])
        t_modl = t_modl[mask]
        y_modl = y_modl[mask]
        y_meas_interp = y_meas_interp[mask]

    # Exclude data within the ignored time range
    if ignore_time_range is not None:
        mask = ~( (t_modl >= ignore_time_range[0]) & (t_modl <= ignore_time_range[1]) )
        t_modl = t_modl[mask]
        y_modl = y_modl[mask]
        y_meas_interp = y_meas_interp[mask]

    squared_error = (y_modl - y_meas_interp)**2

    # Compute RMSE between model and measured voltage
    rmse = np.sqrt(np.nanmean(squared_error))

    if to_plot:

        plt.figure()
        plt.plot(t_modl, y_modl, label='Model')
        plt.plot(t_meas, y_meas, label='Measurement')
        plt.plot(t_modl, y_meas_interp, label='Interpolated Measurement', linestyle='--')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return rmse, t_modl, np.sqrt(squared_error)


def _plot_heatmaps(x_vec, y_vec, mats, label, x_label, y_label, zmin, zmax, scale,
                   tosave, savename, to_annotate, annotation_coord, orientation,
                   line_only=False, line_color='darkred', label_line=False):
    fig, axes = heatmaps.heatmap_trio(x_vec, y_vec, mats, label, x_label, y_label,
                                      zlim=(zmin, zmax), xscale=scale, yscale=scale)
    if to_annotate:
        heatmaps.annotate_panels(axes, mats, x_vec, y_vec, annotation_coord, orientation,
                                 line_only=line_only, line_color=line_color,
                                 label_line=label_line)
    heatmaps.add_legends(axes)

    plt.tight_layout()

    if tosave:
        plt.savefig(savename, bbox_inches='tight', format='svg')


def plot_heatmaps_rho(ec_vec, vc_vec,
                      mat1, mat2, mat3,
                      label,
                      zmin=None,
                      zmax=None,
                      tosave=False,
                      to_annotate=False,
                      annotation_orientation='vertical',
                      annotation_coord=None,
                      savename='temp.svg'):
    """Heatmaps over rho_LVDC (x) and rho_LEDC (y) for the three protocols.

    Matrices are indexed [rho_EC, rho_VC]. With to_annotate, marks
    annotation_coord (default 1.5) with a gray line and a star at the minimum
    along it; annotation_orientation is 'vertical' or 'horizontal'.
    """
    _plot_heatmaps(
        vc_vec, ec_vec, (mat1, mat2, mat3), label,
        r'$\rho_{\mathrm{LVDC}}$ (g/cm$^3$)', r'$\rho_{\mathrm{LEDC}}$ (g/cm$^3$)',
        zmin, zmax, 'linear', tosave, savename, to_annotate,
        1.5 if annotation_coord is None else annotation_coord,
        'horizontal' if annotation_orientation == 'horizontal' else 'vertical',
        line_color='darkgray')


def plot_heatmaps_diff(diff_ec_vec, diff_vc_vec,
                       mat1, mat2, mat3, label,
                       zmin=None,
                       zmax=None,
                       tosave=False,
                       to_annotate=False,
                       annotation_orientation='vertical',
                       annotation_coord=None,
                       annotation_line_only=False,
                       savename='temp.svg'):
    """Heatmaps over D_LVDC (x) and D_LEDC (y), log scale, for the three protocols.

    Matrices are indexed [D_EC, D_VC]. With to_annotate, marks
    annotation_coord (default 1.3e-17; one value or one per panel) with a line
    and a star at the minimum along it. annotation_orientation is 'vertical',
    'horizontal', or 'minimum' (star at the global minimum, no line);
    annotation_line_only draws the line without the star.
    """
    _plot_heatmaps(
        diff_vc_vec, diff_ec_vec, (mat1, mat2, mat3), label,
        r'$D_{\mathrm{LVDC}}$ (m²/s)', r'$D_{\mathrm{LEDC}}$ (m²/s)',
        zmin, zmax, 'log', tosave, savename, to_annotate,
        1.3e-17 if annotation_coord is None else annotation_coord,
        annotation_orientation, line_only=annotation_line_only, label_line=True)


def plot_heatmaps_kappa(kappa_ec_vec, kappa_vc_vec,
                        mat1, mat2, mat3, label,
                        zmin=None,
                        zmax=None,
                        tosave=False,
                        to_annotate=False,
                        annotation_orientation='vertical',
                        annotation_coord=None,
                        annotation_line_only=False,
                        savename='temp.svg'):
    """Heatmaps over kappa_LVDC (x) and kappa_LEDC (y), in mS/m, for the three protocols.

    Pass the conductivities already in mS/m. Annotation options are as in
    plot_heatmaps_diff (default annotation_coord 0.05).
    """
    _plot_heatmaps(
        kappa_vc_vec, kappa_ec_vec, (mat1, mat2, mat3), label,
        r'$\kappa_{\mathrm{LVDC}}$ (mS/m)', r'$\kappa_{\mathrm{LEDC}}$ (mS/m)',
        zmin, zmax, 'linear', tosave, savename, to_annotate,
        0.05 if annotation_coord is None else annotation_coord,
        annotation_orientation, line_only=annotation_line_only, label_line=True)


def interpolate_raw_data(t, y, dt):
    """
    Interpolates raw data to a specified time step.
    Parameters:
    ----------
    t: time vector
    y: data vector
    dt: time step for interpolation
    Returns:
    ----------
    t_interp: interpolated time vector
    y_interp: interpolated data vector
    """
    # Create interpolation function for voltage data

    t_interp = np.arange(t.min(), t.max(), dt)
    y_interp = np.interp(t_interp, t, y)

    return t_interp, y_interp