""" Utilities for running simulations."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.interpolate import interp1d
from src import cellsim

def run_sim(formation_type, 
            diff_ec=3.86e-20,
            diff_vc=1.58e-17,
            rho_ec=1.38,
            rho_vc=1.33,
            kappa_ec=0.081e-3,
            kappa_vc=0.049e-3,
            gamma_kappa=3000, 
            dt=1.0, 
            to_follow_current=False, 
            current_vec=np.nan, 
            include_flag=False,
            dsei_derating_cycling=1.0,
            to_boost=True,
            sei_reactions='all'):
    """
    Run simulation for a given formation type and target EC diffusivity

    Arguments:
    ----------
    formation_type: 'base', 'fast', 'fast+'
    diff_ec: target EC diffusivity, e.g., 4.2e-20 [m2/s]
    diff_vc: target VC diffusivity, e.g., 6.6e-18 [m2/s]
    include_flag: 0 : up to the end of first charge cycle
                  1 : up to formation cycling
                  2 : up to first RPT
                  3 : up to formation aging
                  4 : up to second RPT
                  5 : up to aging cycles
                  False : do nothing
    dsei_derating_cycling: factor to derate D_SEI during cycling, default is 1.0 (no derating)
    to_boost: if True, boost the SEI diffusivity during cycling, default is True
    sei_reactions: 'all', 'sei1', 'sei2', default is 'all'
                    'all': both SEI reactions are included
                    'sei1': only the first SEI reaction is included
                    'sei2': only the second SEI reaction is included
    Returns:
    ----------

    df_sim: DataFrame with simulation results
    """

    vmax = 4.2
    vmin = 3.0
    Icv = 2.5/20

    cell = cellsim.Cell()
    
    # Update cell parameters from default parameters if necessary
    cell.load_config('params/default.yaml')

    if not to_boost:
        cell.gamma_boost = 0
        cell.tau_boost = 0
        cell.tau_decay = 0

    if sei_reactions == 'sei1':
        cell.U_SEI2 = -1000
    elif sei_reactions == 'sei2':
        cell.U_SEI1 = -1000
    elif sei_reactions == 'all':
        pass
    else:
        raise ValueError(f"Invalid value for sei_reactions: {sei_reactions}")

    # Update SEI conductivities
    cell.kappa_SEI1 = kappa_ec
    cell.kappa_SEI2 = kappa_vc

    # Update SEI densities
    cell.rho_SEI1 = rho_ec
    cell.rho_SEI2 = rho_vc

    # Update EC diffusivity
    cell.D_SEI11 = diff_ec
    cell.D_SEI21 = diff_ec

    # Update VC diffusivity
    cell.D_SEI12 = diff_vc
    cell.D_SEI22 = diff_vc

    # Update the GAMMA_KAPPA parameter
    # Usage of this variable is deprecated since it references a Nernst-Einstein 
    # implementation which we found to be incorrect. It be removed.
    cell.GAMMA_KAPPA = gamma_kappa

    if formation_type == 'base':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k
        else:
            sim.run_rest(1, rest_time_hrs=0.5)
            if include_flag == 0:
                sim.run_chg_cccv(1, 2.5/10, 2.5/10, 4.2) # Charge to 4.2V with no CV
                return sim.get_results()
            else:
                sim.run_chg_cccv(2, 2.5/10, 2.5/20, 4.2) # Charge to 4.2V with CV
            sim.run_rest(2, rest_time_hrs=1/6) 
            sim.run_dch_cccv(2, -2.5/10, -2.5/10, 3.0)
            sim.run_rest(2, rest_time_hrs=1/6) 
            sim.run_chg_cccv(3, 2.5/10, 2.5/20, 4.2)
            sim.run_rest(3, rest_time_hrs=1/6) 
            sim.run_dch_cccv(3, -2.5/10, -2.5/10, 3.0)
            sim.run_rest(3, rest_time_hrs=1/6) 
            sim.run_chg_cccv(4, 2.5/10, 2.5/20, 4.2)
            sim.run_rest(4, rest_time_hrs=1/6) 
            sim.run_dch_cccv(4, -2.5/10, 0.05, 3.0)
            sim.run_rest(4, rest_time_hrs=1/6) 
            sim.run_dch_cccv(4, -2.5/10, 0.05, 3.0)
            sim.run_rest(4, rest_time_hrs=6)

            cn = 4
            # rest_before_aging_hrs = 20

    elif formation_type == 'fast':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k
        else:
            sim.run_rest(1, rest_time_hrs=0.5)
            sim.run_chg_cccv(2, 2.5, 2.5, 3.9)
            if include_flag == 0:
                sim.run_chg_cccv(2, 2.5/5, 2.5/5, 4.2) # Charge to 4.2V with no CV
                return sim.get_results()
            else:
                sim.run_chg_cccv(2, 2.5/5, 0.125, 4.2) # Charge to 4.2V with CV
            sim.run_rest(2, rest_time_hrs=1/6)
            sim.run_dch_cccv(2, -2.5/5, 2.5/5, 3.9)
            sim.run_rest(2, rest_time_hrs=1/6)
            sim.run_chg_cccv(3, 2.5/5, 0.125, 4.2)
            sim.run_rest(3, rest_time_hrs=1/6)
            sim.run_dch_cccv(3, -2.5/5, 2.5/5, 3.9)
            sim.run_rest(3, rest_time_hrs=1/6)
            sim.run_chg_cccv(4, 2.5/5, 0.125, 4.2)
            sim.run_rest(4, rest_time_hrs=1/6)
            sim.run_dch_cccv(4, -2.5/5, 2.5/5, 3.9)
            sim.run_rest(4, rest_time_hrs=1/6)
            sim.run_chg_cccv(5, 2.5/5, 0.125, 4.2)
            sim.run_rest(5, rest_time_hrs=1/6)
            sim.run_dch_cccv(5, -2.5/5, 2.5/5, 3.9)
            sim.run_rest(5, rest_time_hrs=1/6)
            sim.run_dch_cccv(5, -2.5/1, -0.05, 3.0)
            sim.run_rest(5, rest_time_hrs=3)

            # rest_before_aging_hrs = 72.5
            cn = 5

    elif formation_type == 'fast+':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k

        else:
            sim.run_rest(1, rest_time_hrs=0.5)
            sim.run_chg_cccv(2, 5.0, 5.0, 3.9)
            if include_flag == 0:
                sim.run_chg_cccv(2, 2.5, 2.5, 4.2) # Charge to 4.2V with no CV
                return sim.get_results()
            else:
                sim.run_chg_cccv(2, 2.5, 0.125, 4.2) # Charge to 4.2V with CV
            sim.run_rest(2, rest_time_hrs=1/6)
            sim.run_dch_cccv(2, -2.5, 2.5, 3.9)
            sim.run_rest(2, rest_time_hrs=1/6)
            sim.run_chg_cccv(3, 2.5, 0.125, 4.2)
            sim.run_rest(3, rest_time_hrs=1/6)
            sim.run_dch_cccv(3, -2.5, 2.5, 3.9)
            sim.run_rest(3, rest_time_hrs=1/6)
            sim.run_chg_cccv(4, 2.5, 0.125, 4.2)
            sim.run_rest(4, rest_time_hrs=1/6)
            sim.run_dch_cccv(4, -2.5, 2.5, 3.9)
            sim.run_rest(4, rest_time_hrs=1/6)
            sim.run_chg_cccv(5, 2.5, 2.5, 4.2)
            sim.run_rest(5, rest_time_hrs=1/6)
            sim.run_dch_cccv(5, -2.5, 2.5, 3.9)
            sim.run_rest(5, rest_time_hrs=1/6)
            sim.run_dch_cccv(5, -5.0, -0.05, 3.0)
            sim.run_rest(5, rest_time_hrs=3)

            # rest_before_aging_hrs = 81.2
            cn = 5

    if include_flag == 1: 
        return sim.get_results()

    # RPT
    sim.run_chg_cccv(cn + 1, +2.5/30, +2.5/30, 4.2) # Approximate HPPC Pulse Charge
    sim.run_dch_cccv(cn + 1, -2.5/20, +2.5/20, 3.0)
    sim.run_chg_cccv(cn + 2, +2.5/20, +2.5/20, 4.2)
    sim.run_rest(cn + 2, rest_time_hrs=12)
    sim.run_dch_cccv(cn + 2, -2.5/2, -2.5/2, 3.0)

    if include_flag == 2:
        return sim.get_results()

    # Formation Aging
    # Synchronize when the formation aging starts
    rest_before_aging_hrs = 174.7799 - (sim.curr_k * sim.dt) / 3600
    sim.run_rest(cn + 2, rest_time_hrs=rest_before_aging_hrs)
    sim.run_chg_cccv(cn + 3, 2.5/2, 0.05, 4.2)
    sim.run_rest(cn + 3, rest_time_hrs=13*24) # 13 days rest
    sim.run_dch_cccv(cn + 3, -2.5/2, -2.5/2, 3.0)
    sim.run_rest(cn + 3, rest_time_hrs=12) # 12 hours rest

    if include_flag == 3:
        return sim.get_results()
    cn = cn + 4

    # RPT
    sim.run_chg_cccv(cn , +2.5/30, +2.5/30, vmax) # Approximate HPPC Pulse Charge
    sim.run_dch_cccv(cn, -2.5/20, Icv, vmin)
    sim.run_chg_cccv(cn+1, +2.5/20, Icv, vmax)
    sim.run_rest(cn+1, rest_time_hrs=12)

    if include_flag == 4:
        return sim.get_results()

    # Cycling:
    cell.D_SEI11 = diff_ec * dsei_derating_cycling
    cell.D_SEI21 = diff_ec * dsei_derating_cycling
    cell.D_SEI12 = diff_vc * dsei_derating_cycling
    cell.D_SEI22 = diff_vc * dsei_derating_cycling

    cycles_to_next_rpt = 100
    for i in np.arange(cn + 2, cn + 2 + cycles_to_next_rpt + 1):
        sim.run_chg_cccv(i, 2.5, 0.125, vmax)
        sim.run_rest(i, rest_time_hrs=1/6)
        sim.run_dch_cccv(i, -2.5, -2.5, vmin)
        sim.run_rest(i, rest_time_hrs=1/6)

    # # RPT
    # sim.run_chg_cccv(111, +2.5/30, +2.5/30, vmax) # Approximate HPPC Pulse Charge
    # sim.run_dch_cccv(111, -2.5/20, Icv, vmin)
    # sim.run_chg_cccv(111, +2.5/20, Icv, vmax)
    # sim.run_rest(112, rest_time_hrs=12)
    # sim.run_dch_cccv(112, -2.5/2, -2.5/2, vmin)

    # # Cycling
    # for i in np.arange(113, 113 + cycles_to_rpt + 1):
    #     sim.run_chg_cccv(i, 2.5, 0.125, vmax)
    #     sim.run_rest(i, rest_time_hrs=1/6)
    #     sim.run_dch_cccv(i, -2.5, -2.5, vmin)
    #     sim.run_rest(i, rest_time_hrs=1/6)

    # # RPT
    # sim.run_chg_cccv(214, +2.5/30, +2.5/30, vmax) # Approximate HPPC Pulse Charge
    # sim.run_dch_cccv(214, -2.5/20, Icv, vmin)
    # sim.run_chg_cccv(215, +2.5/20, Icv, vmax)
    # sim.run_rest(    215, rest_time_hrs=12)
    # sim.run_dch_cccv(215, -2.5/2, -2.5/2, vmin)

    # # Cycling
    # for i in np.arange(216, 216 + cycles_to_rpt + 1):
    #     sim.run_chg_cccv(i, 2.5, 0.125, vmax)
    #     sim.run_rest(i, rest_time_hrs=1/6)
    #     sim.run_dch_cccv(i, -2.5, -2.5, vmin)
    #     sim.run_rest(i, rest_time_hrs=1/6)

    # # RPT
    # sim.run_chg_cccv(317, +2.5/30, +2.5/30, vmax) # Approximate HPPC Pulse Charge
    # sim.run_dch_cccv(317, -2.5/20, Icv, vmin)
    # sim.run_chg_cccv(318, +2.5/20, Icv, vmax)
    # sim.run_rest(    318, rest_time_hrs=12)
    # sim.run_dch_cccv(318, -2.5/2, -2.5/2, vmin)



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


def _plot_heatmaps_base(x_vec, y_vec, mat1, mat2, mat3, label,
                        x_label, y_label,
                        zmin=None, zmax=None,
                        xscale='linear', yscale='linear'):
    """
    Shared scaffold for plot_heatmaps_rho/diff/kappa.

    Returns (fig, ax1, ax2, ax3, X, Y, pcm1, pcm2, pcm3) so callers can
    add titles, annotations, and axis scales on top.
    """
    X, Y = np.meshgrid(x_vec, y_vec)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    axes = (ax1, ax2, ax3)

    for ax in axes:
        ax.set_xlim([min(x_vec), max(x_vec)])
        ax.set_ylim([min(y_vec), max(y_vec)])
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which='both', ls='-', alpha=0.2)

    cmap = cm.get_cmap('viridis_r', 20)

    pcms = []
    for ax, mat, title in zip(axes, [mat1, mat2, mat3],
                               ['Base Formation', 'Fast Formation', 'Fast+ Formation']):
        pcm = ax.pcolormesh(X, Y, mat, shading='auto', cmap=cmap)
        plt.colorbar(pcm, ax=ax, label=label, orientation='horizontal', pad=0.2)
        pcm.set_clim(zmin, zmax)
        ax.set_title(title)
        ax.grid(False)
        pcms.append(pcm)

    return fig, ax1, ax2, ax3, X, Y, pcms[0], pcms[1], pcms[2]


def _annotate_heatmap_panels(axes, mats, x_vec, y_vec,
                              annotation_coord, annotation_orientation):
    """Add reference-line + star marker to each panel of a heatmap trio."""
    for ax, mat in zip(axes, mats):
        if annotation_orientation == 'horizontal':
            i_ref = np.argmin(np.abs(y_vec - annotation_coord))
            j_ref = np.nanargmin(mat[i_ref, :])
            ax.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            j_ref = np.argmin(np.abs(x_vec - annotation_coord))
            i_ref = np.nanargmin(mat[:, j_ref])
            ax.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.plot(x_vec[j_ref], y_vec[i_ref], 'r*', markersize=10,
                label=f'({x_vec[j_ref]:.3g}, {y_vec[i_ref]:.3g})')


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

    fig, ax1, ax2, ax3, *_ = _plot_heatmaps_base(
        vc_vec, ec_vec, mat1, mat2, mat3, label,
        x_label=r'$\rho_{\mathrm{LVDC}}$ (g/cm$^3$)',
        y_label=r'$\rho_{\mathrm{LEDC}}$ (g/cm$^3$)',
        zmin=zmin, zmax=zmax,
    )

    if to_annotate:
        if annotation_coord is None:
            annotation_coord = 1.5
        _annotate_heatmap_panels(
            (ax1, ax2, ax3), (mat1, mat2, mat3),
            vc_vec, ec_vec, annotation_coord, annotation_orientation,
        )

    for ax in (ax1, ax2, ax3):
        ax.legend(facecolor='lightgray', framealpha=1.0, frameon=True, fontsize=10)

    plt.tight_layout()

    if tosave:
        plt.savefig(savename, bbox_inches='tight', format='svg')


def plot_heatmaps_diff(diff_ec_vec, diff_vc_vec,
                       mat1, mat2, mat3, label,
                       zmin=None,
                       zmax=None,
                       tosave=False,
                       to_annotate=False,
                       annotation_orientation='vertical',
                       annotation_coord=None,
                       savename='temp.svg'):

    fig, ax1, ax2, ax3, *_ = _plot_heatmaps_base(
        diff_vc_vec, diff_ec_vec, mat1, mat2, mat3, label,
        x_label=r'$D_{\mathrm{LVDC}}$ (m²/s)',
        y_label=r'$D_{\mathrm{LEDC}}$ (m²/s)',
        zmin=zmin, zmax=zmax,
        xscale='log', yscale='log',
    )

    if to_annotate:
        if annotation_coord is None:
            annotation_coord = 1.3e-17
        _annotate_heatmap_panels(
            (ax1, ax2, ax3), (mat1, mat2, mat3),
            diff_vc_vec, diff_ec_vec, annotation_coord, annotation_orientation,
        )

    for ax in (ax1, ax2, ax3):
        ax.legend(facecolor='lightgray', framealpha=1.0, frameon=True, fontsize=10)

    plt.tight_layout()

    if tosave:
        plt.savefig(savename, bbox_inches='tight', format='svg')


def plot_heatmaps_kappa(diff_ec_vec, diff_vc_vec,
                        mat1, mat2, mat3, label,
                        zmin=None,
                        zmax=None,
                        tosave=False,
                        savename='temp.svg'):

    fig, ax1, ax2, ax3, *_ = _plot_heatmaps_base(
        diff_vc_vec, diff_ec_vec, mat1, mat2, mat3, label,
        x_label=r'$\kappa_{\mathrm{LVDC}}$ (mS/m)',
        y_label=r'$\kappa_{\mathrm{LEDC}}$ (mS/m)',
        zmin=zmin, zmax=zmax,
    )

    # Fixed reference line at kappa_vc = 0.05 mS/m
    ref_vc = 0.05
    j1_ref = np.argmin(np.abs(diff_vc_vec - ref_vc))
    for ax, mat in zip((ax1, ax2, ax3), (mat1, mat2, mat3)):
        i_ref = np.nanargmin(mat[:, j1_ref])
        ax.axvline(x=ref_vc, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.plot(diff_vc_vec[j1_ref], diff_ec_vec[i_ref], 'bo', markersize=14)
        ax.plot(diff_vc_vec[j1_ref], diff_ec_vec[i_ref], 'r*', markersize=15,
                label=f'({diff_vc_vec[j1_ref]:.2g}, {diff_ec_vec[i_ref]:.2g})')
        ax.legend()

    plt.tight_layout()

    if tosave:
        plt.savefig(savename, bbox_inches='tight', format='svg')




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