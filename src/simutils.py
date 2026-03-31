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

    #  Create a meshgrid for X and Y axes
    X, Y = np.meshgrid(vc_vec, ec_vec)

    # Create subplots for the three formation protocols
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_xlim([min(vc_vec), max(vc_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_ylim([min(ec_vec), max(ec_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_xlabel(r'$\rho_{\mathrm{LVDC}}$ (g/cm$^3$)') for ax in (ax1, ax2, ax3)]
    [ax.set_ylabel(r'$\rho_{\mathrm{LEDC}}$ (g/cm$^3$)') for ax in (ax1, ax2, ax3)]
    [ax.grid(True, which="both", ls="-", alpha=0.2) for ax in (ax1, ax2, ax3)]

    # Get the colormap and create a discrete version with 10 bins
    cmap = cm.get_cmap('viridis_r', 20)

    # Base Formation
    pcm1 = ax1.pcolormesh(X, Y, mat1, shading='auto', cmap=cmap)
    plt.colorbar(pcm1, ax=ax1, label=label, orientation='horizontal', pad=0.2)
    pcm1.set_clim(zmin, zmax)
    i, j = np.unravel_index(np.nanargmin(mat1), mat1.shape)
    ax1.set_title('Base Formation')
    ax1.grid(False)

    # Fast Formation
    pcm2 = ax2.pcolormesh(X, Y, mat2, shading='auto', cmap=cmap)
    plt.colorbar(pcm2, ax=ax2, label=label, orientation='horizontal', pad=0.2)
    pcm2.set_clim(zmin, zmax)
    ax2.set_title('Fast Formation')
    ax2.grid(False)

    # Fast+ Formation
    pcm3 = ax3.pcolormesh(X, Y, mat3, shading='auto', cmap=cmap)
    plt.colorbar(pcm3, ax=ax3, label=label, orientation='horizontal', pad=0.2)
    pcm3.set_clim(zmin, zmax)
    ax3.set_title('Fast+ Formation')
    ax3.grid(False)


    # Add the annotation markers and reference line, similar to plot_heatmaps_diff
    if to_annotate:

        # Default to previous hard-coded behavior for backward compatibility
        if annotation_coord is None:
            annotation_coord = 1.5  # g/cm^3 reference LVDC density

        # Base Formation
        if annotation_orientation == 'horizontal':
            # Fix y (ec) at annotation_coord
            i1_ref = np.argmin(np.abs(ec_vec - annotation_coord))
            j1_ref = np.nanargmin(mat1[i1_ref, :])
            ax1.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            # Default: vertical line in vc
            j1_ref = np.argmin(np.abs(vc_vec - annotation_coord))
            i1_ref = np.nanargmin(mat1[:, j1_ref])
            ax1.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax1.plot(vc_vec[j1_ref], ec_vec[i1_ref], 'r*', markersize=10,
                 label=f'({vc_vec[j1_ref]:.3g}, {ec_vec[i1_ref]:.3g})')

        # Fast Formation
        if annotation_orientation == 'horizontal':
            i2_ref = np.argmin(np.abs(ec_vec - annotation_coord))
            j2_ref = np.nanargmin(mat2[i2_ref, :])
            ax2.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            j2_ref = np.argmin(np.abs(vc_vec - annotation_coord))
            i2_ref = np.nanargmin(mat2[:, j2_ref])
            ax2.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax2.plot(vc_vec[j2_ref], ec_vec[i2_ref], 'r*', markersize=10,
                 label=f'({vc_vec[j2_ref]:.3g}, {ec_vec[i2_ref]:.3g})')

        # Fast+ Formation
        if annotation_orientation == 'horizontal':
            i3_ref = np.argmin(np.abs(ec_vec - annotation_coord))
            j3_ref = np.nanargmin(mat3[i3_ref, :])
            ax3.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            j3_ref = np.argmin(np.abs(vc_vec - annotation_coord))
            i3_ref = np.nanargmin(mat3[:, j3_ref])
            ax3.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax3.plot(vc_vec[j3_ref], ec_vec[i3_ref], 'r*', markersize=10,
                 label=f'({vc_vec[j3_ref]:.3g}, {ec_vec[i3_ref]:.3g})')

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

    #  Create a meshgrid for X and Y axes
    X, Y = np.meshgrid(diff_vc_vec, diff_ec_vec)

    # Create subplots for the three formation protocols
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_xlim([min(diff_vc_vec), max(diff_vc_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_ylim([min(diff_ec_vec), max(diff_ec_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_xscale('log') for ax in (ax1, ax2, ax3)]
    [ax.set_yscale('log') for ax in (ax1, ax2, ax3)]
    [ax.set_xlabel(r'$D_{\mathrm{LVDC}}$ (m²/s)') for ax in (ax1, ax2, ax3)]
    [ax.set_ylabel(r'$D_{\mathrm{LEDC}}$ (m²/s)') for ax in (ax1, ax2, ax3)]
    [ax.grid(True, which="both", ls="-", alpha=0.2) for ax in (ax1, ax2, ax3)]


    # Get the colormap and create a discrete version with 10 bins
    cmap = cm.get_cmap('viridis_r', 20)

    # Base Formation
    pcm1 = ax1.pcolormesh(X, Y, mat1, shading='auto', cmap=cmap)
    plt.colorbar(pcm1, ax=ax1, label=label, orientation='horizontal', pad=0.2)
    pcm1.set_clim(zmin, zmax)
    i, j = np.unravel_index(np.nanargmin(mat1), mat1.shape)
    ax1.set_title('Base Formation')
    ax1.grid(False)

    # Fast Formation
    pcm2 = ax2.pcolormesh(X, Y, mat2, shading='auto', cmap=cmap)
    plt.colorbar(pcm2, ax=ax2, label=label, orientation='horizontal', pad=0.2)
    pcm2.set_clim(zmin, zmax)
    ax2.set_title('Fast Formation')
    ax2.grid(False)

    # Fast+ Formation
    pcm3 = ax3.pcolormesh(X, Y, mat3, shading='auto', cmap=cmap)
    plt.colorbar(pcm3, ax=ax3, label=label, orientation='horizontal', pad=0.2)
    pcm3.set_clim(zmin, zmax)
    ax3.set_title('Fast+ Formation')
    ax3.grid(False)

    if to_annotate:

        # Default to previous hard-coded behavior for backward compatibility
        if annotation_coord is None:
            annotation_coord = 1.3e-17

        i1, j1 = np.unravel_index(np.nanargmin(mat1), mat1.shape)

        if annotation_orientation == 'horizontal':
            # Fix y at annotation_coord
            i1_ref = np.argmin(np.abs(diff_ec_vec - annotation_coord))
            j1_ref = np.nanargmin(mat1[i1_ref, :])
            ax1.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            # Default: vertical line in x
            j1_ref = np.argmin(np.abs(diff_vc_vec - annotation_coord))
            i1_ref = np.nanargmin(mat1[:, j1_ref])
            ax1.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax1.plot(diff_vc_vec[j1_ref], diff_ec_vec[i1_ref], 'r*', markersize=10,
        label=f'({diff_vc_vec[j1_ref]:.3g}, {diff_ec_vec[i1_ref]:.3g})')
        # ax1.plot(diff_vc_vec[j1], diff_ec_vec[i1], 'r*', markersize=16,
        # label=f'Min RMSE at\nD_EC={diff_ec_vec[i1]:.3e}\nD_VC={diff_vc_vec[j1]:.3e}')

        i2, j2 = np.unravel_index(np.nanargmin(mat2), mat2.shape)

        if annotation_orientation == 'horizontal':
            i2_ref = np.argmin(np.abs(diff_ec_vec - annotation_coord))
            j2_ref = np.nanargmin(mat2[i2_ref, :])
            ax2.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            j2_ref = np.argmin(np.abs(diff_vc_vec - annotation_coord))
            i2_ref = np.nanargmin(mat2[:, j2_ref])
            ax2.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax2.plot(diff_vc_vec[j2_ref], diff_ec_vec[i2_ref], 'r*', markersize=10,
        label=f'({diff_vc_vec[j2_ref]:.3g}, {diff_ec_vec[i2_ref]:.3g})')
        # ax2.plot(diff_vc_vec[j2], diff_ec_vec[i2], 'r*', markersize=16,
        # label=f'Min RMSE at\nD_EC={diff_ec_vec[i2]:.3e}\nD_VC={diff_vc_vec[j2]:.3e}')
            
        i3, j3 = np.unravel_index(np.nanargmin(mat3), mat3.shape)

        if annotation_orientation == 'horizontal':
            i3_ref = np.argmin(np.abs(diff_ec_vec - annotation_coord))
            j3_ref = np.nanargmin(mat3[i3_ref, :])
            ax3.axhline(y=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)
        else:
            j3_ref = np.argmin(np.abs(diff_vc_vec - annotation_coord))
            i3_ref = np.nanargmin(mat3[:, j3_ref])
            ax3.axvline(x=annotation_coord, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax3.plot(diff_vc_vec[j3_ref], diff_ec_vec[i3_ref], 'r*', markersize=10,
        label=f'({diff_vc_vec[j3_ref]:.3g}, {diff_ec_vec[i3_ref]:.3g})')
        # ax3.plot(diff_vc_vec[j3], diff_ec_vec[i3], 'r*', markersize=16,
        # label=f'Min RMSE at\nD_EC={diff_ec_vec[i3]:.3e}\nD_VC={diff_vc_vec[j3]:.3e}')

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

    #  Create a meshgrid for X and Y axes
    X, Y = np.meshgrid(diff_vc_vec, diff_ec_vec)

    # Create subplots for the three formation protocols
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_xlim([min(diff_vc_vec), max(diff_vc_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_ylim([min(diff_ec_vec), max(diff_ec_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_xlabel(r'$\kappa_{\mathrm{LVDC}}$ (mS/m)') for ax in (ax1, ax2, ax3)]
    [ax.set_ylabel(r'$\kappa_{\mathrm{LEDC}}$ (mS/m)') for ax in (ax1, ax2, ax3)]

    [ax.grid(True, which="both", ls="-", alpha=0.2) for ax in (ax1, ax2, ax3)]


    # Get the colormap and create a discrete version with 10 bins
    cmap = cm.get_cmap('viridis_r', 20)

    # Base Formation
    pcm1 = ax1.pcolormesh(X, Y, mat1, shading='auto', cmap=cmap)
    plt.colorbar(pcm1, ax=ax1, label=label, orientation='horizontal', pad=0.2)
    pcm1.set_clim(zmin, zmax)
    i, j = np.unravel_index(np.nanargmin(mat1), mat1.shape)
    ax1.set_title('Base Formation')

    # Fast Formation
    pcm2 = ax2.pcolormesh(X, Y, mat2, shading='auto', cmap=cmap)
    plt.colorbar(pcm2, ax=ax2, label=label, orientation='horizontal', pad=0.2)
    pcm2.set_clim(zmin, zmax)
    ax2.set_title('Fast Formation')

    # Fast+ Formation
    pcm3 = ax3.pcolormesh(X, Y, mat3, shading='auto', cmap=cmap)
    plt.colorbar(pcm3, ax=ax3, label=label, orientation='horizontal', pad=0.2)
    pcm3.set_clim(zmin, zmax)
    ax3.set_title('Fast+ Formation')

    i1, j1 = np.unravel_index(np.nanargmin(mat1), mat1.shape)
    j1_ref = np.argmin(np.abs(diff_vc_vec - 0.05))
    i1_ref = np.nanargmin(mat1[:, j1_ref])

    ax1.axvline(x=0.05, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.plot(diff_vc_vec[j1_ref], diff_ec_vec[i1_ref], 'bo', markersize=14)
    ax1.plot(diff_vc_vec[j1_ref], diff_ec_vec[i1_ref], 'r*', markersize=16,
             label=f'({diff_vc_vec[j1_ref]:.2g}, {diff_ec_vec[i1_ref]:.2g})')

    # ax1.annotate(
    #             'Tuned',
    #             (diff_vc_vec[j1], diff_ec_vec[i1]),
    #             textcoords="offset points",
    #             xytext=(10, 10),
    #             ha='left',
    #             color='k',
    #             fontsize=12,
    #             fontweight='bold',
    #             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
    #         )
    
    i2, j2 = np.unravel_index(np.nanargmin(mat2), mat2.shape) 
    j2_ref = np.argmin(np.abs(diff_vc_vec - 0.05))
    i2_ref = np.nanargmin(mat2[:, j2_ref])
    ax2.axvline(x=0.05, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.plot(diff_vc_vec[j1_ref], diff_ec_vec[i1_ref], 'bo', markersize=14)
    ax2.plot(diff_vc_vec[j2_ref], diff_ec_vec[i2_ref], 'r*', markersize=15,
                label=f'({diff_vc_vec[j2_ref]:.2g}, {diff_ec_vec[i2_ref]:.2g})')

    # ax2.annotate(
    #     'Tuned',
    #     (diff_vc_vec[j], diff_ec_vec[i]),
    #     textcoords="offset points",
    #     xytext=(10, -10),
    #     ha='left',
    #     color='k',
    #     fontsize=12,
    #     fontweight='bold',
    #     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
    # )
            
    i3, j3 = np.unravel_index(np.nanargmin(mat3), mat3.shape)
    j3_ref = np.argmin(np.abs(diff_vc_vec - 0.05))
    i3_ref = np.nanargmin(mat3[:, j3_ref])
    ax3.axvline(x=0.05, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.plot(diff_vc_vec[j1_ref], diff_ec_vec[i1_ref], 'bo', markersize=14)
    ax3.plot(diff_vc_vec[j3_ref], diff_ec_vec[i3_ref], 'r*', markersize=15,
             label=f'({diff_vc_vec[j3_ref]:.2g}, {diff_ec_vec[i3_ref]:.2g})')

    # [ax.grid(False) for ax in (ax1, ax2, ax3)] 
    [ax.legend() for ax in (ax1, ax2, ax3)]
    # ax3.annotate(
    #         'Tuned',
    #         (diff_vc_vec[j], diff_ec_vec[i]),
    #         textcoords="offset points",
    #         xytext=(10, 10),
    #         ha='left',
    #         color='k',
    #         fontsize=12,
    #         fontweight='bold',
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
    #             )

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