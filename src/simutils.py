""" Utilities for running simulations."""

from src import cellsim as cellsim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def run_sim(type, 
            diff_ec, 
            diff_vc, 
            kappa_ec=4.771e-4,
            kappa_vc=2.05e-5,
            gamma_kappa=3000, 
            dt=5.0, 
            to_follow_current=False, 
            current_vec=np.nan, 
            include_flag=False,
            dsei_derating_cycling=1.0):
    """
    Run simulation for a given formation type and target EC diffusivity

    Arguments:
    ----------
    type: 'base', 'fast', 'fast+'
    diff_ec: target EC diffusivity, e.g., 4.2e-20 [m2/s]
    diff_vc: target VC diffusivity, e.g., 6.6e-18 [m2/s]
    include_flag: 1 : up to formation cycling
                  2 : up to first RPT
                  3 : up to formation aging
                  4 : up to second RPT
                  5 : up to aging cycles
                  False : do nothing
    dsei_derating_cycling: factor to derate D_SEI during cycling, default is 1.0 (no derating)

    Returns:
    ----------

    df_sim: DataFrame with simulation results
    """

    vmax = 4.2
    vmin = 3.0
    Icv = 2.5/20

    cell = cellsim.Cell()
    cell.load_config('params/default.yaml')

    # Update SEI conductivities
    cell.kappa_SEI1 = kappa_ec
    cell.kappa_SEI2 = kappa_vc

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

    if type == 'base':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k 
        else: 
            sim.run_rest(1, rest_time_hrs=0.5)
            sim.run_chg_cccv(2, 2.5/10, 2.5/20, 4.2)
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

    elif type == 'fast':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k
        else:
            sim.run_rest(1, rest_time_hrs=0.5)
            sim.run_chg_cccv(2, 2.5, 2.5, 3.9)
            sim.run_chg_cccv(2, 2.5/5, 0.125, 4.2)
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

    elif type == 'fast+':

        sim = cellsim.Simulation(cell, 1000*3600, dt=dt)

        if to_follow_current:
            for k, current in enumerate(current_vec):
                sim.step(k, mode='cc', icc=current, to_debug=False)
                sim.curr_k = k
        
        else:
            sim.run_rest(1, rest_time_hrs=0.5)
            sim.run_chg_cccv(2, 5.0, 5.0, 3.9)
            sim.run_chg_cccv(2, 2.5, 0.125, 4.2)
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


def calculate_rmse(t_meas, y_meas, t_modl, y_modl,
                   to_plot=False, y_range=None):
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
    y_range: range of y values to include in the RMSE calculation
    to_plot: if True, plots the model and measured data

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


def plot_heatmaps(diff_ec_vec, diff_vc_vec, mat1, mat2, mat3, label, 
                    is_error_plot=True, 
                    zmin=None, 
                    zmax=None, 
                    tosave=False, 
                    savename='temp.svg', 
                    to_annotate=False,
                    to_show_tuned=True, 
                    is_kappa=False):

    # Reference diffusivities from Weng2023
    diff_ec_ow = 4.2e-20  # Baseline d_ec from Weng2023
    diff_vc_ow = 6.6e-18  # Baseline d_vc from Weng2023

    # Global / reference from optimizing expansion RMSE from baseline formation
    # diff_ec_o = 3.831e-20
    # diff_vc_o = 2.154e-17

    if is_kappa:
        diff_vc_o = 2.05e-5*1000
        diff_ec_o = 4.771e-4*1000
    else:
        diff_vc_o = 4.217e-16 # old set
        diff_ec_o = 3.162e-20 # old set
        diff_vc_o = 3.162e-17 # new set (after kappa optimization)
        diff_ec_o = 3.665e-20 # new set (after kappa optimization)

    # diff_ec_o = 3.162e-20  # Baseline d_ec from the latest study
    # diff_vc_o = 4.217e-16  # Baseline d_vc from the latest study

    #  Create a meshgrid for X and Y axes
    X, Y = np.meshgrid(diff_vc_vec, diff_ec_vec)

    # Create subplots for the three formation protocols
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_xlim([min(diff_vc_vec), max(diff_vc_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_ylim([min(diff_ec_vec), max(diff_ec_vec)]) for ax in (ax1, ax2, ax3)]
    if not is_kappa:
        [ax.set_xscale('log') for ax in (ax1, ax2, ax3)]
        [ax.set_yscale('log') for ax in (ax1, ax2, ax3)]
    if is_kappa:
        [ax.set_xlabel('$\kappa_{\mathrm{LVDC}}$ (mS/m)') for ax in (ax1, ax2, ax3)]
        [ax.set_ylabel('$\kappa_{\mathrm{LEDC}}$ (mS/m)') for ax in (ax1, ax2, ax3)]
    else:
        [ax.set_xlabel('$D_{\mathrm{LVDC}}$ (m²/s)') for ax in (ax1, ax2, ax3)]
        [ax.set_ylabel('$D_{\mathrm{LEDC}}$ (m²/s)') for ax in (ax1, ax2, ax3)]
    [ax.grid(True, which="both", ls="-", alpha=0.2) for ax in (ax1, ax2, ax3)]


    # Get the colormap and create a discrete version with 10 bins
    cmap = cm.get_cmap('viridis_r', 20)

    # Base Formation
    pcm1 = ax1.pcolormesh(X, Y, mat1, shading='auto', cmap=cmap)
    vmin, vmax = np.nanmin(mat1), np.nanmax(mat1)
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

    if is_error_plot:
        i, j = np.unravel_index(np.nanargmin(mat1), mat1.shape)

        
        # ax1.plot(diff_vc_ow, diff_ec_ow, 'b*', markersize=14)

        if to_show_tuned:

            ax1.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=16,
             label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')

            ax1.annotate(
                'Tuned',
                (diff_vc_vec[j], diff_ec_vec[i]),
                textcoords="offset points",
                xytext=(10, 10),
                ha='left',
                color='k',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
            )
    
        ax1.plot(diff_vc_o, diff_ec_o, 'b*', markersize=14)
        ax1.annotate('Ref', (diff_vc_o, diff_ec_o),
            textcoords="offset points",
            xytext=(-10, -10),                 
            ha='right',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )

        # ax1.annotate('Ref', (diff_vc_ow, diff_ec_ow),
        #     textcoords="offset points",
        #     xytext=(-10, 10),                 
        #     ha='right',
        #     color='k',
        #     fontsize=12,
        #     fontweight='bold',
        #     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        # )

        if to_annotate: 
            ax1.annotate(f'({diff_vc_vec[j]:.3e},\n {diff_ec_vec[i]:.3e})',
                     (diff_vc_vec[j], diff_ec_vec[i]),
                     textcoords="offset points",
                     xytext=(10, -10),
                     ha='left',
                     color='red',
                     fontsize=8)

        i, j = np.unravel_index(np.nanargmin(mat2), mat2.shape)
        
        ax2.plot(diff_vc_o, diff_ec_o, 'g*', markersize=15)


        if to_show_tuned: 

            ax2.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=15,
            label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')
        
            ax2.annotate(
                'Tuned',
                (diff_vc_vec[j], diff_ec_vec[i]),
                textcoords="offset points",
                xytext=(10, -10),
                ha='left',
                color='k',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
            )
            if to_annotate:
                ax2.annotate(f'({diff_vc_vec[j]:.1e},\n {diff_ec_vec[i]:.1e})',
                        (diff_vc_vec[j], diff_ec_vec[i]),
                        textcoords="offset points",
                        xytext=(10, -10),
                        ha='left',
                        color='red',
                        fontsize=8) 
    
        ax2.plot(diff_vc_o, diff_ec_o, 'b*', markersize=14)

        ax2.annotate('Ref', (diff_vc_o, diff_ec_o),
                 textcoords="offset points",
                 xytext=(-10, -10),
                 ha='right',
                 color='k',
                 fontsize=12,
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )

        i, j = np.unravel_index(np.nanargmin(mat3), mat3.shape)

        if to_show_tuned:
                
            ax3.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=15,
             label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')
       
       
            ax3.annotate(
                'Tuned',
                (diff_vc_vec[j], diff_ec_vec[i]),
                textcoords="offset points",
                xytext=(10, 10),
                ha='left',
                color='k',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
            )

            if to_annotate:
                ax3.annotate(f'({diff_vc_vec[j]:.3e},\n {diff_ec_vec[i]:.3e})',
                        (diff_vc_vec[j], diff_ec_vec[i]),
                        textcoords="offset points",
                        xytext=(10, -10),
                        ha='left',
                        color='red',
                        fontsize=8)
    
        ax3.plot(diff_vc_o, diff_ec_o, 'b*', markersize=15)

        ax3.annotate('Ref', (diff_vc_o, diff_ec_o),
            textcoords="offset points",
            xytext=(-10, -10),                 
            ha='right',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )

    plt.tight_layout()

    if tosave:
        plt.savefig(savename, bbox_inches='tight', format='svg')


def plot_diffusivity_heatmaps2(diff_ec_vec, diff_vc_vec, mat1, mat2, mat3, label, is_error_plot=True, 
                              zmin=None, zmax=None, tosave=False, savename='temp.svg', to_annotate=False,
                              is_log_scale=True):

    diff_ec_ow = 4.2e-20  # Baseline d_ec from Weng2023
    diff_vc_ow = 6.6e-18  # Baseline d_vc from Weng2023

    diff_ec_o = 3.162e-20  # Baseline d_ec from the latest study
    diff_vc_o = 4.217e-16  # Baseline d_vc from the latest study

    #  Create a meshgrid for X and Y axes
    X, Y = np.meshgrid(diff_vc_vec, diff_ec_vec)

    # Create subplots for the three formation protocols
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
    [ax.set_xlim([min(diff_vc_vec), max(diff_vc_vec)]) for ax in (ax1, ax2, ax3)]
    [ax.set_ylim([min(diff_ec_vec), max(diff_ec_vec)]) for ax in (ax1, ax2, ax3)]
    if is_log_scale:
        [ax.set_xscale('log') for ax in (ax1, ax2, ax3)]
        [ax.set_yscale('log') for ax in (ax1, ax2, ax3)]
    [ax.set_xlabel('$D_{LVDC}^0$ (m²/s)') for ax in (ax1, ax2, ax3)]
    [ax.set_ylabel('$D_{LEDC}^0$ (m²/s)') for ax in (ax1, ax2, ax3)]
    [ax.grid(True, which="both", ls="-", alpha=0.2) for ax in (ax1, ax2, ax3)]


    # Get the colormap and create a discrete version with 10 bins
    cmap = cm.get_cmap('viridis_r', 20)

    # Base Formation
    pcm1 = ax1.pcolormesh(X, Y, mat1, shading='auto', cmap=cmap)
    vmin, vmax = np.nanmin(mat1), np.nanmax(mat1)
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

    if is_error_plot:
        i, j = np.unravel_index(np.nanargmin(mat1), mat1.shape)
        ax1.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=16,
             label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')

        
        ax1.plot(diff_vc_o, diff_ec_o, 'g*', markersize=14)
        ax1.plot(diff_vc_ow, diff_ec_ow, 'b*', markersize=14)

        ax1.annotate(
            'Tuned',
            (diff_vc_vec[j], diff_ec_vec[i]),
            textcoords="offset points",
            xytext=(0, -20),
            ha='center',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )
    
        ax1.annotate('Global', (diff_vc_o, diff_ec_o),
            textcoords="offset points",
            xytext=(10, 10),                 
            ha='left',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )

        ax1.annotate('Ref', (diff_vc_ow, diff_ec_ow),
            textcoords="offset points",
            xytext=(-10, 10),                 
            ha='right',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )
        
        # ax1.annotate(f'({diff_vc_vec[j]:.3e},\n {diff_ec_vec[i]:.3e})',
        #              (diff_vc_vec[j], diff_ec_vec[i]),
        #              textcoords="offset points",
        #              xytext=(10, -10),
        #              ha='left',
        #              color='red',
        #              fontsize=8)

        i, j = np.unravel_index(np.nanargmin(mat2), mat2.shape)
        ax2.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=15,
             label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')
        ax2.plot(diff_vc_o, diff_ec_o, 'g*', markersize=15)
        # ax2.annotate(f'({diff_vc_vec[j]:.1e},\n {diff_ec_vec[i]:.1e})',
        #              (diff_vc_vec[j], diff_ec_vec[i]),
        #              textcoords="offset points",
        #              xytext=(10, -10),
        #              ha='left',
        #              color='red',
        #              fontsize=8) 
        
        ax2.plot(diff_vc_o, diff_ec_o, 'g*', markersize=14)

        
        ax2.annotate(
            'Tuned',
            (diff_vc_vec[j], diff_ec_vec[i]),
            textcoords="offset points",
            xytext=(-10, -10),
            ha='right',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )
    
        ax2.annotate('Global', (diff_vc_o, diff_ec_o),
                 textcoords="offset points",
                 xytext=(10, 10),
                 ha='left',
                 color='k',
                 fontsize=12,
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )

        i, j = np.unravel_index(np.nanargmin(mat3), mat3.shape)
        ax3.plot(diff_vc_vec[j], diff_ec_vec[i], 'r*', markersize=15,
             label=f'Min RMSE at\nD_EC={diff_ec_vec[i]:.2e}\nD_VC={diff_vc_vec[j]:.2e}')
        ax3.plot(diff_vc_o, diff_ec_o, 'g*', markersize=15)
        # ax3.annotate(f'({diff_vc_vec[j]:.1e},\n {diff_ec_vec[i]:.1e})',
                    #  (diff_vc_vec[j], diff_ec_vec[i]),
                    #  textcoords="offset points",
                    #  xytext=(10, -10),
                    #  ha='left',
                    #  color='red',
                    #  fontsize=8)

        ax3.annotate(
            'Tuned',
            (diff_vc_vec[j], diff_ec_vec[i]),
            textcoords="offset points",
            xytext=(-10, -10),
            ha='right',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )
    
        ax3.annotate('Global', (diff_vc_o, diff_ec_o),
            textcoords="offset points",
            xytext=(10, 10),                 
            ha='left',
            color='k',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
        )
        

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