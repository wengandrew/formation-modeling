"""
Classes for the formation modeling work

Model components:

Thermodynamics  : eSOH
Kinetics        : R RC
SEI growth      : Butler-Volmer (Yang2017)
Expansion       : ?
"""

import plotter as plotter
import modelutils as mu
import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt

plotter.initialize(plt, style='ieee')

F = 96485.33212    # C/mol      Faraday's constant
T = 273.15 + 25    # Kelvin     Temperature
R = 8.314          # J/mol/K    Universal gas constant

STEP_NUM_CHARGE_CC = 0
STEP_NUM_CHARGE_CV = 1
STEP_NUM_DISCHARGE_CC = 2
STEP_NUM_DISCHARGE_CV = 4
STEP_NUM_REST = 4

class Cell:

    def __init__(self, name=''):
        """
        Initialize a cell object.

        Parameters
        ---------
        name (str): name of the cell (optional)
        config (dict): the parameters defining this cell
        """

        self.name = name

        # Initialize the OCP functions
        self.Un = mu.Un
        self.Up = mu.Up

        # Initialize the expansion functions
        self.En = mu.En
        self.Ep = mu.Ep


    def load_config(self, file_path: str):
        """
        Load a configuration from file
        """

        with open(f"{file_path}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize cell parameters based on config file
        for (key, value) in config.items():
            setattr(self, key, value)


    def get_tag(self):

        tag = f'k={self.k_SEI1}, '\
              f'D={self.D_SEI11}, '\
              f'U={self.U_SEI1}, '\
              f'Rn={self.R0n}, '\
              f'Rp={self.R0p}'

        return tag



class Simulation:

    def __init__(self, cell: Cell, sim_time_s: int, name='DefaultSim'):
        """
        Parameters
        ----------
        cell:       a Cell object
        sim_time_s: simulation time in seconds
        name (str): A string name for the simulation
        """

        self.name = name
        self.cell = cell

        # Numerical details
        self.dt = 5.0
        self.t = np.arange(0, sim_time_s, self.dt)

        # Track where we are in the simulation
        self.curr_k = 0

        # Initialize output vectors
        self.i_app = np.zeros(self.t.shape)

        self.cycle_number = np.zeros(self.t.shape)
        self.step_number = np.zeros(self.t.shape)

        # eSOH states
        self.theta_n = mu.initialize(self.t, cell.theta_n)
        self.theta_p = mu.initialize(self.t, cell.theta_p)
        self.ocv_n   = mu.initialize(self.t, cell.Un(cell.theta_n))
        self.ocv_p   = mu.initialize(self.t, cell.Up(cell.theta_p))
        self.eta_n   = mu.initialize(self.t, 0)
        self.eta_p   = mu.initialize(self.t, 0)
        self.ocv     = mu.initialize(self.t, self.ocv_p[0] - self.ocv_n[0])
        self.vt      = mu.initialize(self.t, self.ocv_p[0] - self.ocv_n[0])
        self.i_int   = mu.initialize(self.t, 0)

        # RC states
        self.i_r1p   = mu.initialize(self.t, 0)
        self.i_r1n   = mu.initialize(self.t, 0)

        # SEI states

        # Total quantities
        self.j_sei   = mu.initialize(self.t, 0)
        self.i_sei   = mu.initialize(self.t, 0)
        self.q_sei   = mu.initialize(self.t, 0)

        # Homogenized quantities
        self.D_sei1   = mu.initialize(self.t, 1/(1/cell.D_SEI11 + 1/cell.D_SEI12))
        self.D_sei2   = mu.initialize(self.t, 1/(1/cell.D_SEI21 + 1/cell.D_SEI22))

        # Individual components
        self.j_sei1     = mu.initialize(self.t, 0)
        self.j_sei2     = mu.initialize(self.t, 0)
        self.c_sei1     = mu.initialize(self.t, cell.c_SEI1_0)
        self.c_sei2     = mu.initialize(self.t, cell.c_SEI2_0)
        self.i_sei1     = mu.initialize(self.t, 0)
        self.i_sei2     = mu.initialize(self.t, 0)
        self.q_sei1     = mu.initialize(self.t, 0)
        self.q_sei2     = mu.initialize(self.t, 0)
        self.j_sei_rxn1 = mu.initialize(self.t, np.NaN)
        self.j_sei_rxn2 = mu.initialize(self.t, np.NaN)
        self.j_sei_dif1 = mu.initialize(self.t, np.NaN)
        self.j_sei_dif2 = mu.initialize(self.t, np.NaN)
        self.eta_sei1   = mu.initialize(self.t, 0)
        self.eta_sei2   = mu.initialize(self.t, 0)

        # Expansion states
        self.delta_sei  = mu.initialize(self.t, cell.delta_SEI_0)
        self.delta_sei1 = mu.initialize(self.t, cell.delta_SEI_0/2)
        self.delta_sei2 = mu.initialize(self.t, cell.delta_SEI_0/2)
        self.delta_n    = mu.initialize(self.t, cell.En(cell.theta_n))
        self.delta_p    = mu.initialize(self.t, cell.Ep(cell.theta_p))
        self.dndt       = mu.initialize(self.t, 0)
        self.boost      = mu.initialize(self.t, 0)
        self.expansion_rev   = mu.initialize(self.t, 0)
        self.expansion_irrev = mu.initialize(self.t, 0)


    def step(self, k: int, mode: str, icc=0, icv=0,
             cyc_num=np.NaN, step_num=np.NaN,
             vcv=0, to_debug=False):
        """
        Run a single step.

        Parameters
        ----------
        k:         current time index
        mode:      'cc' or 'cv'
        icc:       applied current in CC mode
        icv:       current cut-off in CV mode
        cyc_num:   cycle number to associate with this step
        step_num:  step number to associate with this step
        vcv:       voltage for the CV hold
        to_debug:  if true then print some stuff
        """

        p = self.cell

        self.cycle_number[k] = cyc_num
        self.step_number[k] = step_num

        if mode == 'cc':
            self.i_app[k] = icc
        elif mode == 'cv':
            self.i_app[k] = ( self.vt[k] - self.ocv[k] - \
                           p.R1p * np.exp(-self.dt/(p.R1p*p.C1p)) * self.i_r1p[k] - \
                           p.R1n * np.exp(-self.dt/(p.R1n*p.C1n)) * self.i_r1n[k] ) / \
                        ( ( 1 - np.exp(-self.dt/(p.R1n*p.C1n)) ) * p.R1n + p.R0n + \
                          ( 1 - np.exp(-self.dt/(p.R1p*p.C1p)) ) * p.R1p + p.R0p )


        dQint = self.i_int[k] * self.dt / 3600 # Amp-hours
        dQapp = self.i_app[k] * self.dt / 3600 # Amp-hours

        # assert dQint > 0, 'How are we decreasing theta_n during lithiation?'

        self.theta_n[k + 1] = self.theta_n[k] + dQint / p.Cn
        self.theta_p[k + 1] = self.theta_p[k] - dQapp / p.Cp

        # Equilibrium potential updates
        self.ocv_n[k + 1] = self.cell.Un(self.theta_n[k + 1])
        self.ocv_p[k + 1] = self.cell.Up(self.theta_p[k + 1])
        self.delta_n[k + 1] = self.cell.En(self.theta_n[k + 1])
        self.delta_p[k + 1] = self.cell.Ep(self.theta_p[k + 1])

        self.ocv[k + 1] = self.ocv_p[k+1] - self.ocv_n[k+1]

        # Current updates (branch current for RC element)
        self.i_r1p[k+1] =      np.exp(-self.dt/(p.R1p*p.C1p))  * self.i_r1p[k] + \
                          (1 - np.exp(-self.dt/(p.R1p*p.C1p))) * self.i_app[k]

        self.i_r1n[k+1] =      np.exp(-self.dt/(p.R1n*p.C1n))  * self.i_r1n[k] + \
                          (1 - np.exp(-self.dt/(p.R1n*p.C1n))) * self.i_app[k]

        self.eta_p[k+1] = p.R1p * self.i_r1p[k] + p.R0p * self.i_app[k]
        self.eta_n[k+1] = p.R1n * self.i_r1n[k] + p.R0n * self.i_int[k]

        # Terminal voltage update
        if mode == 'cc':
            self.vt[k+1] = self.ocv_p[k+1] + self.eta_p[k] - \
                          (self.ocv_n[k+1] - self.eta_n[k])
        elif mode == 'cv':
            self.vt[k+1] = vcv

        # SEI growth update
        self.eta_sei1[k+1] = self.eta_n[k+1] + self.ocv_n[k+1] - p.U_SEI1
        self.eta_sei2[k+1] = self.eta_n[k+1] + self.ocv_n[k+1] - p.U_SEI2

        # Mixed reaction and diffusion limited SEI current density
        # Boosted SEI reaction during cycling
        self.dndt[k+1] = (self.delta_n[k+1] - self.delta_n[k]) / self.dt

        if self.i_app[k] > 0: # charging / lithiating graphite
            self.boost[k+1] = self.boost[k] + self.dt / p.tau_boost * \
                                 (self.dndt[k+1] * p.gamma_boost - self.boost[k])
        else: # discharging or resting
            self.boost[k+1] = self.boost[k] - self.dt * self.boost[k] / p.tau_decay

        # SEI limiting current densities
        self.j_sei_rxn1[k+1] = p.n_SEI1 * F * self.c_sei1[k] \
                                * p.k_SEI1 \
                                * np.exp( -p.alpha_SEI * p.n_SEI1 * \
                                         F * self.eta_sei1[k+1] / (R * T) )

        self.j_sei_dif1[k+1] = self.D_sei1[k] \
                                * (1 + self.boost[k+1]) \
                                * self.c_sei1[k] * p.n_SEI1 * F \
                                / (self.delta_sei[k]) # should this be k or k+1?

        self.j_sei_rxn2[k+1] = p.n_SEI2 * F * self.c_sei2[k] \
                                * p.k_SEI2 \
                                * np.exp( -p.alpha_SEI * p.n_SEI2 * \
                                         F * self.eta_sei2[k+1] / (R * T) )

        self.j_sei_dif2[k+1] = self.D_sei2[k] \
                                * (1 + self.boost[k+1]) \
                                * self.c_sei2[k] * p.n_SEI2 * F \
                                / (self.delta_sei[k]) # should this be k or k+1?

        # SEI current density
        self.j_sei1[k+1] = 1 / (1/self.j_sei_rxn1[k+1] + 1/self.j_sei_dif1[k+1])
        self.j_sei2[k+1] = 1 / (1/self.j_sei_rxn2[k+1] + 1/self.j_sei_dif2[k+1])

        self.j_sei[k+1] = self.j_sei1[k+1] + self.j_sei2[k+1]

        ## Current density to current conversion
        self.i_sei[k+1]  = self.j_sei[k+1]  * (p.a_sn * p.A_n * p.L_n)
        self.i_sei1[k+1] = self.j_sei1[k+1] * (p.a_sn * p.A_n * p.L_n)
        self.i_sei2[k+1] = self.j_sei2[k+1] * (p.a_sn * p.A_n * p.L_n)

        # Update SEI reacting species concentrations
        self.c_sei1[k+1] = self.c_sei1[k] - self.dt * \
                            (p.a_sn * self.j_sei1[k+1] / (p.n_SEI1 * F))

        self.c_sei2[k+1] = self.c_sei2[k] - self.dt * \
                            (p.a_sn * self.j_sei2[k+1] / (p.n_SEI2 * F))

        # Update the intercalation current
        self.i_int[k+1] = self.i_app[k]  - self.i_sei[k+1]

        if to_debug:
            print(
                f'k: {k} | '\
                f'Iint: {self.i_int[k]:.3f} | '\
                f'Iapp: {self.i_app[k]:.3f} | '\
                f'Isei: {self.i_sei[k]:.3f} | '\
                f'Irxn: {self.j_sei_rxn1[k]:.3e} | '\
                f'Idif: {self.j_sei_dif1[k]:.3e} | '\
                f'dndt: {self.dndt[k]:.3e} | '\
                f'dn: {self.delta_n[k]:.6f}'
                )

        # Integrate SEI current to get SEI capacity
        self.q_sei1[k+1] = self.q_sei1[k] + self.i_sei1[k+1] * self.dt / 3600
        self.q_sei2[k+1] = self.q_sei2[k] + self.i_sei2[k+1] * self.dt / 3600
        self.q_sei[k+1]  = self.q_sei[k]  + self.i_sei[k+1] * self.dt / 3600

        # Update homogenized variables
        mu1 = self.q_sei1[k+1] / (self.q_sei[k+1])
        mu2 = self.q_sei2[k+1] / (self.q_sei[k+1])
        self.D_sei1[k+1] = 1 / (mu1/p.D_SEI11 + mu2/p.D_SEI12)
        self.D_sei2[k+1] = 1 / (mu1/p.D_SEI21 + mu2/p.D_SEI22)

        # Update SEI thickness
        self.delta_sei1[k+1] = self.delta_sei1[k] + \
                              self.dt * (p.V_SEI1 * \
                                        np.abs(self.j_sei1[k+1]) ) / (p.n_SEI1 * F)

        self.delta_sei2[k+1] = self.delta_sei2[k] + \
                              self.dt * (p.V_SEI2 * \
                                        np.abs(self.j_sei2[k+1]) ) / (p.n_SEI2 * F)

        self.delta_sei[k+1] = self.delta_sei1[k+1] + self.delta_sei2[k+1]

        # Expansion update
        # Cathode and anode expansion function update
        self.expansion_rev[k+1] = p.c1 * self.delta_p[k+1] + \
                                  p.c2 * self.delta_n[k+1]
        self.expansion_irrev[k+1] = p.c0 * self.delta_sei[k+1]


    def run_rest(self, cycle_number: int, rest_time_hrs: float):
        """
        Run a rest step

        Parameters
        ----------
        cycle_number (int): a cycle number label
        rest_time_hrs (float): hours to rest
        """

        print(f'Running Cyc{cycle_number}: Rest for {rest_time_hrs} hours...')
        k = self.curr_k

        kmax = k + int(rest_time_hrs*3600 / self.dt)

        assert kmax < len(self.t) - 1, 'Ran out of time array.'

        while k < kmax:
            self.step(k, 'cc',
                      icc=0,
                      cyc_num=cycle_number,
                      step_num=STEP_NUM_REST)
            k += 1

        self.curr_k = k


    def run_chg_cccv(self, cycle_number: int,
                     icc: float, icv: float, vmax: float):
        """
        Run a charge CCCV step.

        Parameters:
        -----------
        cycle_number (int): a cycle number label
        icc (float): CC current in amps (+ve is charge)
        icc (float): CV current cutoff in amps (+ve is charge)
        vmax (float): charge voltage target
        """

        print(f'Running Cyc{cycle_number}: Charge to {vmax}V...')

        assert icc >= icv, 'CV hold current cannot be bigger than CC current.'

        k = self.curr_k

        mode = 'cc'

        while True:

            if mode == 'cc':

                self.step(k, 'cc', icc=icc,
                                   cyc_num=cycle_number,
                                   step_num=STEP_NUM_CHARGE_CC,
                                   vcv=vmax)

                if self.vt[k+1] >= vmax:
                    mode = 'cv'
                    self.vt[k+1] = vmax

                k += 1

            if mode == 'cv' and np.abs(icc) >= np.abs(icv):

                self.step(k, 'cv', icv=icv,
                                   cyc_num=cycle_number,
                                   step_num=STEP_NUM_CHARGE_CV,
                                   vcv=vmax)

                # End condition
                if np.abs(self.i_app[k]) < np.abs(icv):
                    self.curr_k = k
                    break

                k += 1


    def run_dch_cccv(self, cycle_number: int,
                     icc: float, icv: float, vmin: float):
        """
        Run a discharge CCCV step.

        Parameters:
        -----------
        cycle_number (int): a cycle number label
        icc (float): CC current in amps (+ve is charge)
        icc (float): CV current cutoff in amps (+ve is charge)
        vmin (float): discharge voltage target
        """

        print(f'Running Cyc{cycle_number}: Discharge to {vmin}V...')

        k = self.curr_k

        mode = 'cc'

        while True:

            if mode == 'cc':

                self.step(k, 'cc', icc=icc,
                                   cyc_num=cycle_number,
                                   step_num=STEP_NUM_DISCHARGE_CC,
                                   vcv=vmin)

                if self.vt[k+1] <= vmin:
                    mode = 'cv'
                    self.vt[k+1] = vmin

                k += 1

            if mode == 'cv' and np.abs(icc) >= np.abs(icv):

                self.step(k, 'cv', icv=icv,
                                   cyc_num=cycle_number,
                                   step_num=STEP_NUM_DISCHARGE_CV,
                                   vcv=vmin)

                # End condition
                if np.abs(self.i_app[k]) < np.abs(icv):
                    self.curr_k = k
                    break

                k += 1


    def get_results(self) -> pd.DataFrame:
        """
        Return the simulation results in a DataFrame.

        Append the DataFrame with some derived results
        """

        df = pd.DataFrame(self.__dict__)

        # Incremental capacities corresponding to the total current and SEI
        # currents
        df['dq'] = np.abs(df['dt'] * df['i_app'] / 3600)
        df['dqsei'] = np.abs(df['dt'] * df['i_sei'] / 3600)
        df['dqsei1'] = np.abs(df['dt'] * df['i_sei1'] / 3600)
        df['dqsei2'] = np.abs(df['dt'] * df['i_sei2'] / 3600)

        # Get rid of extra NaN rows beyond the simulation time
        df = df.iloc[0:self.curr_k, :]

        return df


    def plot_view_1(self, to_save=False, xlims=None):
        """
        Make "View 1"

        Focus on voltage, current, and expansion
        """

        num_subplots = 3

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(8, num_subplots * 4.5),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        xx = self.t/3600

        if xlims is not None:
            axs[0].set_xlim(xlims)

        # Voltages and Potentials
        i = 0
        axs[i].plot(xx, self.ocv_p + self.eta_p, ls='-', c='b', label='$U^+ + \eta^+$')
        axs[i].plot(xx, self.vt, ls='-', c='k', label='$V_t$')
        # axs[i].plot(xx, self.ocv, ls='--', c='k')
        axs[i].set_ylabel('Voltage/Potential [V]')
        # axs[i].plot(xx, self.ocv_p, ls='--', c='b', label='$U_p$')
        # axs[i].plot(xx, self.ocv_n, ls='--', c='r', label='$U_n$')
        axs[i].plot(xx, self.ocv_n - self.eta_n, ls='-', c='r', label='$U^- - \eta^-$')
        # axs[i].axhline(y=self.cell.U_SEI2, ls='--', c='m', label=rf'$U_{{\mathrm{{SEI,B}}}}$ = {self.cell.U_SEI2} V')
        axs[i].axhline(y=self.cell.U_SEI2, ls='--', c='m', label='$U_{\mathrm{sei,1}}$ = 1.35V')
        axs[i].axhline(y=self.cell.U_SEI1, ls='--', c='c', label='$U_{\mathrm{sei,2}}$ = 0.80V')
        axs[i].text(1.5, 0.9, 'SEI 2 (EC)', fontsize=20, fontweight='bold', color='c')
        axs[i].text(1.5, 1.45, 'SEI 1 (VC)', fontsize=20, fontweight='bold', color='m')
        axs[i].set_ylim((0.0, 4.0))
        # axs[i].axhline(y=self.cell.U_SEI1, ls='--', c='c', label=rf'$U_{{\mathrm{{SEI,A}}}}$ = {self.cell.U_SEI1} V')
        lh = axs[i].legend(loc='right', fancybox=False, frameon=True, fontsize=18)
        lh.get_frame().set_alpha(None)


        # Currents
        i += 1
        axs[i].axhline(y=0, ls='-', label='', c='k', lw=0.5)
        axs[i].plot(xx, self.i_app, c='k', label=r'$I_{\mathrm{app}}$')
        axs[i].plot(np.NaN, np.NaN, c='k', lw=2, ls='--', label=r'$I_{\mathrm{sei}}$')
        axs[i].plot(xx, self.i_sei2, c='m', ls='-', label=r'$I_{\mathrm{sei,1}}$')
        axs[i].plot(xx, self.i_sei1, c='c', ls='-', label=r'$I_{\mathrm{sei,2}}$')
        axs[i].plot(xx, self.i_sei, c='k', lw=2, ls='--', label='')
        axs[i].legend(loc='center right', fancybox=False, frameon=False, fontsize=18)
        axs[i].text(1, 0.17, 'SEI 1', fontsize=20, fontweight='bold', color='m')
        axs[i].text(1.85, 0.05, 'SEI 2', fontsize=20, fontweight='bold', color='c')
        axs[i].set_ylabel('Current [A]')
        axs[i].set_ylim((0, 0.27))

        # Total cell expansion
        i += 1
        ff = self.delta_sei2 / (self.delta_sei2 + self.delta_sei1)
        axs[i].plot(xx, (self.expansion_rev + self.expansion_irrev)*1e6, c='k', label=r'$\Delta_{\mathrm{sei,1}}$ + $\Delta_{\mathrm{sei,2}}$ + $\Delta_{\mathrm{rev}}$')
        axs[i].plot(xx, self.expansion_irrev*1e6, c='c', ls='-', label=r'$\Delta_{\mathrm{sei,1}}$ + $\Delta_{\mathrm{sei,2}}$')
        axs[i].plot(xx, ff*self.expansion_irrev*1e6, c='m', ls='-', label=r'$\Delta_{\mathrm{sei,1}}$')
        axs[i].legend(loc='upper left', fontsize=18)
        axs[i].set_ylabel(r'Thickness [$\mu$m]')
        axs[i].set_xlabel('Time [hrs]')
        axs[i].set_ylim((-5, 75))

        axs[i].text(4, -0, 'SEI 1', fontsize=20, fontweight='bold', color='m')
        axs[i].text(4, 19, 'SEI 2', fontsize=20, fontweight='bold', color='c')
        axs[i].text(4, 43, 'Reversible', fontsize=20, fontweight='bold', color='k')

        # Markers
        bbox_dict = dict(boxstyle='circle', pad=0.1, facecolor='none', edgecolor='k')
        axs[0].text(0.25, 0.30, '1', fontsize=16, bbox=bbox_dict)
        axs[1].text(1.02, 0.22, '2', fontsize=16, bbox=bbox_dict)
        axs[1].text(1.42, 0.11, '3', fontsize=16, bbox=bbox_dict)
        axs[2].text(4.40, 64.0, '4', fontsize=16, bbox=bbox_dict)
        axs[1].text(4.40, 0.023, '5', fontsize=16, bbox=bbox_dict)

        # Panel Numbers
        axs[0].text(-0.16, 0.92, 'A', transform=axs[0].transAxes, fontsize=30, fontweight='bold')
        axs[1].text(-0.16, 0.92, 'B', transform=axs[1].transAxes, fontsize=30, fontweight='bold')
        axs[2].text(-0.16, 0.92, 'C', transform=axs[2].transAxes, fontsize=30, fontweight='bold')

        fig.align_ylabels()

        if to_save:
            plt.savefig(f'outputs/figures/fig_timeseries_1.png',
                        bbox_inches='tight', dpi=200)


    def plot_view_2(self, to_save=False, xlims=None):
        """
        Make "View 2"

        Focus on diffusion-limitation, reaction limitation.

        Under diffusion limitation, there are two ways this gets limited:
        - thicknesss of the SEI
        - solvent depletion
        """

        num_subplots = 4

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(10, num_subplots * 5),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        xx = self.t/3600

        if xlims is not None:
            axs[0].set_xlim(xlims)

        # Currents
        i = 0
        # axs[i].axhline(y=0, ls='-', label='', c='k', lw=0.5)
        axs[i].plot(xx, self.i_app, c='k', lw=2, label=r'$I_{\mathrm{app}}$')
        axs[i].plot(np.NaN, np.NaN, c='k', lw=3, ls='--', label=r'$I_{\mathrm{sei}}$')
        axs[i].plot(xx, self.i_sei1, c='c', ls='-', lw=3, label=r'$I_{\mathrm{sei,A}}$')
        axs[i].plot(xx, self.i_sei2, c='m', ls='-', lw=3, label=r'$I_{\mathrm{sei,B}}$')
        axs[i].plot(xx, self.i_sei, c='k', lw=3, ls='--', label=r'')
        axs[i].text(0.86, 0.20, 'VC', fontsize=26, fontweight='bold', color='m')
        axs[i].text(1.43, 0.07, 'EC', fontsize=26, fontweight='bold', color='c')

        axs[i].legend(loc='right', fontsize=18)
        axs[i].set_ylabel(r'$I$ [A]')
        axs[i].set_ylim((-0.01, 0.27))

        # Current density of SEI 2
        i += 1
        axs[i].set_yscale('log')
        axs[i].plot(xx, self.j_sei2, c='m', linewidth=4, ls='-', label=r'$j_{\mathrm{sei,B}}$')
        axs[i].plot(xx, self.j_sei_rxn2, c='k', linewidth=2, ls='--', label=r'$\tilde{j}_{\mathrm{sei,B,rxn}}$')
        axs[i].plot(xx, self.j_sei_dif2, c='k', linewidth=2, ls='-.', label=r'$\tilde{j}_{\mathrm{sei,B,dif}}$')
        axs[i].legend(loc='lower right', fontsize=18)
        axs[i].axvline(0.775, color=(0.7, 0.7, 0.7), ls=':')
        axs[i].set_ylim((1e-13, 1e8))
        axs[i].text(0.1, 1e-5, 'VC', fontsize=26, fontweight='bold', color='m')
        axs[i].set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')


        # Current density of SEI 1
        i += 1
        axs[i].set_yscale('log')
        axs[i].plot(xx, self.j_sei1, c='c', linewidth=4, ls='-', label=r'$j_{\mathrm{sei,A}}$')
        axs[i].plot(xx, self.j_sei_rxn1, c='k', linewidth=2, ls='--', label=r'$\tilde{j}_{\mathrm{sei,A,rxn}}$')
        axs[i].plot(xx, self.j_sei_dif1, c='k', linewidth=2, ls='-.', label=r'$\tilde{j}_{\mathrm{sei,A,dif}}$')
        axs[i].legend(loc='lower right', fontsize=18)
        axs[i].text(0.1, 1e-5, 'EC', fontsize=26, fontweight='bold', color='c')
        axs[i].axvline(1.14, color=(0.7, 0.7, 0.7), ls=':')
        axs[i].text(0.1, 1e5, 'Reaction-Limited', color=(0.4, 0.4, 0.4), fontsize=20, fontweight='bold')
        axs[i].text(1.3, 1e5, 'Diffusion-Limited', color=(0.4, 0.4, 0.4), fontsize=20, fontweight='bold')
        axs[i].set_ylim((1e-13, 1e8))
        axs[i].set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')

        # Total cell expansion
        # i += 1
        # ff = self.delta_sei2 / (self.delta_sei2 + self.delta_sei1)
        # axs[i].plot(xx, (self.expansion_rev + self.expansion_irrev)*1e6, c='k', label=r'$dL_{\mathrm{SEI,A}} + dL_{\mathrm{SEI,B}} + dL_{\mathrm{rev}}$')
        # axs[i].plot(xx, self.expansion_irrev*1e6, c='c', ls='-', label=r'$dL_{\mathrm{SEI,A}} + dL_{\mathrm{SEI,B}}$')
        # axs[i].plot(xx, ff*self.expansion_irrev*1e6, c='m', ls='-', label=r'$dL_{\mathrm{SEI,B}}$')
        # axs[i].legend(loc='upper right', fontsize=16)
        # axs[i].set_ylabel(r'$dL$ ($\mu$m)')
        # axs[i].set_ylim(-2, 70)

        # Solvent consumption
        i += 1
        axs[i].plot(np.NaN, np.NaN,      c='c', lw=3, label=r'$c^{\mathrm{0}}_{\mathrm{sei,A}}$') # dummy
        axs[i].plot(xx, self.c_sei2/1e3, c='m', lw=3, label=r'$c^{\mathrm{0}}_{\mathrm{sei,B}}$')
        ax2 = axs[i].twinx()
        ax2.plot(xx, self.c_sei1/1e3, c='c', lw=3, label=r'$c^{\mathrm{bulk}}_{\mathrm{sei,A}}$')
        ax2.grid(False)
        axs[i].arrow(0.85, 0.20, -0.3, 0, head_width=0.01, head_length=0.05, color='m')
        ax2.arrow(1.5, 4.5, 0.3, 0, head_width=0.03, head_length=0.05, color='c')
        axs[i].set_ylabel(r'$c^{\mathrm{0}}$ [kmol/m$^3$]')
        axs[i].set_xlabel('Time [hrs]')
        axs[i].legend(loc='center right', fontsize=18)

        axs[i].text(0.55, 0.21, 'VC', fontsize=26, fontweight='bold', color='m')
        ax2.text(1.9, 4.47, 'EC', fontsize=26, fontweight='bold', color='c')

        axs[0].text(-0.15, 0.92, 'A', transform=axs[0].transAxes, fontsize=30, fontweight='bold')
        axs[1].text(-0.15, 0.92, 'B', transform=axs[1].transAxes, fontsize=30, fontweight='bold')
        axs[2].text(-0.15, 0.92, 'C', transform=axs[2].transAxes, fontsize=30, fontweight='bold')
        axs[3].text(-0.15, 0.92, 'D', transform=axs[3].transAxes, fontsize=30, fontweight='bold')

        fig.align_ylabels()

        bbox_dict = dict(boxstyle='circle', pad=0.1, facecolor='w', edgecolor='k')
        axs[2].text(0.4, 1e-12, '1', fontsize=16, bbox=bbox_dict)
        axs[2].text(0.8, 1e-8, '2', fontsize=16, bbox=bbox_dict)
        axs[2].text(1.12, 5e-3, '3', fontsize=16, bbox=bbox_dict)
        axs[1].text(0.4, 1e-12, '1', fontsize=16, bbox=bbox_dict)
        axs[1].text(0.65, 1e-8, '2', fontsize=16, bbox=bbox_dict)
        axs[1].text(0.75, 1e-2, '3', fontsize=16, bbox=bbox_dict)
        axs[1].text(1.20, 5e-2, '4', fontsize=16, bbox=bbox_dict)
        axs[1].text(0.60, 9e1, '5', fontsize=16, bbox=bbox_dict)
        axs[3].text(1.15, 0.17, '6', fontsize=16, bbox=bbox_dict)

        if to_save:
            plt.savefig(f'outputs/figures/fig_timeseries_2.tif', bbox_inches='tight',
                        dpi=200)


    def plot_view_3(self, to_save=False, xlims=None):
        """
        Make "View 3"

        Focus on understanding SEI growth boosting during cycling
        """

        num_subplots = 4

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(8, num_subplots * 4),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        xx = self.t/3600

        if xlims is not None:
            axs[0].set_xlim(xlims)

        i = 0
        # axs[i].plot(xx, self.ocv_p, ls='--', c='b', label='$U_p$')
        axs[i].plot(xx, self.ocv_p + self.eta_p, ls='-', c='b', label='$U_p + \eta_p$')
        # axs[i].plot(xx, self.ocv, ls='--', c='k', label='')
        axs[i].plot(xx, self.vt, ls='-', c='k', label='$V_t$')
        # axs[i].plot(xx, self.ocv_n, ls='--', c='r', label='$U_n$')
        axs[i].plot(xx, self.ocv_n - self.eta_n, ls='-', c='r', label='$U_n - \eta_n$')
        axs[i].set_ylabel('Voltage/Potential [V]')
        axs[i].legend(loc='right', fontsize=18)
        # axs[i].axhline(y=self.cell.U_SEI1, ls='--', c='g', label=rf'$U_{{\mathrm{{SEI,1}}}}$ = {self.cell.U_SEI1} V')
        # axs[i].axhline(y=self.cell.U_SEI2, ls='--', c='m', label=rf'$U_{{\mathrm{{SEI,2}}}}$ = {self.cell.U_SEI2} V')

        i += 1
        axs[i].set_ylabel(r'$\nu$ [-]')
        axs[i].plot(xx, self.delta_n, c='r', label=r'$\nu_\mathrm{n}$')
        axs[i].plot(xx, self.delta_p, c='b', label=r'$\nu_\mathrm{p}$')
        axs[i].legend(loc='center right', fontsize=18)

        i += 1
        axs[i].plot(xx, self.boost, c='k', label=r'$B$')
        axs[i].plot(xx, self.dndt*self.cell.gamma_boost, c='k',
                     ls='--', label=r'$\gamma \frac{d\nu_\mathrm{n}}{dt}$')
        axs[i].set_ylabel(r'$B$ [-]')
        axs[i].legend(loc='lower right', fontsize=18)

        i += 1
        axs[i].plot(xx, self.delta_sei1 * 1e9 + self.delta_sei2 * 1e9, c='k', label=r'$\delta_{\mathrm{SEI}}$')
        axs[i].plot(xx, self.delta_sei1 * 1e9, c='c', label=r'$\delta_{\mathrm{SEI,A}}$ (EC)')
        axs[i].plot(xx, self.delta_sei2 * 1e9, c='m', label=r'$\delta_{\mathrm{SEI,B}}$ (VC)')
        axs[i].legend(loc='right', fontsize=18)
        axs[i].set_ylabel(r'$\delta_{\mathrm{SEI}}$ [$nm$]')

        axs[i].set_xlabel('Time [hrs]')

        fig.align_ylabels()

        axs[0].text(-0.15, 0.92, 'A', transform=axs[0].transAxes, fontsize=30, fontweight='bold')
        axs[1].text(-0.15, 0.92, 'B', transform=axs[1].transAxes, fontsize=30, fontweight='bold')
        axs[2].text(-0.15, 0.92, 'C', transform=axs[2].transAxes, fontsize=30, fontweight='bold')
        axs[3].text(-0.15, 0.92, 'D', transform=axs[3].transAxes, fontsize=30, fontweight='bold')

        bbox_dict = dict(boxstyle='circle', pad=0.1, facecolor='none', edgecolor='k')
        axs[2].text(5, 0, '1', fontsize=16, bbox=bbox_dict)
        axs[1].text(30, 0.09, '2', fontsize=16, bbox=bbox_dict)
        axs[2].text(30, 35, '3', fontsize=16, bbox=bbox_dict)
        axs[2].text(55, 10, '4', fontsize=16, bbox=bbox_dict)

        plt.savefig(f'outputs/figures/fig_timeseries_3.tif',
                    bbox_inches='tight', dpi=200)


    def plot(self, to_save=True,
                   xlims=None,
                   fig_name=None):
        """
        Make a standard plot of the outputs
        """

        num_subplots = 13

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(10, num_subplots * 4),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        xx = self.t/3600

        if xlims is not None:
            axs[0].set_xlim(xlims)

        # Currents
        i = 0
        axs[i].axhline(y=0, ls='-', label='', c='k', lw=0.5)
        axs[i].plot(xx, self.i_app, c='k', label='$I_{app}$')
        axs[i].plot(xx, self.i_int, c='k', ls=':', label='$I_{int}$')
        axs[i].plot(xx, self.i_sei, c='g', ls='--', label='$I_{SEI}$')
        # axs[0].plot(xx, self.i_r1n, c='r', label='$I_{R_{1,n}}$')
        # axs[0].plot(xx, self.i_r1p, c='b', ls='--', label='$I_{R_{1,p}}$')
        axs[i].set_ylabel('Current (A)')
        axs[i].legend(loc='upper right')

        # Voltages and Potentials
        i += 1
        axs[i].plot(xx, self.vt, ls='-', c='k')
        axs[i].plot(xx, self.ocv, ls='--', c='k')
        axs[i].set_ylabel('V / V vs $Li/Li^+$ (V)')
        axs[i].plot(xx, self.ocv_p, ls='--', c='b', label='$U_p$')
        axs[i].plot(xx, self.ocv_p + self.eta_p, ls='-', c='b', label='$U_p + \eta_p$')
        axs[i].plot(xx, self.ocv_n, ls='--', c='r', label='$U_n$')
        axs[i].plot(xx, self.ocv_n - self.eta_n, ls='-', c='r', label='$U_n - \eta_n$')
        axs[i].axhline(y=self.cell.U_SEI1, ls='--', c='c', label=rf'$U_{{\mathrm{{SEI,A}}}}$ = {self.cell.U_SEI1} V')
        axs[i].axhline(y=self.cell.U_SEI2, ls='--', c='m', label=rf'$U_{{\mathrm{{SEI,B}}}}$ = {self.cell.U_SEI2} V')

        # Electrode stoichiometries
        i += 1
        axs[i].plot(xx, self.theta_n, c='r')
        axs[i].plot(xx, self.theta_p, c='b')
        axs[i].axhline(y=1, ls='-', label='', c='k', lw=0.5)
        axs[i].axhline(y=0, ls='-', label='', c='k', lw=0.5)
        axs[i].legend([r'$\theta_n$', r'$\theta_p$'], loc='upper right')
        axs[i].set_ylabel(r'$\theta$')
        axs[i].set_ylim((-0.1, 1.1))

        # Electrode expansion factors
        i += 1
        axs[i].set_ylabel(r'$\delta$')
        axs[i].plot(xx, self.delta_n, c='r')
        axs[i].plot(xx, self.delta_p, c='b')
        axs[i].legend([r'$\delta_n$', r'$\delta_p$'], loc='upper right')

        # SEI expansion factor
        i += 1
        axs[i].plot(xx, self.delta_sei1 * 1e9, c='c', label=r'$\delta_{\mathrm{sei},1}$')
        axs[i].plot(xx, self.delta_sei2 * 1e9, c='m', label=r'$\delta_{\mathrm{sei,2}}$')
        axs[i].plot(xx, self.delta_sei1 * 1e9 + self.delta_sei2 * 1e9, c='k', label=r'$\delta_{\mathrm{sei}}$')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(r'$\delta_{\mathrm{sei}}$ [$nm$]')

        # Total cell expansion
        i += 1
        axs[i].set_ylabel(r'$\epsilon$ ($\mu$m)')
        axs[i].plot(xx, self.expansion_irrev*1e6, c='c', label='$\epsilon_{irrev}$')
        axs[i].plot(xx, (self.expansion_rev + self.expansion_irrev)*1e6,
                    c='k', label='$\epsilon_{irrev} + \epsilon_{rev}$')
        axs[i].legend()

        # SEI reaction current densities
        i += 1
        axs[i].set_yscale('log')
        axs[i].plot(xx, self.j_sei_rxn1, c='g', label='$j_{sei,1,rxn}$')
        axs[i].plot(xx, self.j_sei_dif1, c='m', label='$j_{sei,1,dif}$')
        axs[i].plot(xx, np.abs(self.j_sei1), c='k', label='$j_{sei,1}$')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')

        # SEI reaction current densities
        i += 1
        axs[i].set_yscale('log')
        axs[i].plot(xx, self.j_sei_rxn2, c='g', label='$j_{sei,2,rxn}$')
        axs[i].plot(xx, self.j_sei_dif2, c='m', label='$j_{sei,2,dif}$')
        axs[i].plot(xx, np.abs(self.j_sei2), c='k', label='$j_{sei,2}$')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')

        # Total SEI reaction current
        i += 1
        axs[i].plot(xx, self.i_sei1, c='g', label='$I_{sei,1}$')
        axs[i].plot(xx, self.i_sei2, c='m', label='$I_{sei,2}$')
        axs[i].plot(xx, self.i_sei, c='k', label='$I_{sei}$')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(r'$I_{\mathrm{sei}}$ [A]')

        # Total SEI capacity
        i += 1
        axs[i].plot(xx, self.q_sei1, c='g', label='$Q_{\mathrm{sei,1}}$')
        axs[i].plot(xx, self.q_sei2, c='m', label='$Q_{\mathrm{sei,2}}$')
        axs[i].plot(xx, self.q_sei, c='k', label='$Q_{\mathrm{sei}}$')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(r'$Q_{\mathrm{sei}}$ [Ah]')

        i += 1
        axs[i].set_yscale('log')
        axs[i].plot(xx, self.D_sei1, c='g', label='$D_{sei,1}$')
        axs[i].plot(xx, self.D_sei2, c='m', label='$D_{sei,2}$')
        axs[i].axhline(y=self.cell.D_SEI11, ls='--', label='$D_{sei,11}$', c='g')
        axs[i].axhline(y=self.cell.D_SEI22, ls='--', label='$D_{sei,22}$', c='m')
        axs[i].legend(loc='right')
        axs[i].set_ylabel(r'$D_{sei}$')

        # Boost
        i += 1
        axs[i].plot(xx, self.dndt*self.cell.gamma_boost, c='k',
                     ls='--', label=r'$\gamma \frac{d\delta_n}{dt}$')
        axs[i].plot(xx, self.boost, c='k', label=r'$B$')
        axs[i].set_ylabel(r'$B$')
        axs[i].legend(loc='right')

        # SEI concentrations
        i += 1
        axs[i].plot(xx, self.c_sei1/1e3, c='g', label=r'$c^{\mathrm{bulk}}_{SEI,1}$')
        axs[i].plot(xx, self.c_sei2/1e3, c='m', label=r'$c^{\mathrm{bulk}}_{SEI,2}$')
        axs[i].set_ylabel(r'$c^{\mathrm{bulk}}$ (kmol/m$^3$)')
        axs[i].set_xlabel('Time [hrs]')
        axs[i].legend(loc='right')

        fig.align_ylabels()

        if fig_name is None:
            fig_name = f'{self.name}_output'

        if to_save:
            plt.savefig(f'outputs/figures/{fig_name}.png',
                        bbox_inches='tight',
                        dpi=150)
