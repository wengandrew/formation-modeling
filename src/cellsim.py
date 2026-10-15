"""
Classes for the formation modeling work
"""

from src import plotter as plotter
from src import modelutils as mu
import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt

plotter.initialize(plt, style='default')

F = 96485.33212    # C/mol      Faraday's constant
T = 273.15 + 25    # Kelvin     Temperature
R = 8.314          # J/mol/K    Universal gas constant

STEP_NUM_CHARGE_CC = 0
STEP_NUM_CHARGE_CV = 1
STEP_NUM_DISCHARGE_CC = 2
STEP_NUM_DISCHARGE_CV = 3
STEP_NUM_REST = 4

# Where the Simulation plotting methods save figures
FIGURE_DIR = 'notebooks/outputs/figures'

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

        return ', '.join([
            f'k={self.k_SEI1}',
            f'D={self.D_SEI11}',
            f'U={self.U_SEI1}',
        ])



class Simulation:

    def __init__(self, cell: Cell, sim_time_s: int, name='DefaultSim', dt=5.0):
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
        self.dt = dt
        self.t = np.arange(0, sim_time_s, self.dt)

        # Track where we are in the simulation
        self.curr_k = 0

        # Initialize output vectors
        self.i_app = np.zeros(self.t.shape)

        self.cycle_number = np.zeros(self.t.shape)
        self.step_number = np.zeros(self.t.shape)
        self.phase = np.zeros(self.t.shape, dtype=int)

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

        # Homogenized quantities
        self.D_sei1   = mu.initialize(self.t,
                                      1/(1/cell.D_SEI11 + 1/cell.D_SEI12))
        self.D_sei2   = mu.initialize(self.t,
                                      1/(1/cell.D_SEI21 + 1/cell.D_SEI22))

        # Individual components
        self.j_sei1     = mu.initialize(self.t, 0)
        self.j_sei2     = mu.initialize(self.t, 0)
        self.c_sei1     = mu.initialize(self.t, cell.c_SEI1_0)
        self.c_sei2     = mu.initialize(self.t, cell.c_SEI2_0)
        self.i_sei1     = mu.initialize(self.t, 0)
        self.i_sei2     = mu.initialize(self.t, 0)
        self.q_sei1     = mu.initialize(self.t, 0)
        self.q_sei2     = mu.initialize(self.t, 0)
        self.j_sei_rxn1 = mu.initialize(self.t, np.nan)
        self.j_sei_rxn2 = mu.initialize(self.t, np.nan)
        self.j_sei_dif1 = mu.initialize(self.t, np.nan)
        self.j_sei_dif2 = mu.initialize(self.t, np.nan)
        self.eta_sei1   = mu.initialize(self.t, 0)
        self.eta_sei2   = mu.initialize(self.t, 0)


        # Expansion states
        self.delta_sei1 = mu.initialize(self.t, cell.delta_SEI1_0)
        self.delta_sei2 = mu.initialize(self.t, cell.delta_SEI2_0)
        self.delta_sei  = mu.initialize(self.t, \
                                        self.delta_sei1[0] + self.delta_sei2[0])
        self.delta_n    = mu.initialize(self.t, cell.En(cell.theta_n))
        self.delta_p    = mu.initialize(self.t, cell.Ep(cell.theta_p))
        self.dndt       = mu.initialize(self.t, 0)
        self.boost      = mu.initialize(self.t, 0)
        self.expansion_rev   = mu.initialize(self.t, 0)
        self.expansion_irrev = mu.initialize(self.t, 0)
        self.expansion  = mu.initialize(self.t, 0)

        # Total quantities
        self.j_sei   = mu.initialize(self.t, 0)
        self.i_sei   = mu.initialize(self.t, 0)
        self.q_sei   = mu.initialize(self.t, 0)

        a_n_init = (3 * cell.epsilon_n / (cell.R_n * (1 + cell.En(cell.theta_n))**(1/3))) * cell.A_n * cell.L_n

        self.R_sei1     = mu.initialize(self.t,
                                self.delta_sei1[0] / (cell.kappa_SEI1 * a_n_init))

        self.R_sei2     = mu.initialize(self.t,
                                self.delta_sei2[0] / (cell.kappa_SEI2 * a_n_init))

        self.R_sei   = mu.initialize(self.t, self.R_sei1[0] + self.R_sei2[0])
 
        # SEI density
        self.rho_sei = mu.initialize(self.t, np.nan) # g/cm^3


    def step(self, k: int, mode: str, icc=0, icv=0,
             cyc_num=np.nan, step_num=np.nan,
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
                        ( ( 1 - np.exp(-self.dt/(p.R1n*p.C1n)) ) * p.R1n + self.R_sei[k] + \
                          ( 1 - np.exp(-self.dt/(p.R1p*p.C1p)) ) * p.R1p + p.R0p )



        dQint = self.i_int[k] * self.dt / 3600 # Amp-hours
        dQapp = self.i_app[k] * self.dt / 3600 # Amp-hours

        # assert dQint > 0, 'How are we decreasing theta_n during lithiation?'

        self.theta_n[k + 1] = self.theta_n[k] + dQint / p.Cn
        self.theta_p[k + 1] = self.theta_p[k] - dQapp / p.Cp

        # Number of graphite anode particles (from electrode geometry)
        N_PARTICLES = 163045167
        curr_epsilon_n = N_PARTICLES * 4/3 * np.pi * p.R_n**3 * (1 + self.cell.En(self.theta_n[k+1])) / (p.A_n * p.L_n)
        # Active anode surface area [m^2] accounting for current porosity
        a_n = (3 * curr_epsilon_n / (p.R_n * (1 + self.cell.En(self.theta_n[k+1]))**(1/3))) * p.A_n * p.L_n

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
        self.eta_n[k+1] = p.R1n * self.i_r1n[k] + self.R_sei[k] * self.i_int[k]

        # Terminal voltage update
        if mode == 'cc':
            self.vt[k+1] = self.ocv_p[k+1] + self.eta_p[k] - \
                          (self.ocv_n[k+1] - self.eta_n[k])
        elif mode == 'cv':
            self.vt[k+1] = vcv

        # SEI growth update
        self.eta_sei1[k+1] = -self.eta_n[k+1] + self.ocv_n[k+1] - p.U_SEI1
        self.eta_sei2[k+1] = -self.eta_n[k+1] + self.ocv_n[k+1] - p.U_SEI2

        # Mixed reaction and diffusion limited SEI current density
        # Boosted SEI reaction during cycling
        self.dndt[k+1] = (self.delta_n[k+1] - self.delta_n[k]) / self.dt

        if self.i_app[k] > 0: # charging / lithiating graphite
            if p.tau_boost != 0:
                self.boost[k+1] = self.boost[k] + self.dt / p.tau_boost * \
                                     (self.dndt[k+1] * p.gamma_boost - self.boost[k])
            else:
                self.boost[k+1] = self.boost[k]
        else: # discharging or resting
            if p.tau_decay != 0:
                self.boost[k+1] = self.boost[k] - self.dt * self.boost[k] / p.tau_decay
            else:
                self.boost[k+1] = self.boost[k]

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

        # Current density to current conversion
        self.i_sei[k+1]  = self.j_sei[k+1]  * a_n
        self.i_sei1[k+1] = self.j_sei1[k+1] * a_n
        self.i_sei2[k+1] = self.j_sei2[k+1] * a_n

        # Update SEI reacting species concentrations
        self.c_sei1[k+1] = self.c_sei1[k] - self.dt * (a_n / p.A_n / p.L_n) * self.j_sei1[k+1] / (p.n_SEI1 * F)
        self.c_sei2[k+1] = self.c_sei2[k] - self.dt * (a_n / p.A_n / p.L_n) * self.j_sei2[k+1] / (p.n_SEI2 * F)

        # Update the intercalation current
        self.i_int[k+1] = self.i_app[k]  - self.i_sei[k+1]

        if to_debug:
            print(
                f'k: {k} | '\
                f'Vt: {self.vt[k]:.3f} | '\
                f'theta_n: {self.theta_n[k]:.3f} | '\
                f'theta_p: {self.theta_p[k]:.3f} | '\
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
                              self.dt * ( (p.MW_SEI1 / (p.rho_SEI1 * 1e6)) * \
                                        np.abs(self.j_sei1[k+1]) ) / (p.n_SEI1 * F)

        self.delta_sei2[k+1] = self.delta_sei2[k] + \
                              self.dt * ( (p.MW_SEI2 / (p.rho_SEI2 * 1e6)) * \
                                        np.abs(self.j_sei2[k+1]) ) / (p.n_SEI2 * F)

        self.delta_sei[k+1] = self.delta_sei1[k+1] + self.delta_sei2[k+1]

        # Expansion update
        # Cathode and anode expansion function update
        self.expansion_rev[k+1] = p.c1 * self.delta_p[k+1] + \
                                  p.c2 * self.delta_n[k+1]
        self.expansion_irrev[k+1] = p.c0 * self.delta_sei[k+1]
        self.expansion[k+1] = self.expansion_rev[k+1] + \
                              self.expansion_irrev[k+1]

        # SEI resistance updates
        self.R_sei1[k+1] = self.delta_sei1[k+1] / (p.kappa_SEI1 * a_n)
        self.R_sei2[k+1] = self.delta_sei2[k+1] / (p.kappa_SEI2 * a_n)
        self.R_sei[k+1] = self.R_sei1[k+1] + self.R_sei2[k+1]

        # SEI density update
        self.rho_sei[k+1] = (self.delta_sei1[k+1] * p.rho_SEI1 + self.delta_sei2[k+1] * p.rho_SEI2) / \
                            (self.delta_sei1[k+1] + self.delta_sei2[k+1])


    def run_rest(self, cycle_number: int, rest_time_hrs: float,
                 to_print=False):
        """
        Run a rest step

        Parameters
        ----------
        cycle_number (int): a cycle number label
        rest_time_hrs (float): hours to rest
        """

        if to_print:
            print(f'Running Cyc{cycle_number}: Rest for {rest_time_hrs:.2f} hours...')

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
                     icc: float, icv: float, vmax: float,
                     to_print=False):
        """
        Run a charge CCCV step.

        Parameters:
        -----------
        cycle_number (int): a cycle number label
        icc (float): CC current in amps (+ve is charge)
        icc (float): CV current cutoff in amps (+ve is charge)
        vmax (float): charge voltage target
        """

        if to_print:
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
                     icc: float, icv: float, vmin: float,
                     to_print=False):
        """
        Run a discharge CCCV step.

        Parameters:
        -----------
        cycle_number (int): a cycle number label
        icc (float): CC current in amps (+ve is charge)
        icc (float): CV current cutoff in amps (+ve is charge)
        vmin (float): discharge voltage target
        """

        if to_print:
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

        # Get rid of extra nan rows beyond the simulation time
        df = df.iloc[0:self.curr_k, :]

        return df


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _new_figure(self, num_subplots, figsize, xlims=None):
        """Vertically stacked panels sharing the time axis [hrs]."""
        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))
        fig, axs = plt.subplots(nrows=num_subplots, ncols=1, figsize=figsize,
                                gridspec_kw=gridspec, sharex=True)
        for ax in axs:
            ax.grid(False)
        if xlims is not None:
            axs[0].set_xlim(xlims)
        return fig, axs, self.t / 3600

    @staticmethod
    def _panel_letters(axs, x):
        for ax, letter in zip(axs, 'ABCD'):
            ax.text(x, 0.92, letter, transform=ax.transAxes, fontsize=30, fontweight='bold')

    @staticmethod
    def _numbered_markers(markers, facecolor='none'):
        """Circled numbers; markers is a list of (ax, x, y, number)."""
        bbox_dict = dict(boxstyle='circle', pad=0.1, facecolor=facecolor, edgecolor='k')
        for ax, x, y, number in markers:
            ax.text(x, y, number, fontsize=16, bbox=bbox_dict)

    def _plot_potentials_vs_sei_onset(self, ax, xx, ec_text, vc_text):
        """Electrode potentials with the EC and VC reaction potentials."""
        ax.plot(xx, self.ocv_p + self.eta_p, ls='-', c='b', label=r'$U^+ + \eta^+$')
        ax.plot(xx, self.vt, ls='-', c='k', label='$V_t$')
        ax.set_ylabel('Voltage/Potential [V]')
        ax.plot(xx, self.ocv_n - self.eta_n, ls='-', c='r', label=r'$U^- - \eta^-$')
        ax.axhline(y=self.cell.U_SEI2, ls='--', c='m', label=r'$U_{\mathrm{VC}}$ = 1.35V')
        ax.axhline(y=self.cell.U_SEI1, ls='--', c='c', label=r'$U_{\mathrm{EC}}$ = 0.80V')
        ax.text(1.5, 0.9, ec_text, fontsize=20, fontweight='bold', color='c')
        ax.text(1.5, 1.45, vc_text, fontsize=20, fontweight='bold', color='m')
        ax.set_ylim((0.0, 4.0))
        lh = ax.legend(loc='right', fancybox=False, frameon=True, fontsize=18)
        lh.get_frame().set_alpha(None)

    def _plot_sei_currents(self, ax, xx):
        """Applied current and the EC/VC SEI currents."""
        ax.axhline(y=0, ls='-', label='', c='k', lw=0.5)
        ax.plot(xx, self.i_app, c='k', label=r'$I_{\mathrm{app}}$')
        ax.plot(np.nan, np.nan, c='k', lw=2, ls='--', label=r'$I_{\mathrm{sei}}$')
        ax.plot(xx, self.i_sei2, c='m', ls='-', label=r'$I_{\mathrm{VC}}$')
        ax.plot(xx, self.i_sei1, c='c', ls='-', label=r'$I_{\mathrm{EC}}$')
        ax.plot(xx, self.i_sei, c='k', lw=2, ls='--', label='')
        ax.legend(loc='center right', fancybox=False, frameon=False, fontsize=18)

    def _plot_current_density_limits(self, ax, xx, n, color, letter):
        """SEI n current density with its reaction and diffusion limits (log scale)."""
        ax.set_yscale('log')
        ax.plot(xx, getattr(self, f'j_sei{n}'), c=color, linewidth=4, ls='-',
                label=rf'$j_{{\mathrm{{sei,{letter}}}}}$')
        ax.plot(xx, getattr(self, f'j_sei_rxn{n}'), c='k', linewidth=2, ls='--',
                label=rf'$\tilde{{j}}_{{\mathrm{{sei,{letter},rxn}}}}$')
        ax.plot(xx, getattr(self, f'j_sei_dif{n}'), c='k', linewidth=2, ls='-.',
                label=rf'$\tilde{{j}}_{{\mathrm{{sei,{letter},dif}}}}$')
        ax.legend(loc='lower right', fontsize=18)

    @staticmethod
    def _plot_series(ax, xx, series, ylabel=None, legend_loc=None, yscale=None):
        """Plot (y, kwargs) pairs on one panel."""
        if yscale:
            ax.set_yscale(yscale)
        for y, kwargs in series:
            ax.plot(xx, y, **kwargs)
        if legend_loc is not None:
            ax.legend(loc=legend_loc)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def _first_charge_markers(self, axs):
        self._numbered_markers([
            (axs[0], 0.25, 0.30, '1'),
            (axs[1], 1.02, 0.22, '2'),
            (axs[1], 1.42, 0.11, '3'),
            (axs[2], 4.40, 64.0, '4'),
            (axs[1], 4.40, 0.023, '5'),
        ])

    def plot_view_4(self, to_save=False, xlims=None,
                    to_annotate=True, ylim_current=None,
                    ylim_thickness=None):
        """
        Make "View 4"

        Focus on voltage, current, and SEI capacity.
        ylim_thickness is unused (kept for backward compatibility).
        """
        fig, axs, xx = self._new_figure(3, (8, 3 * 4.5), xlims)

        self._plot_potentials_vs_sei_onset(axs[0], xx, 'EC', 'VC')

        self._plot_sei_currents(axs[1], xx)
        axs[1].set_ylabel('Current [A]')
        axs[1].set_ylim((0, 0.27) if ylim_current is None else ylim_current)

        axs[2].plot(xx, self.q_sei2, c='m', label=r'$Q_{\mathrm{VC}}$')
        axs[2].plot(xx, self.q_sei1, c='c', label=r'$Q_{\mathrm{EC}}$')
        axs[2].plot(xx, self.q_sei, c='k', label=r'$Q_{\mathrm{sei}}$')
        axs[2].legend(loc='upper right')
        axs[2].set_ylabel(r'$Q_{\mathrm{sei}}$ [Ah]')
        axs[2].set_ylim((-0.0, 0.29))
        axs[2].set_xlabel('Time [hrs]')

        if to_annotate:
            self._first_charge_markers(axs)
            self._panel_letters(axs, -0.16)

        fig.align_ylabels()

        if to_save:
            plt.savefig(f'{FIGURE_DIR}/fig_timeseries_1.png',
                        bbox_inches='tight', dpi=200)

    def plot_view_1(self, to_save=False, xlims=None):
        """
        Make "View 1"

        Focus on voltage, current, and expansion
        """
        fig, axs, xx = self._new_figure(3, (10, 3 * 4), xlims)

        self._plot_potentials_vs_sei_onset(axs[0], xx, 'SEI 2 (EC)', 'SEI 1 (VC)')

        self._plot_sei_currents(axs[1], xx)
        axs[1].text(1, 0.17, 'SEI 1', fontsize=20, fontweight='bold', color='m')
        axs[1].text(1.85, 0.05, 'SEI 2', fontsize=20, fontweight='bold', color='c')
        axs[1].set_ylabel('Current [A]')
        axs[1].set_ylim((0, 0.27))

        # Total cell expansion
        ax = axs[2]
        ff = self.delta_sei2 / (self.delta_sei2 + self.delta_sei1)
        ax.plot(xx, (self.expansion_rev + self.expansion_irrev)*1e6, c='k', label=r'$\Delta_{\mathrm{VC}}$ + $\Delta_{\mathrm{EC}}$ + $\Delta_{\mathrm{rev}}$')
        ax.plot(xx, self.expansion_irrev*1e6, c='c', ls='-', label=r'$\Delta_{\mathrm{VC}}$ + $\Delta_{\mathrm{EC}}$')
        ax.plot(xx, ff*self.expansion_irrev*1e6, c='m', ls='-', label=r'$\Delta_{\mathrm{VC}}$')
        ax.legend(loc='upper left', fontsize=18)
        ax.set_ylabel(r'Thickness [$\mu$m]')
        ax.set_xlabel('Time [hrs]')
        ax.set_ylim((-5, 75))
        ax.text(4, -0, 'SEI 1', fontsize=20, fontweight='bold', color='m')
        ax.text(4, 19, 'SEI 2', fontsize=20, fontweight='bold', color='c')
        ax.text(4, 43, 'Reversible', fontsize=20, fontweight='bold', color='k')

        self._first_charge_markers(axs)
        self._panel_letters(axs, -0.16)

        fig.align_ylabels()

        if to_save:
            plt.savefig(f'{FIGURE_DIR}/fig_timeseries_1.png',
                        bbox_inches='tight', dpi=200)

    def plot_view_2(self, to_save=False, xlims=None):
        """
        Make "View 2"

        Focus on diffusion-limitation, reaction limitation.

        Under diffusion limitation, there are two ways this gets limited:
        - thicknesss of the SEI
        - solvent depletion
        """
        fig, axs, xx = self._new_figure(4, (10, 4 * 5), xlims)

        # Currents
        ax = axs[0]
        ax.plot(xx, self.i_app, c='k', lw=2, label=r'$I_{\mathrm{app}}$')
        ax.plot(np.nan, np.nan, c='k', lw=3, ls='--', label=r'$I_{\mathrm{sei}}$')
        ax.plot(xx, self.i_sei1, c='c', ls='-', lw=3, label=r'$I_{\mathrm{sei,A}}$')
        ax.plot(xx, self.i_sei2, c='m', ls='-', lw=3, label=r'$I_{\mathrm{sei,B}}$')
        ax.plot(xx, self.i_sei, c='k', lw=3, ls='--', label=r'')
        ax.text(0.86, 0.20, 'VC', fontsize=26, fontweight='bold', color='m')
        ax.text(1.43, 0.07, 'EC', fontsize=26, fontweight='bold', color='c')
        ax.legend(loc='right', fontsize=18)
        ax.set_ylabel(r'$I$ [A]')
        ax.set_ylim((-0.01, 0.27))

        # Current density of SEI 2 (VC)
        ax = axs[1]
        self._plot_current_density_limits(ax, xx, 2, 'm', 'B')
        ax.axvline(0.775, color=(0.7, 0.7, 0.7), ls=':')
        ax.set_ylim((1e-13, 1e8))
        ax.text(0.1, 1e-5, 'VC', fontsize=26, fontweight='bold', color='m')
        ax.set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')

        # Current density of SEI 1 (EC)
        ax = axs[2]
        self._plot_current_density_limits(ax, xx, 1, 'c', 'A')
        ax.text(0.1, 1e-5, 'EC', fontsize=26, fontweight='bold', color='c')
        ax.axvline(1.14, color=(0.7, 0.7, 0.7), ls=':')
        ax.text(0.1, 1e5, 'Reaction-Limited', color=(0.4, 0.4, 0.4), fontsize=20, fontweight='bold')
        ax.text(1.3, 1e5, 'Diffusion-Limited', color=(0.4, 0.4, 0.4), fontsize=20, fontweight='bold')
        ax.set_ylim((1e-13, 1e8))
        ax.set_ylabel(r'$j_{\mathrm{sei}}$ [A/m$^2$]')

        # Solvent consumption
        ax = axs[3]
        ax.plot(np.nan, np.nan,      c='c', lw=3, label=r'$c^{\mathrm{0}}_{\mathrm{sei,A}}$') # dummy
        ax.plot(xx, self.c_sei2/1e3, c='m', lw=3, label=r'$c^{\mathrm{0}}_{\mathrm{sei,B}}$')
        ax2 = ax.twinx()
        ax2.plot(xx, self.c_sei1/1e3, c='c', lw=3, label=r'$c^{\mathrm{bulk}}_{\mathrm{sei,A}}$')
        ax2.grid(False)
        ax.arrow(0.85, 0.20, -0.3, 0, head_width=0.01, head_length=0.05, color='m')
        ax2.arrow(1.5, 4.5, 0.3, 0, head_width=0.03, head_length=0.05, color='c')
        ax.set_ylabel(r'$c^{\mathrm{0}}$ [kmol/m$^3$]')
        ax.set_xlabel('Time [hrs]')
        ax.legend(loc='center right', fontsize=18)
        ax.text(0.55, 0.21, 'VC', fontsize=26, fontweight='bold', color='m')
        ax2.text(1.9, 4.47, 'EC', fontsize=26, fontweight='bold', color='c')

        self._panel_letters(axs, -0.15)

        fig.align_ylabels()

        self._numbered_markers([
            (axs[2], 0.4, 1e-12, '1'),
            (axs[2], 0.8, 1e-8, '2'),
            (axs[2], 1.12, 5e-3, '3'),
            (axs[1], 0.4, 1e-12, '1'),
            (axs[1], 0.65, 1e-8, '2'),
            (axs[1], 0.75, 1e-2, '3'),
            (axs[1], 1.20, 5e-2, '4'),
            (axs[1], 0.60, 9e1, '5'),
            (axs[3], 1.15, 0.17, '6'),
        ], facecolor='w')

        if to_save:
            plt.savefig(f'{FIGURE_DIR}/fig_timeseries_2.tif', bbox_inches='tight',
                        dpi=200)

    def plot_view_3(self, to_save=False, xlims=None):
        """
        Make "View 3"

        Focus on understanding SEI growth boosting during cycling.
        Always saves the figure (to_save is ignored, as before).
        """
        fig, axs, xx = self._new_figure(4, (8, 4 * 4), xlims)

        ax = axs[0]
        ax.plot(xx, self.ocv_p + self.eta_p, ls='-', c='b', label=r'$U_p + \eta_p$')
        ax.plot(xx, self.vt, ls='-', c='k', label='$V_t$')
        ax.plot(xx, self.ocv_n - self.eta_n, ls='-', c='r', label=r'$U_n - \eta_n$')
        ax.set_ylabel('Voltage/Potential [V]')
        ax.legend(loc='right', fontsize=18)

        ax = axs[1]
        ax.set_ylabel(r'$\nu$ [-]')
        ax.plot(xx, self.delta_n, c='r', label=r'$\nu_\mathrm{n}$')
        ax.plot(xx, self.delta_p, c='b', label=r'$\nu_\mathrm{p}$')
        ax.legend(loc='center right', fontsize=18)

        ax = axs[2]
        ax.plot(xx, self.boost, c='k', label=r'$B$')
        ax.plot(xx, self.dndt*self.cell.gamma_boost, c='k',
                ls='--', label=r'$\gamma \frac{d\nu_\mathrm{n}}{dt}$')
        ax.set_ylabel(r'$B$ [-]')
        ax.legend(loc='lower right', fontsize=18)

        ax = axs[3]
        ax.plot(xx, self.delta_sei1 * 1e9 + self.delta_sei2 * 1e9, c='k', label=r'$\delta_{\mathrm{SEI}}$')
        ax.plot(xx, self.delta_sei1 * 1e9, c='c', label=r'$\delta_{\mathrm{SEI,A}}$ (EC)')
        ax.plot(xx, self.delta_sei2 * 1e9, c='m', label=r'$\delta_{\mathrm{SEI,B}}$ (VC)')
        ax.legend(loc='right', fontsize=18)
        ax.set_ylabel(r'$\delta_{\mathrm{SEI}}$ [$nm$]')
        ax.set_xlabel('Time [hrs]')

        fig.align_ylabels()

        self._panel_letters(axs, -0.15)
        self._numbered_markers([
            (axs[2], 5, 0, '1'),
            (axs[1], 30, 0.09, '2'),
            (axs[2], 30, 35, '3'),
            (axs[2], 55, 10, '4'),
        ])

        plt.savefig(f'{FIGURE_DIR}/fig_timeseries_3.tif',
                    bbox_inches='tight', dpi=200)

    def plot(self, to_save=True,
                   xlims=None,
                   fig_name=None):
        """
        Make a standard plot of the outputs
        """
        fig, axs, xx = self._new_figure(13, (10, 13 * 4), xlims)
        series = self._plot_series

        # Currents
        axs[0].axhline(y=0, ls='-', label='', c='k', lw=0.5)
        series(axs[0], xx, [
            (self.i_app, dict(c='k', label='$I_{app}$')),
            (self.i_int, dict(c='k', ls=':', label='$I_{int}$')),
            (self.i_sei, dict(c='g', ls='--', label='$I_{SEI}$')),
        ])
        axs[0].set_ylabel('Current (A)')
        axs[0].legend(loc='upper right')

        # Voltages and Potentials
        ax = axs[1]
        ax.plot(xx, self.vt, ls='-', c='k')
        ax.plot(xx, self.ocv, ls='--', c='k')
        ax.set_ylabel('V / V vs $Li/Li^+$ (V)')
        series(ax, xx, [
            (self.ocv_p, dict(ls='--', c='b', label='$U_p$')),
            (self.ocv_p + self.eta_p, dict(ls='-', c='b', label=r'$U_p + \eta_p$')),
            (self.ocv_n, dict(ls='--', c='r', label='$U_n$')),
            (self.ocv_n - self.eta_n, dict(ls='-', c='r', label=r'$U_n - \eta_n$')),
        ])
        ax.axhline(y=self.cell.U_SEI1, ls='--', c='c', label=rf'$U_{{\mathrm{{SEI,A}}}}$ = {self.cell.U_SEI1} V')
        ax.axhline(y=self.cell.U_SEI2, ls='--', c='m', label=rf'$U_{{\mathrm{{SEI,B}}}}$ = {self.cell.U_SEI2} V')

        # Electrode stoichiometries
        ax = axs[2]
        ax.plot(xx, self.theta_n, c='r')
        ax.plot(xx, self.theta_p, c='b')
        ax.axhline(y=1, ls='-', label='', c='k', lw=0.5)
        ax.axhline(y=0, ls='-', label='', c='k', lw=0.5)
        ax.legend([r'$\theta_n$', r'$\theta_p$'], loc='upper right')
        ax.set_ylabel(r'$\theta$')
        ax.set_ylim((-0.1, 1.1))

        # Electrode expansion factors
        ax = axs[3]
        ax.set_ylabel(r'$\delta$')
        ax.plot(xx, self.delta_n, c='r')
        ax.plot(xx, self.delta_p, c='b')
        ax.legend([r'$\delta_n$', r'$\delta_p$'], loc='upper right')

        # SEI thickness
        series(axs[4], xx, [
            (self.delta_sei1 * 1e9, dict(c='c', label=r'$\delta_{\mathrm{sei},1}$')),
            (self.delta_sei2 * 1e9, dict(c='m', label=r'$\delta_{\mathrm{sei,2}}$')),
            (self.delta_sei1 * 1e9 + self.delta_sei2 * 1e9, dict(c='k', label=r'$\delta_{\mathrm{sei}}$')),
        ], ylabel=r'$\delta_{\mathrm{sei}}$ [$nm$]', legend_loc='upper right')

        # Total cell expansion
        ax = axs[5]
        ax.set_ylabel(r'$\epsilon$ ($\mu$m)')
        ax.plot(xx, self.expansion_irrev*1e6, c='c', label=r'$\epsilon_{irrev}$')
        ax.plot(xx, (self.expansion_rev + self.expansion_irrev)*1e6,
                c='k', label=r'$\epsilon_{irrev} + \epsilon_{rev}$')
        ax.legend()

        # SEI reaction current densities
        for ax, n in ((axs[6], 1), (axs[7], 2)):
            series(ax, xx, [
                (getattr(self, f'j_sei_rxn{n}'), dict(c='g', label=f'$j_{{sei,{n},rxn}}$')),
                (getattr(self, f'j_sei_dif{n}'), dict(c='m', label=f'$j_{{sei,{n},dif}}$')),
                (np.abs(getattr(self, f'j_sei{n}')), dict(c='k', label=f'$j_{{sei,{n}}}$')),
            ], ylabel=r'$j_{\mathrm{sei}}$ [A/m$^2$]', legend_loc='upper right', yscale='log')

        # Total SEI reaction current
        series(axs[8], xx, [
            (self.i_sei1, dict(c='g', label='$I_{sei,1}$')),
            (self.i_sei2, dict(c='m', label='$I_{sei,2}$')),
            (self.i_sei, dict(c='k', label='$I_{sei}$')),
        ], ylabel=r'$I_{\mathrm{sei}}$ [A]', legend_loc='upper right')

        # Total SEI capacity
        series(axs[9], xx, [
            (self.q_sei1, dict(c='g', label=r'$Q_{\mathrm{sei,1}}$')),
            (self.q_sei2, dict(c='m', label=r'$Q_{\mathrm{sei,2}}$')),
            (self.q_sei, dict(c='k', label=r'$Q_{\mathrm{sei}}$')),
        ], ylabel=r'$Q_{\mathrm{sei}}$ [Ah]', legend_loc='upper right')

        # SEI diffusivities
        ax = axs[10]
        ax.set_yscale('log')
        ax.plot(xx, self.D_sei1, c='g', label='$D_{sei,1}$')
        ax.plot(xx, self.D_sei2, c='m', label='$D_{sei,2}$')
        ax.axhline(y=self.cell.D_SEI11, ls='--', label='$D_{sei,11}$', c='g')
        ax.axhline(y=self.cell.D_SEI22, ls='--', label='$D_{sei,22}$', c='m')
        ax.legend(loc='right')
        ax.set_ylabel(r'$D_{sei}$')

        # Boost
        series(axs[11], xx, [
            (self.dndt*self.cell.gamma_boost, dict(c='k', ls='--', label=r'$\gamma \frac{d\delta_n}{dt}$')),
            (self.boost, dict(c='k', label=r'$B$')),
        ], ylabel=r'$B$', legend_loc='right')

        # SEI concentrations
        ax = axs[12]
        ax.plot(xx, self.c_sei1/1e3, c='g', label=r'$c^{\mathrm{bulk}}_{SEI,1}$')
        ax.plot(xx, self.c_sei2/1e3, c='m', label=r'$c^{\mathrm{bulk}}_{SEI,2}$')
        ax.set_ylabel(r'$c^{\mathrm{bulk}}$ (kmol/m$^3$)')
        ax.set_xlabel('Time [hrs]')
        ax.legend(loc='right')

        fig.align_ylabels()

        if fig_name is None:
            fig_name = f'{self.name}_output'

        if to_save:
            plt.savefig(f'{FIGURE_DIR}/{fig_name}.png',
                        bbox_inches='tight',
                        dpi=150)
