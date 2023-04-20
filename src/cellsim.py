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
import numpy as np
from matplotlib import pyplot as plt

plotter.initialize(plt)

F = 96485.33212    # C/mol      Faraday's constant
T = 273.15 + 25    # Kelvin     Temperature
R = 8.314          # J/mol/K    Universal gas constant

class Cell:

    def __init__(self, name=''):

        self.name = name

        # Initialize default cell parameters

        # SEI growth parameters
        self.alpha_SEI   = 0.5        # [-]        SEI charge transfer coefficient
        self.c_EC_bulk   = 4541       # mol/m3     Concentration of EC in bulk electrolyte
        self.delta_SEI_0 = 5e-9       # m          Initial SEI thickness
        self.V_sei       = 9.585e-5   # m3/mol     SEI molar volume
        self.L_n         = 80e-6      # m          Negative electrode thickness
        self.k_SEI       = 1e-11      # m/s        SEI kinetic rate constant
        self.D_SEI       = 2e-16      # m2/s       SEI layer diffusivity
        self.R_n         = 20e-6      # m          Anode particle radius
        self.epsilon_n   = 0.7        # [-]        Anode solid material fraction
        self.a_SEI       = 3 * self.epsilon_n / self.R_n # 1/m   Anode specific surface area
        self.A_n         = 0.100      # m2         Anode active area
        self.U_SEI       = 0.4        # V          SEI equilibrium reaction potential

        # Expansion parameters
        self.c0 = 15
        self.c1 = 1/600
        self.c2 = 1/600

        # eR-RC parameters
        self.R0p = 0.041
        self.R0n = 0.041
        self.R1p = 0.158
        self.R1n = 0.158
        self.C1p = 38000
        self.C1n = 38000

        # eSOH parameters
        self.Cn = 5 # Ah
        self.Cp = 6 # Ah
        self.theta_n = 0.0
        self.theta_p = 1.0



class Simulation:

    def __init__(self, cell: Cell, sim_time_s: int):
        """
        Parameters
        ----------
        cell:       a Cell object
        sim_time_s: simulation time in seconds
        """

        self.cell = cell

        # Numerical details
        self.dt = 1.0
        self.t = np.arange(0, sim_time_s, self.dt)

        # Simulation parameters
        self.vmax = 4.2
        self.vmin = 3.0
        self.i_cv = 0.5 / 20 # CV hold current cut-off condition

        # Initialize output vectors
        self.i_app = np.zeros(self.t.shape)

        self.cycle_number = np.zeros(self.t.shape)
        self.step_number = np.zeros(self.t.shape)

        # eSOH states
        self.theta_n = mu.initialize(self.t, cell.theta_n)
        self.theta_p = mu.initialize(self.t, cell.theta_p)
        self.ocv_n   = mu.initialize(self.t, mu.Un(cell.theta_n))
        self.ocv_p   = mu.initialize(self.t, mu.Up(cell.theta_p))
        self.ocv     = mu.initialize(self.t, self.ocv_p[0] - self.ocv_n[0])
        self.vt      = mu.initialize(self.t, self.ocv_p[0] - self.ocv_n[0])
        self.i_int   = mu.initialize(self.t, 0)

        # RC states
        self.i_r1p   = mu.initialize(self.t, 0)
        self.i_r1n   = mu.initialize(self.t, 0)

        # SEI states
        self.eta_sei = mu.initialize(self.t, 0)
        self.j_sei_rxn = mu.initialize(self.t, 0)
        self.j_sei_dif = mu.initialize(self.t, 0)
        self.j_sei   = mu.initialize(self.t, 0)
        self.i_sei   = mu.initialize(self.t, 0)
        self.q_sei   = mu.initialize(self.t, 0)

        # Expansion states
        self.delta_sei = mu.initialize(self.t, cell.delta_SEI_0)
        self.delta_n = mu.initialize(self.t, mu.En(cell.theta_n))
        self.delta_p = mu.initialize(self.t, mu.Ep(cell.theta_p))
        self.expansion_rev = mu.initialize(self.t, 0)
        self.expansion_irrev = mu.initialize(self.t, 0)


    def step(self, k: int, mode: str, icc=0, icv=0, cyc_num=np.NaN, step_num=np.NaN):
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



        dQ = self.i_int[k] * self.dt / 3600 # Amp-hours
        self.theta_n[k + 1] = self.theta_n[k] + dQ / p.Cn
        self.theta_p[k + 1] = self.theta_p[k] - dQ / p.Cp

        # Equilibrium potential updates
        self.ocv_n[k + 1] = mu.Un(self.theta_n[k + 1])
        self.ocv_p[k + 1] = mu.Up(self.theta_p[k + 1])
        self.delta_n[k + 1] = mu.En(self.theta_n[k + 1])
        self.delta_p[k + 1] = mu.Ep(self.theta_p[k + 1])

        self.ocv[k + 1] = self.ocv_p[k+1] - self.ocv_n[k+1]

        # Current updates (branch current for RC element)
        self.i_r1p[k+1] =  np.exp(-self.dt/(p.R1p*p.C1p))  * self.i_r1p[k] + \
                          (1 - np.exp(-self.dt/(p.R1p*p.C1p))) * self.i_app[k]

        self.i_r1n[k+1] =  np.exp(-self.dt/(p.R1n*p.C1n))  * self.i_r1n[k] + \
                          (1 - np.exp(-self.dt/(p.R1n*p.C1n))) * self.i_app[k]

        # Terminal voltage update
        # Vt = Up + eta_p - Un + eta_n
        if mode == 'cc':
            self.vt[k+1] = self.ocv_p[k+1] + p.R1p * self.i_r1p[k] \
                             + p.R0p * self.i_app[k] - \
                              (self.ocv_n[k+1] - p.R1n * self.i_r1n[k] - \
                               p.R0n * self.i_app[k])
        elif mode == 'cv':
            self.vt[k+1] = self.vmax

        # SEI growth update
        eta_int = self.i_app[k] * p.R0n
        self.eta_sei[k+1] = eta_int + self.ocv_n[k+1] - p.U_SEI

        # Mixed reaction and diffusion limited SEI current density
        self.j_sei_rxn[k+1] = F * p.c_EC_bulk * p.k_SEI * \
                                np.exp( -p.alpha_SEI * F * self.eta_sei[k+1] / \
                                       (R * T) )
        self.j_sei_dif[k+1] = p.D_SEI * p.c_EC_bulk * F / \
                                (self.delta_sei[k]) # should this be k or k+1?
        self.j_sei[k+1] = - 1 / (1/self.j_sei_rxn[k+1] + 1/self.j_sei_dif[k+1])

        ## Current density to current conversion
        self.i_sei[k+1] = - self.j_sei[k+1] * (p.a_SEI * p.A_n * p.L_n)

        # Update the intercalation current for the next time step
        # Stoichiometry update; only include intercalation current
        sign = -np.sign(self.i_app[k])
        self.i_int[k+1] = self.i_app[k] + sign*self.i_sei[k]

        # Integrate SEI current to get SEI capacity
        self.q_sei[k+1] = self.q_sei[k] + self.i_sei[k+1] * self.dt / 3600

        # Update SEI thickness
        self.delta_sei[k+1] = self.delta_sei[k] + \
                              self.dt * (p.V_sei * \
                                        np.abs(self.j_sei[k+1]) ) / (2 * F)

        # Expansion update
        # Cathode and anode expansion function update
        self.expansion_rev[k+1] = p.c1 * self.delta_p[k+1] + \
                                 p.c2 * self.delta_n[k+1]
        self.expansion_irrev[k+1] = p.c0 * self.delta_sei[k+1]



    def plot(self, to_save=True):
        """
        Make a standard plot of the outputs
        """

        num_subplots = 11

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(16, num_subplots * 4),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        # Currents
        axs[0].axhline(y=0, linestyle='-', label='', color='k', linewidth=0.5)
        axs[0].plot(self.t/3600, self.i_app, color='k', marker='o', ms=1, label='$I_{app}$')
        axs[0].plot(self.t/3600, self.i_int, color='g', ls='--', label='$I_{int}$')
        axs[0].plot(self.t/3600, self.i_r1n, color='r', marker='o', ms=1, label='$I_{R_{1,n}}$')
        axs[0].plot(self.t/3600, self.i_r1p, color='b', ls='--', label='$I_{R_{1,p}}$')
        axs[0].set_ylabel('Current (A)')
        axs[0].legend()

        # Voltages and Potentials
        axs[1].plot(self.t/3600, self.vt, ls='--', c='k')
        axs[1].plot(self.t/3600, self.ocv, marker='o', ms=1, color='k')
        axs[1].legend(['$V_t$', '$V_{oc}$'])
        axs[1].set_ylabel('Voltage (V)')

        # Positive potential
        axs[2].plot(self.t/3600, self.ocv_p, marker='o', ms=1, color='b')
        axs[2].legend(['$U_p$'])
        axs[2].set_ylabel('V vs $Li/Li^+$')

        # Negative potential
        axs[3].plot(self.t/3600, self.ocv_n, marker='o', ms=1, color='r')
        axs[3].axhline(y=self.cell.U_SEI, linestyle='--', color='k')
        axs[3].legend(['$U_n$', f'$U_{{\mathrm{{SEI}}}}$ = {self.cell.U_SEI} V'])
        axs[3].set_ylabel('V vs $Li/Li^+$')

        # Electrode stoichiometries
        axs[4].plot(self.t/3600, self.theta_n, color='r', marker='o', ms=1)
        axs[4].plot(self.t/3600, self.theta_p, color='b', marker='o', ms=1)
        axs[4].axhline(y=1, linestyle='-', label='', color='k', linewidth=0.5)
        axs[4].axhline(y=0, linestyle='-', label='', color='k', linewidth=0.5)
        axs[4].legend([r'$\theta_n$', r'$\theta_p$'])
        axs[4].set_ylabel(r'$\theta$')
        axs[4].set_ylim((-0.1, 1.1))

        # Electrode expansion factors
        axs[5].set_ylabel(r'$\delta$')
        axs[5].plot(self.t/3600, self.delta_n, color='r', marker='o', ms=1)
        axs[5].plot(self.t/3600, self.delta_p, color='b', marker='o', ms=1)
        axs[5].legend([r'$\delta_n$', r'$\delta_p$'])

        # SEI expansion factor
        axs[6].plot(self.t/3600, self.delta_sei * 1e9, color='g', marker='o', ms=1)
        axs[6].legend([r'$\delta_{\mathrm{sei}}$'])
        axs[6].set_ylabel(r'$\delta_{\mathrm{sei}}$ [$nm$]')

        # Total cell expansion
        axs[7].set_ylabel(r'$\epsilon$ ($\mu$m)')
        axs[7].plot(self.t/3600, self.expansion_irrev*1e6, color='g', marker='o', ms=1, label='$\epsilon_{irrev}$')
        axs[7].plot(self.t/3600, (self.expansion_rev + self.expansion_irrev)*1e6,
                    color='k', marker='o', ms=1, label='$\epsilon_{irrev} + \epsilon_{rev}$')
        axs[7].legend()

        # SEI reaction current densities
        axs[8].set_yscale('log')
        axs[8].plot(self.t/3600, self.j_sei_rxn, color='r', marker='o', ms=1, label='$j_{sei,rxn}$')
        axs[8].plot(self.t/3600, self.j_sei_dif, color='b', marker='o', ms=1, label='$j_{sei,dif}$')
        axs[8].plot(self.t/3600, np.abs(self.j_sei), color='g', ls='--', lw=2, label='$j_{sei}$')
        axs[8].legend()
        axs[8].set_ylabel(r'$|j_{\mathrm{sei}}|$ [A/m$^2$]')

        # Total SEI reaction current
        axs[9].plot(self.t/3600, self.i_sei, color='g', marker='o', ms=1)
        axs[9].legend([r'$I_{\mathrm{sei}}$'])
        axs[9].set_ylabel(r'$I_{\mathrm{sei}}$ [A]')

        # Total SEI capacity
        axs[10].plot(self.t/3600, self.q_sei, color='g', marker='o', ms=1)
        axs[10].legend([r'$Q_{\mathrm{sei}}$'])
        axs[10].set_ylabel(r'$Q_{\mathrm{sei}}$ [Ah]')
        axs[10].set_xlabel('Time (hr)')

        if to_save:
            plt.savefig('outputs/figures/fig_formation_simulation.png',
                        bbox_inches='tight',
                        dpi=150)
