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
        self.k_SEI       = 1e-14      # m/s        SEI kinetic rate constant
        self.D_SEI       = 2e-16      # m2/s       SEI layer diffusivity
        self.R_n         = 20e-6      # m          Anode particle radius
        self.epsilon_n   = 0.7        # [-]        Anode solid material fraction
        self.a_SEI       = 3 * self.epsilon_n / self.R_n # 1/m   Anode specific surface area
        self.A_n         = 0.100      # m2         Anode active area
        self.U_SEI       = 0.4        # V          SEI equilibrium reaction potential

        # Expansion parameters
        self.c0 = 1/6
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
        self.t_vec = np.arange(0, sim_time_s, self.dt)

        # Simulation parameters
        self.vmax = 4.2
        self.vmin = 3.0
        self.i_cv = 0.5 / 20 # CV hold current cut-off condition

        # Initialize output vectors
        self.i_app_vec = np.zeros(self.t_vec.shape)

        # eSOH states
        self.theta_n_vec = mu.initialize_sim_vec(self.t_vec, cell.theta_n)
        self.theta_p_vec = mu.initialize_sim_vec(self.t_vec, cell.theta_p)
        self.ocv_n_vec   = mu.initialize_sim_vec(self.t_vec, mu.Un(cell.theta_n))
        self.ocv_p_vec   = mu.initialize_sim_vec(self.t_vec, mu.Up(cell.theta_p))
        self.ocv_vec     = mu.initialize_sim_vec(self.t_vec, self.ocv_p_vec[0] - self.ocv_n_vec[0])
        self.vt_vec      = mu.initialize_sim_vec(self.t_vec, self.ocv_p_vec[0] - self.ocv_n_vec[0])

        # RC states
        self.I_r1p_vec   = mu.initialize_sim_vec(self.t_vec, 0)
        self.I_r1n_vec   = mu.initialize_sim_vec(self.t_vec, 0)

        # SEI states
        self.eta_sei_vec = mu.initialize_sim_vec(self.t_vec, 0)
        self.j_sei_rxn_vec = mu.initialize_sim_vec(self.t_vec, 0)
        self.j_sei_dif_vec = mu.initialize_sim_vec(self.t_vec, 0)
        self.j_sei_vec   = mu.initialize_sim_vec(self.t_vec, 0)
        self.I_sei_vec   = mu.initialize_sim_vec(self.t_vec, 0)
        self.Q_sei_vec   = mu.initialize_sim_vec(self.t_vec, 0)

        # Expansion states
        self.delta_sei_vec = mu.initialize_sim_vec(self.t_vec, cell.delta_SEI_0)
        self.delta_n_vec = mu.initialize_sim_vec(self.t_vec, mu.En(cell.theta_n))
        self.delta_p_vec = mu.initialize_sim_vec(self.t_vec, mu.Ep(cell.theta_p))
        self.expansion_rev_vec = mu.initialize_sim_vec(self.t_vec, 0)
        self.expansion_irrev_vec = mu.initialize_sim_vec(self.t_vec, 0)


    def step(self, k: int, mode: str, icc=0, icv=0):
        """
        Run a single step.

        Parameters
        ----------
        k:         current time index
        mode:      'cc' or 'cv'
        icc: applied current in CC mode
        icv: current cut-off in CV mode

        """

        p = self.cell

        if mode == 'cc':
            self.i_app_vec[k] = icc
        elif mode == 'cv':
            self.i_app_vec[k] = ( self.vt_vec[k] - self.ocv_vec[k] - \
                           p.R1p * np.exp(-self.dt/(p.R1p*p.C1p)) * self.I_r1p_vec[k] - \
                           p.R1n * np.exp(-self.dt/(p.R1n*p.C1n)) * self.I_r1n_vec[k] ) / \
                        ( ( 1 - np.exp(-self.dt/(p.R1n*p.C1n)) ) * p.R1n + p.R0n + \
                          ( 1 - np.exp(-self.dt/(p.R1p*p.C1p)) ) * p.R1p + p.R0p )

        dQ = self.i_app_vec[k] * self.dt / 3600 # Amp-hours

        # Stoichiometry update
        self.theta_n_vec[k + 1] = self.theta_n_vec[k] + dQ / p.Cn
        self.theta_p_vec[k + 1] = self.theta_p_vec[k] - dQ / p.Cp

        # Equilibrium potential updates
        self.ocv_n_vec[k + 1] = mu.Un(self.theta_n_vec[k + 1])
        self.ocv_p_vec[k + 1] = mu.Up(self.theta_p_vec[k + 1])
        self.delta_n_vec[k + 1] = mu.En(self.theta_n_vec[k + 1])
        self.delta_p_vec[k + 1] = mu.Ep(self.theta_p_vec[k + 1])

        self.ocv_vec[k + 1] = self.ocv_p_vec[k+1] - self.ocv_n_vec[k+1]

        # Current updates (branch current for RC element)
        self.I_r1p_vec[k+1] =  np.exp(-self.dt/(p.R1p*p.C1p))  * self.I_r1p_vec[k] + \
                          (1 - np.exp(-self.dt/(p.R1p*p.C1p))) * self.i_app_vec[k]

        self.I_r1n_vec[k+1] =  np.exp(-self.dt/(p.R1n*p.C1n))  * self.I_r1n_vec[k] + \
                          (1 - np.exp(-self.dt/(p.R1n*p.C1n))) * self.i_app_vec[k]

        # Terminal voltage update
        # Vt = Up + eta_p - Un + eta_n
        if mode == 'cc':
            self.vt_vec[k+1] = self.ocv_p_vec[k+1] + p.R1p * self.I_r1p_vec[k] + p.R0p * self.i_app_vec[k] - \
                        (self.ocv_n_vec[k+1] - p.R1n * self.I_r1n_vec[k] - p.R0n * self.i_app_vec[k])
        elif mode == 'cv':
            self.vt_vec[k+1] = self.vmax

        # SEI growth update
        eta_int = self.i_app_vec[k] * p.R0n
        self.eta_sei_vec[k+1] = eta_int + self.ocv_n_vec[k+1] - p.U_SEI

        # Mixed reaction and diffusion limited SEI current density
        self.j_sei_rxn_vec[k+1] = F * p.c_EC_bulk * p.k_SEI * np.exp( -p.alpha_SEI * F * self.eta_sei_vec[k+1] / (R * T) )
        self.j_sei_dif_vec[k+1] = p.D_SEI * p.c_EC_bulk * F / (self.delta_sei_vec[k]) # should this be k or k+1?
        self.j_sei_vec[k+1] = - 1 / (1/self.j_sei_rxn_vec[k+1] + 1/self.j_sei_dif_vec[k+1])

        ## Current density to current conversion
        self.I_sei_vec[k+1] = - self.j_sei_vec[k+1] * (p.a_SEI * p.A_n * p.L_n)

        # Integrate SEI current to get SEI capacity
        self.Q_sei_vec[k+1] = self.Q_sei_vec[k] + self.I_sei_vec[k+1] * self.dt / 3600

        # Update SEI thickness
        self.delta_sei_vec[k+1] = self.delta_sei_vec[k] + \
                             self.dt * (p.V_sei * p.a_SEI * np.abs(self.j_sei_vec[k+1]) ) / (2 * F)

        # Expansion update
        # Cathode and anode expansion function update
        self.expansion_rev_vec[k+1] = p.c1 * self.delta_p_vec[k+1] + \
                                 p.c2 * self.delta_n_vec[k+1]
        self.expansion_irrev_vec[k+1] = p.c0 * self.delta_sei_vec[k+1]



    def plot(self, to_save=True):
        """
        Make a standard plot of the outputs
        """

        num_subplots = 10

        gridspec = dict(hspace=0.05, height_ratios=np.ones(num_subplots))

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1,
                                figsize=(12, num_subplots * 4),
                                gridspec_kw=gridspec,
                                sharex=True)

        [ax.grid(False) for ax in axs]

        # Currents
        axs[0].plot(self.t_vec/3600, self.i_app_vec, color='k', marker='o', ms=1)
        axs[0].plot(self.t_vec/3600, self.I_r1n_vec, color='r', marker='o', ms=1)
        axs[0].plot(self.t_vec/3600, self.I_r1p_vec, color='b', ls='--')
        axs[0].set_ylabel('Current (A)')
        axs[0].legend(['$I_{applied}$', '$I_{R_{1,n}}$', '$I_{R_{1,p}}$'])

        # Voltages and Potentials
        axs[1].plot(self.t_vec/3600, self.vt_vec, ls='--', c='k')
        axs[1].plot(self.t_vec/3600, self.ocv_vec, marker='o', ms=1, color='k')
        axs[1].legend(['$V_t$', '$V_{oc}$'])
        axs[1].set_ylabel('Voltage (V)')

        axs[2].plot(self.t_vec/3600, self.ocv_p_vec, marker='o', ms=1, color='b')
        axs[2].legend(['$U_p$'])
        axs[2].set_ylabel('V vs $Li/Li^+$')

        axs[3].plot(self.t_vec/3600, self.ocv_n_vec, marker='o', ms=1, color='r')
        axs[3].axhline(y=self.cell.U_SEI, linestyle='--', color='k')
        axs[3].legend(['$U_n$', f'$U_{{\mathrm{{SEI}}}}$ = {self.cell.U_SEI} V'])
        axs[3].set_ylabel('V vs $Li/Li^+$')

        axs[4].plot(self.t_vec/3600, self.theta_n_vec, color='r', marker='o', ms=1)
        axs[4].plot(self.t_vec/3600, self.theta_p_vec, color='b', marker='o', ms=1)
        axs[4].legend([r'$\theta_n$', r'$\theta_p$'])
        axs[4].set_ylabel(r'$\theta$')
        axs[4].set_ylim((-0.1, 1.1))

        axs[5].set_ylabel(r'$\delta$')
        axs[5].plot(self.t_vec/3600, self.delta_n_vec, color='r', marker='o', ms=1)
        axs[5].plot(self.t_vec/3600, self.delta_p_vec, color='b', marker='o', ms=1)
        axs[5].legend([r'$\delta_n$', r'$\delta_p$'])

        axs[6].plot(self.t_vec/3600, self.delta_sei_vec, color='r', marker='o', ms=1)
        axs[6].legend([r'$\delta_{\mathrm{sei}}$'])
        axs[6].set_ylabel(r'$\delta_{\mathrm{sei}}$ [$m$]')

        axs[7].set_ylabel(r'$\epsilon$ ($\mu$m)')
        axs[7].plot(self.t_vec/3600, self.expansion_irrev_vec*1e6, color='b', marker='o', ms=1, label='$\epsilon_{irrev}$')
        axs[7].plot(self.t_vec/3600, (self.expansion_rev_vec + self.expansion_irrev_vec)*1e6, color='k', marker='o', ms=1, label='$\epsilon_{irrev} + \epsilon_{rev}$')
        axs[7].legend()

        axs[8].set_yscale('log')
        axs[8].plot(self.t_vec/3600, self.j_sei_rxn_vec, color='r', marker='o', ms=1, label='$j_{sei,rxn}$')
        axs[8].plot(self.t_vec/3600, self.j_sei_dif_vec, color='b', marker='o', ms=1, label='$j_{sei,dif}$')
        axs[8].plot(self.t_vec/3600, np.abs(self.j_sei_vec), color='k', ls='--', label='$j_{sei}$')
        axs[8].legend()
        axs[8].set_ylabel(r'$|j_{\mathrm{sei}}|$ [A/m$^2$]')

        axs[9].plot(self.t_vec/3600, self.Q_sei_vec, color='r', marker='o', ms=1)
        axs[9].legend([r'$Q_{\mathrm{sei}}$'])
        axs[9].set_ylabel(r'$Q_{\mathrm{sei}}$ [Ah]')
        axs[9].set_xlabel('Time (hr)')

        if to_save:
            plt.savefig('outputs/figures/fig_formation_simulation.png', bbox_inches='tight',
                    dpi=150)
