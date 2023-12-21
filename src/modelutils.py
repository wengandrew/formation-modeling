"""
Utility functions to aid in battery simulation

Mostly contains lookup functions and helper functions.
"""

import numpy as np
from scipy import interpolate

def initialize(time_vec, initial_val=np.NaN):
    """
    Initialize a vector to store simulation results.
    Allows a user to specify the initial value.
    Set the remaining values to NaN.

    NaN is preferred to indicate 'no value present'

    Parameters
    ---------
    time_vec:    a Numpy array of dimension n
    initial_val: the initial value

    Returns
    ---------
    output_vec: a Numpy array of dimension n
    """

    output_vec = np.empty(len(time_vec))
    output_vec[:] = np.NaN
    output_vec[0] = initial_val

    return output_vec



def EnSei(sto):
    """
    Graphite expansion function adapted from
    Kupper2018 (JES 165(14))
    """

    # Effective Young's modulus satisfying boundary condition at
    # stoichiometry = 1
    max_strain = 0.1318
    max_stress = 24.172

    E = max_strain / max_stress

    return stressSEI(sto) * E


def stressSEI(sto):
    """
    SEI stress function from Kupper2018 (JES 165(14))

    Returns tangential stress on the SEI in units of MPa
    """

    return (-931.0/5) * sto ** 5 \
             + 329.99 * sto ** 4 \
             - 67.228 * sto ** 3 \
             - 120.29 * sto ** 2 \
               + 67.9 * sto


def En(sto):
    """
    Graphite expansion function

    Cite: Mohtat et al. 2020. “Differential Expansion and Voltage Model for
    Li-Ion Batteries at Practical Charging Rates.” Journal of the
    Electrochemical Society 167 (11): 110561.

    Parameters
    ----------
    sto : stoichiometry of material (lithium fraction)
    """

    if sto < 0.12:
        expansion = 0.2 * sto
    elif sto >= 0.12 and sto < 0.18:
        expansion = 0.16 * sto + 5e-3 - 0.0002
    elif sto >= 0.18 and sto < 0.24:
        expansion = 0.17 * sto + 3e-3
    elif sto >= 0.24 and sto < 0.50:
        expansion = 0.05 * sto + 0.03 + 1.8e-3
    elif sto >= 0.5:
        expansion = 0.15 * sto - 0.02 + 1.8e-3

    return expansion

def Ep(sto):
    """
    Nickel Manganese Cobalt Oxide (NMC) expansion function

    Cite: Mohtat et al. 2020. “Differential Expansion and Voltage Model for
    Li-Ion Batteries at Practical Charging Rates.” Journal of the
    Electrochemical Society 167 (11): 110561.

    Parameters
    ----------
    sto : stoichiometry of material (lithium fraction)
    """

    return -1.1e-2 * (1 - sto)


def UnGr(sto):
    """
    UMBL2022FEB Un function from Hamid Mohavedi, April 2023
    """

    # Make an adjustment to satisfy pre-formation boundary condition
    # (When the neg. electrode is completely empty, enforce that the potential
    # is some fixed value such that, when combined with the positive electrode
    # potential, describes the measured pre-formation full cell potential.
    # This requires extrapolating the function slightly.
    gamma = -0.02694418
    sto = (1 - gamma)*sto + gamma

    var= [-0.014706865941596, -0.012143997194139, -0.002418437987814,-0.008309680311046,-0.012498890102608, \
    -0.053808682683538, 0.672164169443950, 0.097531836156448,0.150321382880000,  0.178875736938146, \
    0.250298777953980, 0.211789560126373, 0.515765253254034, 0.996096565665399, -0.002990596665583, \
    0.016114544615164, 0.017934277004896, 0.006850209194758, 0.013282726871941, 0.018011720489539, \
    0.048224536960530, -0.024336868170844]

    p_eq = var[0:8]
    a_eq = var[8:15]
    b_eq = var[15:22]

    u_eq2 = p_eq[7] + p_eq[6]*np.exp((sto - a_eq[6])/b_eq[6])

    for i in np.arange(6):
        u_eq2 += p_eq[i]*np.tanh((sto - a_eq[i])/b_eq[i])

    return u_eq2


def UpNMC622(sto):
    """
    UMBL2022FEB Up function from Hamid Mohavedi, April 2023
    """

    a1 = 2.992
    a2 = -2.098
    a3 = -0.6943
    a4 = 4.341
    a5 = -3.883
    a6 = 0.611
    a7 = 0.8258
    b1 = 0.4484
    b2 = 0.4757

    u_eq = a1 + a2*sto + a3*sto**2 + a4*sto**3 + \
           a5*sto**4 + a6*sto**5 + a7*np.exp(b1*sto + b2)

    return u_eq


def Up(sto, adjust=True):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.
    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : Stochiometry of material (lithium fraction)
    adjust (boolean): if True then make stoichiometry adjustment
    """

    # Make an adjustment to satisfy pre-formation boundary condition
    # (When the pos. electrode is completely full, enforce that the potential
    # is some fixed value such that, when combined with the positive electrode
    # potential, describes the measured pre-formation full cell potential.
    # This requires extrapolating the function slightly.
    if adjust:
        beta = 1.00206269
        sto = beta*sto

    # Don't extrapolate too much with these functions
    if np.any(sto < -0.1):
        raise ValueError(f'stoichiometry ({sto}) is too small.')

    if np.any(sto > +1.1):
        raise ValueError(f'Stoichiometry ({sto}) is too large.')

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto ** 2)
        - 2.0843 * (sto ** 3)
        + 3.5146 * (sto ** 4)
        - 2.2166 * (sto ** 5)
        - 0.5623e-4 * np.exp(109.451 * sto - 100.006)
    )

    return u_eq


def Un(sto, adjust=True):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].
    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)

    Parameters
    ---------
    sto : Stoichiometry of material (lithium fraction0)
    adjust (boolean): if True then make stoichiometry adjustment
    """

    # Make an adjustment to satisfy pre-formation boundary condition
    # (When the neg. electrode is completely empty, enforce that the potential
    # is some fixed value such that, when combined with the positive electrode
    # potential, describes the measured pre-formation full cell potential.
    # This requires extrapolating the function slightly.
    if adjust:
        gamma = -0.01185456
        sto = (1 - gamma)*sto + gamma

    # Don't extrapolate too much with these functions
    if np.any(sto < -0.1):
        raise ValueError(f'stoichiometry ({sto}) is too small.')

    if np.any(sto > +1.1):
        raise ValueError(f'Stoichiometry ({sto}) is too large.')

    u_eq = (
        0.063
        + 0.8 * np.exp(-75 * (sto + 0.001))
        - 0.0120 * np.tanh((sto - 0.127) / 0.016)
        - 0.0118 * np.tanh((sto - 0.155) / 0.016)
        - 0.0035 * np.tanh((sto - 0.220) / 0.020)
        - 0.0095 * np.tanh((sto - 0.190) / 0.013)
        - 0.0145 * np.tanh((sto - 0.490) / 0.020)
        - 0.0800 * np.tanh((sto - 1.030) / 0.055)
    )

    return u_eq


def state_update_ocv_r_2rc(I, T, Rs, R1, R2, C1, C2):
    """
    State update equations for the OCV-R-RC model

    Parameters
    ---------
    I: input current (A), positive is discharge
    T: input temperature in Celsius
    Rs: series resistance in Ohms
    R1: R1 in Ohms
    R2: R2 in Ohms
    C1: C1 in Farads
    C2: C2 in Farads

    Outputs:
    x = [VT, z, V1, V2]^T

    """

    return None


def ocv(z):
    """
    Return the open circuit voltage at a specific SOC

    Parameters
    ---------
    z: state of charge (0-1) (float)

    Returns
    ---------
    V: open circuit voltage (V)

    """

    assert isinstance(z, float) or isinstance(z, int), \
           'Only floats or integers accepted for "z".'

    if z < 0:
        raise ValueError('SOC cannot be less than zero.')

    if z > 1:
        raise ValueError('SOC cannot be greater than one.')

    alpha   = 1.2
    V0      = 2
    beta    = 20
    gamma   = 0.6
    zeta    = 0.3
    epsilon = 0.01

    # Handle division-by-zero issue
    temp = 0 if z == 1 else np.exp(-epsilon/(1 - z))

    OCV = V0 + alpha * (1 - np.exp(-beta * z)) + gamma * z + \
               zeta * (1 - temp)

    return OCV


def update_esoh(z, q_max, x100, y100, Cn, Cp):
    """
    OCV update equations for the eSOH OCV model

    Parameters:
    ---------
    z     : input state of charge
    q_max : maximum battery capacity in Ah
    x100  : neg. electrode stoichiometry at z = 1
    y100  : pos. electrode stoichiometry at z = 1
    Cn    : neg. electrode capacity
    Cp    : pos. electrode capacity

    Returns:
    ---------
    x     : updated neg. electrode stoichiometry
    y     : updated pos. electrode stoichiometry
    un    : updated neg. electrode potential
    up    : updated pos. electrode potential
    ocv   : updated full cell potential
    """

    Qd = (1 - z) * q_max
    y = y100 + Qd / Cp
    x = x100 - Qd / Cn
    up = Up(y)
    un = Un(x)
    ocv = up - un

    return (x, y, un, up, ocv)


def update_ocv(z):
    """
    OCV upate equation for the basic OCV model

    Parameters:
    ---------
    z: input state of charge

    Returns:
    ---------
    OCV at z
    """

    return ocv(z)

def decompose_resistance_curve(capacity_vec,
                              resistance_vec,
                              frac_cathode_resistance=0.7,
                              capacity_shift_ah=0,
                              resistance_growth_rate=0,
                              adjust_for_resistance_change=False,
                              swap_cathode_anode=False,
                              split_proportionally=False):
    """
    Take a full cell capacity vs resistance curve and decompose it into half-cell
    resistances based on some scaling assumptions.

    We will apportion the resistances to qualitatively match the empirical results.

    A more quantitative break-down of resistances will require more careful data
    analysis of the cathode and anode contributions to the total cell resistance.
    This will require some more experiments which we will leave as future work.

    The advantage of this model construction is that it lets us easily study model
    sensitivities (i.e. hypothetical scenarios of different cathode/anode breakdown
    of the total cell resistance).

    Parameters
    ---------
    capacity_vec (np.array)
      vector of capacity for full cell

    resistance_vec (np.array)
      vector of resistance for full cell in Ohms

    frac_cathode_rct (0-1)
      fraction of total measured resistance attributed to cathode charge tranfser

    capacity_shift_ah (float)
      capacity corresponding to extra lithium lithium lost to SEI during formation

    resistance_growth_rate (float)
      SEI resistance growth per Ah of Li

    adjust_for_resistance_change (boolean)
      if True then will make an additional adjustment to the curve to account for
      intrinsic resistance growth

    swap_cathode_anode (boolean)
      if True then swap the cathode and anode resistance curves (for sensitivity analysis)

    split_proportionally (boolean)
      if True then split the resistance proportionally between cathode and anode
      based on frac_cathode_resistance


    Outputs
    ---------
    a dictionary containing:

      capacity_expanded
        an updated capacity vector that matches the dimensions of the shifted resistances

      resistance_full_modeled
        modeled full cell resistance after shifting (Ohms)

      resistance_cathode
        modeled cathod charge transfer resistance (Ohms)

      resistance_other
        modeled 'other' resistance (Ohms)


    Invariants
    ---------

    resistance_other + resistance_cathode_shifted = resistance_full_modeled

    """

    # Definition of "base" resistance
    capacity_threshold = 1
    resistance_base = np.min(resistance_vec[capacity_vec < capacity_threshold])

    # Reference resistance (intermediate step for constructing cathode resistance)
    resistance_ref = (1 - frac_cathode_resistance) * resistance_base * np.ones(np.size(resistance_vec))

    # Construct the cathode charge transfer resistance curve
    # Assumes:
    # - Cathode inherits all of the resistances at low capacities up to some reference point
    # - Cathode resistance flattens out after the capacity threshold
    resistance_cathode = resistance_vec - resistance_ref
    resistance_cathode[capacity_vec > capacity_threshold] = \
       resistance_cathode[capacity_vec <= capacity_threshold][-1]

    # Definition of R_other
    resistance_other = resistance_vec - resistance_cathode

    # Define shifted cathode charge transfer resistance

    # Shifting the resistance curve!

    # Expand the the capacity vector to include negative values
    cap_vec_min = 0
    cap_vec_max = np.max(capacity_vec)
    cap_vec_diff = np.diff(capacity_vec)[0]

    capacity_vec_expanded = np.arange(cap_vec_min, cap_vec_max + cap_vec_diff, cap_vec_diff)

    fn = interpolate.interp1d(capacity_vec, resistance_cathode, bounds_error=False, fill_value='extrapolate')
    resistance_cathode_shifted = fn(capacity_vec_expanded + capacity_shift_ah)

    fn2 = interpolate.interp1d(capacity_vec, resistance_other, bounds_error=False, fill_value='extrapolate')
    resistance_other_expanded = fn2(capacity_vec_expanded)

    if adjust_for_resistance_change:
        resistance_other += resistance_growth_rate * capacity_shift_ah

    # Calculated the shifted full cell resistance
    resistance_full_modeled = resistance_other_expanded + resistance_cathode_shifted

    # Assert invariants hold
#     assert np.all(resistance_cathode_shifted + resistance_other_expanded == resistance_full_modeled)

    output = dict()
    output['resistance_full_modeled'] = resistance_full_modeled

    output['capacity_expanded'] = capacity_vec_expanded
    output['resistance_cathode'] = resistance_cathode_shifted
    output['resistance_other'] = resistance_other_expanded

    # Define some simple operations for sensitivity studies

    # The cathode and anode curves are swapped
    if swap_cathode_anode:
        output['resistance_cathode'] = resistance_other_expanded
        output['resistance_other'] = resistance_cathode_shifted

    # Ignore all of that math we just did and simply split the total measured resistance
    if split_proportionally:
        output['resistance_cathode'] = resistance_full_modeled * frac_cathode_resistance
        output['resistance_other'] = resistance_full_modeled * (1 - frac_cathode_resistance)

    return output


def get_resistance_curves():
    """
    Return positive and negative electrode resistance curves
    """

    # Data source: data/processed/hppc_1.csv
    res_p = np.array([0.08179587, 0.05349805, 0.02520035, 0.01853486, 0.01554122,
        0.01360811, 0.01241951, 0.01167911, 0.01119615, 0.0111314 ,
        0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 ,
        0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 ,
        0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 , 0.0111314 ,
        0.0111314 , 0.0111314 ])

    theta_p = np.linspace(1, 0, len(res_p))

    f_res_p = interpolate.interp1d(theta_p, res_p, kind='quadratic', fill_value='extrapolate')


    res_n = np.array([0.0047706 , 0.0047706 , 0.0047706 , 0.0047706 , 0.0047706 ,
        0.0047706 , 0.0047706 , 0.0047706 , 0.0047706 , 0.0047706 ,
        0.00464267, 0.00477042, 0.00493099, 0.00531774, 0.00638144,
        0.00599402, 0.00467134, 0.00435176, 0.00431948, 0.00428838,
        0.00425448, 0.00432057, 0.00435294, 0.00428712, 0.00444787,
        0.0044497 , 0.00458338])

    theta_n = np.linspace(0, 1, len(res_n))

    f_res_n = interpolate.interp1d(theta_n, res_n, kind='quadratic', fill_value='extrapolate')


    return (f_res_p, f_res_n)
