"""
Equivalent circuit models for batteries
"""

import numpy as np

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
        expansion = 0.16 * sto + 5e-3
    elif sto >= 0.18 and sto < 0.24:
        expansion = 0.17 * sto + 3e-3
    elif sto >= 0.24 and sto < 0.50:
        expansion = 0.05 * sto + 0.03
    elif sto >= 0.5:
        expansion = 0.15 * sto - 0.02

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


def Up(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.
    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : Stochiometry of material (lithium fraction)
    """

    # Don't extrapolate too much with these functions
    if sto < -0.1:
        raise ValueError(f'stoichiometry ({sto}) is too small.')

    if sto > +1.1:
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


def Un(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].
    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)

    Parameters
    ---------
    sto : Stoichiometry of material (lithium fraction0)
    """

    # Don't extrapolate too much with these functions
    if sto < -0.1:
        raise ValueError(f'stoichiometry ({sto}) is too small.')

    if sto > +1.1:
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

