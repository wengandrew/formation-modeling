"""
Equivalent circuit models for batteries
"""

import numpy as np

def initialize_sim_vec(time_vec, initial_val=np.NaN):
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

    if sto < 0:
        raise ValueError('stoichiometry cannot be less than zero.')

    if sto > 1:
        raise ValueError('Stoichiometry cannot be greater than one.')

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

    if sto < 0:
        raise ValueError('stoichiometry cannot be less than zero.')

    if sto > 1:
        raise ValueError('Stoichiometry cannot be greater than one.')

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

