"""
Utility functions for dV/dQ analysis
"""

import numpy as np
from scipy import interpolate

from src import modelutils as mu


def f_pos_ocv(sto):
    """
    NMC cathode OCP. Delegates to modelutils.Up with adjust=False (no
    stoichiometry shift) so the raw polynomial is used directly.
    """
    return mu.Up(sto, adjust=False)


def f_neg_ocv(sto):
    """
    Graphite anode OCP. Delegates to modelutils.Un with adjust=False (no
    stoichiometry shift) so the raw polynomial is used directly.
    """
    return mu.Un(sto, adjust=False)


def esoh_to_voc(x100, y100, Cn, Cp, q):
    """
    Convert (x100, y100, Cn, Cp, C) to Voc(q), Un(x), Up(y)

    Parameters:
    ---------
    x100: negative electrode stoichiometry at 100% SOC
    y100: positive electrode stoichiometry at 100% SOC
    Cn:   negative electrode capacity (Ah)
    Cp:   positive electrode capacity (Ah)
    q:    full cell capacity vector (Ah)

    Outputs:
    ---------
    Voc(q):
    Un(x):
    Up(y):

    """

    # Map full cell capacity to pos. (y) and neg. (x) stoichiometries
    # y = y100 + (np.max(q) - q) / Cp
    # x = x100 - (np.max(q) - q) / Cn

    x0 = x100 - np.max(q)/Cn
    y0 = y100 + np.max(q)/Cp

    # Alternate method if x0 and y0 are given
    y = y0 - q / Cp
    x = x0 + q / Cn

    # Calculate the full cell open circuit potential
    Voc = f_pos_ocv(y) - f_neg_ocv(x)

    return (Voc, f_pos_ocv(y), f_neg_ocv(x))


def deg_to_voc_graphical(lam_pe, lam_ne, lli, Cp, Cn):
    """
    Convert degradation vector (LAM_PE, LAM_NE, LLI) to Voc

    Parameters:
    ---------
    lam_pe  : loss of active material in the positive electrode (%)
    lam_ne  : loss of active material in the negative electrode (%)
    lli     : loss of lithium inventory (Ah)
    Cp      : initial positive electrode capacity (Ah)
    Cn      : initial negative electrode capacity (Ah)

    Outputs:
    ---------
    Voc(Q)   : full cell open circuit potential
    Un(Q)    : negative electrode open circuit potential
    Up(Q)    : positive electrode open circuit potential
    capacity : shared capacity basis (Ah)
    """

    # Initial positive and negative stoic curves
    y_vec = np.linspace(0, 1, 1000)
    Up_vec = f_pos_ocv(1 - y_vec) # Cathode stoic is flipped!

    x_vec = np.linspace(0, 1, 1000)
    Un_vec = f_neg_ocv(x_vec)

    # Convert the stoic curves to capacity curves
    Cp_vec = np.linspace(0, Cp * (1 - lam_pe), 1000) - lli
    Cn_vec = np.linspace(0, Cn * (1 - lam_ne), 1000)

    # Expanded capacity vector
    capacity = np.linspace(np.min([np.min(Cp_vec), np.min(Cn_vec)]),
                           np.max([np.max(Cp_vec), np.max(Cn_vec)]), 1000)

    # Expanded voltage vectors
    fp = interpolate.interp1d(Cp_vec, Up_vec, bounds_error=False)
    Up_vec_interp = fp(capacity)

    fn = interpolate.interp1d(Cn_vec, Un_vec, bounds_error=False)
    Un_vec_interp = fn(capacity)

    Voc = Up_vec_interp - Un_vec_interp

    return (Voc, Un_vec_interp, Up_vec_interp, capacity)

    # Extract theta values
    # VMIN = 3.0
    # VMAX = 4.2

    # q0_ref = np.interp(VMIN, U, C)
    # q100_ref = np.interp(VMAX, U, C)

    # theta_p_0 = 1 - (np.interp(q0_ref, Cp_vec, Up_vec)) / Cp
    # theta_p_100 = 1 - (np.interp(q100_ref, Cp_vec, Up_vec)) / Cp

    # theta_n_0 = q0_ref / Cn
    # theta_n_100 = q100_ref / Cn
