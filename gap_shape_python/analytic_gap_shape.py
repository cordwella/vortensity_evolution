import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import iv, ive, kv, kve

from gap_shape_python.calc_shock import compute_f_dep


# Implements the analytic solutions for the initial evolution of a gap in a 
# protoplanetary disc as in Cordwell & Rafikov 2024 

def non_local_change_in_density(R, p, q, h_p, m_p, f_r=None):
    """
    Use the global solution for the initial evolution of a gap in a 
    protoplanetary disc.
    
    See equation C4 in Cordwell & Rafikov (2024)
    
    Inputs:
    - R, input array of locations to evaluate
    - p, background gradient in surface density (Sigma_0 = R^(-p))
    - q, background gradient in disc temperature (c_s = c_{s, p} R^(-q) )
    - h_p, scale height of the disc at the planet
    - m_p, mass of the planet in thermal masses
    - f_r (optional), pre-evaluated angular momentum deposition function
    
    Returns:
    - d\sigma / dt, array with the same shape as R
    """
    
    if q == 0.5:
        raise Exception("Solution not provided for q = 1/2")


    if f_r is None:
        # Allow override
        f_r = compute_f_dep(R, p, h_p, m_p)[0]

    A = h_p

    S = 1/np.pi * np.gradient(R**(1/2 - p) * f_r)/np.gradient(R)
    
    x = R**(q - 1/2)/( (q - 1/2) * A)
    x = x.astype(complex)
    
    beta = np.abs((p + 2 * q - 3)/(1 - 2 * q))

    I_3 = iv(beta, np.abs(x))
    K_3 = kv(beta, np.abs(x))
    
    if q < 1/2:
        i_int = cumulative_trapezoid((I_3 * S * R**(p/2 + q - 3/2))[::-1], R[::-1], initial=0)[::-1]
        k_int = cumulative_trapezoid((K_3 * S * R**(p/2 + q - 3/2)), R, initial=0)
    else:
        i_int = cumulative_trapezoid((I_3 * S * R**(p/2 + q - 3/2)), R, initial=0)
        k_int = cumulative_trapezoid((K_3 * S * R**(p/2 + q - 3/2))[::-1], R[::-1], initial=0)[::-1]
    
    return A **(-2) * R**(p/2 + q - 3/2) * (
        I_3 * k_int - K_3 * i_int)/(q - 1/2)


def local_approximation_change_in_density(R, p, h_p, m_p, f_r=None):
    """
    Use the local solution for the initial evolution of a gap in a 
    protoplanetary disc.
    
    See equation X in Cordwell & Rafikov (2024). 
    
    Inputs:
    - R, input array of locations to evaluate
    - p, background gradient in surface density (Sigma_0 = R^(-p))
    - h_p, scale height of the disc at the planet
    - m_p, mass of the planet in thermal masses
    - f_r (optional), pre-evaluated angular momentum deposition function
    
    Returns:
    - dX/dt, array with the same shape as R
    """
    
    if f_r is None:
        # Allow override
        f_r = compute_K_R_CR21(R, p, h_p, m_p)[0]

    l_sh = 0.8 * h_p * (5/6 * m_p)**(-2/5)

    plus_e = np.exp(R / h_p)
    minus_e = np.exp(-R / h_p)
    
    m_int = cumulative_trapezoid(f_r * minus_e, R, initial=0)
    p_int = cumulative_trapezoid(f_r * plus_e, R, initial=0)
    a = (plus_e * m_int + minus_e * p_int)
    
    c_1 = - m_int[-1]
    c_2 = m_int[-1]

    return (1/(2 * np.pi * h_p**2)) * (
        c_1 * plus_e + c_2 * minus_e + a)


def no_pressure_support_change_in_density(R, p, h_p, m_p, f_r=None):
    """    
    Inputs:
    - R, input array of locations to evaluate
    - p, background gradient in surface density (Sigma_0 = R^(-p))
    - m_p, mass of the planet in thermal masses
    - f_r (optional), pre-evaluated angular momentum deposition function
    
    Returns:
    - dX/dt, array with the same shape as R
    """

    if f_r is None:
        # Allow override
        f_r = compute_f_dep(R, p, h_p, m_p)[0]
        
    inner_source = f_r/(np.pi * R**(p -1/2))

    return -1 * np.gradient(inner_source)/np.gradient(R) * R**(p - 1)


def no_pressure_support_change_in_density_local(R, p, h_p, m_p, f_r=None):
    """    
    Inputs:
    - R, input array of locations to evaluate
    - p, background gradient in surface density (Sigma_0 = R^(-p))
    - m_p, mass of the planet in thermal masses
    - f_r (optional), pre-evaluated angular momentum deposition function
    
    Returns:
    - dX/dt, array with the same shape as R
    """

    if f_r is None:
        # Allow override
        f_r = compute_f_R_CR21(R, p, h_p, m_p)[0]
    
    return -1 * np.gradient(f_r)/np.gradient(R) / np.pi