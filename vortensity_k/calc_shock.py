import numpy as np
import scipy.integrate as integrate

from .constants import FIT_PARAMETERS

# NOTE: Please see the associated implementation Jupyter notebook for
# references and equation sources

""" Functions to generate the density profile of a disk perturbed by
a sub thermal mass planet """


def compute_tau_R_keplerian(R, p, h_p, m_p_by_m_th):
    """Compute tau(R) where R is either a single value or a numpy array

    R must be in units of the planetary radius

    """
    # TODO: provide non-Keplerian version of this
    # Using equation 18
    def of_s(s):
        return (np.abs(s**(3/2) - 1))**(3/2) * s**(p/2 - 11/4)


    # This is inefficent, but it's also very quick so I'm not fixing it
    # effectively a for loop
    perform_integral = np.vectorize(
        lambda x: integrate.quad(of_s, 1, x))
        
    t = perform_integral(R)

    tau = (np.sign(R - 1) * (3/(2**(5/4))) * m_p_by_m_th *
           h_p**(-5/2) * np.abs(t[0]))

    # Formally tau isn't absolute values, however for the computation
    # we're doing here it fixes some issues
    return np.abs(tau)


def compute_delta_chi(R, tau, tau_0, inner_smooth=True):
    """Compute delta chi(tau) using the fit parameters"""

    # Compute the inner fit
    tau_squiggle = np.abs(tau - tau_0)

    A, tau_b, alpha_1, alpha_2, delta = FIT_PARAMETERS['inner']

    inner = (A * (tau_squiggle/tau_b)**(-1 * alpha_1) *
             (1 + (tau_squiggle/tau_b)**(1/delta))**(
                (alpha_1 - alpha_2)*delta))

    # Compute the outer fit
    A, tau_b, alpha_1, alpha_2, delta = FIT_PARAMETERS['outer']
    outer = (A * np.abs(tau_squiggle/tau_b)**(-1 * alpha_1) *
             (1 + (tau_squiggle/tau_b)**(1/delta))**(
                (alpha_1 - alpha_2)*delta))

    chi = (inner * (R <= 1).astype(int) + outer * (R > 1).astype(int)) * (
        tau >= 1.05 * tau_0).astype(int)
    # Only defined for tau > tau_0, zero otherwise
    # however this is currently adding some numerical instabilities so lets 
    # add in a small amount of smoothing inside - 5% of the inside of tau_0        
    
    return chi


def compute_omega_k_squared(R):
    """Give the keplerian angular speed squared, using  G = 1, M_star=1"""
    return R**(-3)


def compute_K_R(R, p, h_p, m_p):
    
    # Mass is in thermal units
    # Compute thermal mass at the planet
    m_th = h_p**3  # M_th = (H_p/R_p)**3 M_star

    # Define the sound speed next to the planet by the relationship
    # m_th = c_s**3/(omega * G)
    m_p_by_m_th = m_p
    
    # Planetary mass in true units
    m_p = m_p * m_th
    
    sigma_0 = R**(-p)  # Sigma_p * (R/R_p) ** (-p)

    # Compute scaling factor tau
    tau = compute_tau_R_keplerian(R, p, h_p, m_p_by_m_th)

    tau_0 = 1.89 * m_p_by_m_th  # equation 16 in CR21
    delta_chi = compute_delta_chi(R, tau, tau_0)

    epsilon = delta_chi * m_p_by_m_th * (np.abs(R**(-3/2) - 1)/(R**(1 - p) * h_p * np.sqrt(2)))**(1/2)
    
    psi = (epsilon * (2 + epsilon) - 2 * (1 + epsilon) * np.log(1 + epsilon))/(2 * (1 + epsilon))
    
    k_R = np.sign(1 - R**(-3/2)) * 1 * R * h_p**2 * psi * 2 * np.pi
    
    return k_R, tau, delta_chi
