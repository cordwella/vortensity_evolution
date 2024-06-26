import numpy as np
import scipy.integrate as integrate

from .calc_shock import compute_delta_chi, compute_tau_R_keplerian
from .constants import FIT_PARAMETERS

# NOTE: Please see the associated implementation Jupyter notebook for
# references and equation sources

""" Functions to generate the density profile of a disk perturbed by
a sub thermal mass planet """

def compute_omega_k_squared(R):
    """Give the keplerian angular speed squared, using  G = 1, M_star=1"""
    return R**(-3)


def compute_delta_zeta(R, p, h_p, m_p,
                       linear_shock=False, mass_in_thermal=True,
                       use_cr_21=True):
    """
    Find the vortensity jump at a shock front for a given system

    R - Radial positions to compute at in units of planet radii
    c_s - sound speed of the disk in units of?????
    p - power law slope of density (unitless)
    m_p - planet mass in units of the stellar/thermal mass
    h_p - height of the disk divided by radius of the disk at the planet
    
    This implements the equations as in CR21. To reproduce the correct
    physical output this must be multiplied by 2 pi.
    """

    # Basic properties of the system
    # using Omega_k(R_p) = Sigma_0(R_p) = G = M_star = 1
    # Radial position of the planet is one by this definition (given kepler)

    # Compute thermal mass at the planet
    m_th = h_p**3  # M_th = (H_p/R_p)**3 M_star

    # Define the sound speed next to the planet by the relationship
    # m_th = c_s**3/(omega * G)
    c_s = m_th**(1/3)
    m_p_by_m_th = 0

    if mass_in_thermal:
        # Rescale to be in stellar masses
        # which is easy as we are dealing with units of M_stellar = 1
        m_p_by_m_th = m_p
        m_p = m_p * m_th

    else:
        m_p_by_m_th = m_p / m_th

    # Unperturbed angular velocity of disk fluid
    # taking to be keplerian
    # omega_0 = np.sqrt(omega_k_squared(R))  # - p * c_s**2)
    # Unperturbed surface density
    sigma_0 = R**(-p)  # Sigma_p * (R/R_p) ** (-p)

    # Compute scaling factor tau
    tau = compute_tau_R_keplerian(R, p, h_p, m_p_by_m_th)

    # Comptute shock strength
    tau_0 = 1.89 * m_p_by_m_th  # equation 16
    delta_chi = compute_delta_chi(R, tau, tau_0, use_cr_21)

    # Combine to get the vortensity jump
    # Compute B(R) and C(R)
    # CR21 C4
    b_R = np.sqrt(R**(p-1) * np.abs(R**(-3/2) - 1))

    c_R = 0
    if linear_shock:
        # CR21 C6
        c_R = np.sign(R - 1) / np.sqrt(
            1 + h_p**(-2) * R**2 * ((R**(-3/2) - 1)**2))
    else:
        delta_phi = 1
        d_tau_d_r = np.gradient(tau, R)
        correction = delta_phi * h_p ** 2 * d_tau_d_r/(
            2 * np.abs(tau - tau_0)**(1/2))

        c_R = np.sign(R - 1) / np.sqrt(
            1 + h_p**(-2) * R**2 * ((R**(-3/2) - 1 + correction)**2))

    # Split up the equation
    sec_2 = (1 + m_p_by_m_th * b_R * delta_chi/(2**(1/4) * h_p**(1/2)))**(-5/2)

    # Note np.gradient allows us to not use ghost points which is nice
    sec_3 = c_R * np.gradient(b_R * delta_chi, R[1] - R[0])

    scaling = c_s * m_p_by_m_th**3 / (2**(7/4) * h_p**(3/2))
    return scaling * (1/sigma_0) * b_R**2 * delta_chi**2 * sec_2 * sec_3


def reconstruct_surface_density(R, p, h_p, zeta,
                                c=2/3, max_relative_error=1e-7,
                                max_iter=500000, guess=None):
    """Reconstruct the surface density of a disk perturbed by a planet
    using vortensity

    R - Radial positions in units of R_p MUST BE LINEARLY SPACED
    p - power of unperturbed surface density
    h_p - width of the disk at the planet
    m_p - planetary mass either in thermal or stellar mass
    """

    # Define the sound speed next to the planet by the relationship
    # m_th = c_s**3/(omega * G)u
    c_s = h_p
    # Simplify due to choice of unit
    w = R[1] - R[0]

    # iteration count
    i = 0

    def compute_f(zeta_r, with_ghost, r_2):
        sig_i = with_ghost[1:-1]
        sig_diff = with_ghost[2:] - with_ghost[:-2]

        if (sig_i < 0).any():
            print(sig_i)
            print("Error sigma < 0")
            raise Exception

        a = (2 * zeta_r * sig_i**2/c_s**2) * (
                    r_2[1:-1]**(-3) + c_s**2 * sig_diff/(
                        2 * w * r_2[1:-1] * sig_i))**(1/2)

        return (a - (r_2[1:-1]**(-3) * sig_i/c_s**2)
                + (sig_diff**2/(4 * w**2 * sig_i))
                - (3 * sig_diff/(2 * w * r_2[1:-1])))

    # Background (unperturbed state)
    with_ghost_r = np.concatenate(
        (np.array([R[0] - w]), R, np.array([R[-1] + w])))

    with_ghost = with_ghost_r**(-p)

    if not (guess is None):
        with_ghost[1:-1] = guess

    f = compute_f(zeta, with_ghost, with_ghost_r)

    new_abs_error = np.abs(
        (with_ghost[:-2] + 2*with_ghost[1:-1] + with_ghost[2:])/w**2
        - f)

    previous_error = np.sum(new_abs_error)
    new_relative_error = np.max(
        np.abs(new_abs_error/(with_ghost_r**(-p))[1:-1]))
    total_error = previous_error
    
    while i < max_iter:
        if new_relative_error < max_relative_error:
            # print("Convergence found")
            break

        if total_error > previous_error:
            print(new_relative_error)
            print("Convergence not increasing")
            break

        # Update

        a = (1 - c) * with_ghost[1:-1] + (c/2) * (
            (-1 * w**2 * f) + with_ghost[2:] + with_ghost[:-2])

        with_ghost[1:-1] = a.copy()

        # Check convergence using a relative error scoring
        f = compute_f(zeta, with_ghost, with_ghost_r)

        new_abs_error = np.abs(
            (with_ghost[2:] - 2 * with_ghost[1:-1] + with_ghost[:-2])
            - f * w**2)
        total_error = np.sum(new_abs_error)/with_ghost[1:-1].shape[0]

        # Compute relative error compared to the initial guess
        new_relative_error = np.max(
            np.abs(new_abs_error/(with_ghost_r**(-p))[1:-1]))

        previous_error = total_error
        i = i + 1

    else:
        print("Max iterations exceeded")

    # print("Iterations: {}".format(i))
    # print("Final error: {}, {}".format(total_error, new_relative_error))
    return with_ghost[1:-1], total_error


def construct_surface_density(R, p, h_p, m_p, orbits, use_cr_21=True,
                                           c=2/3, max_error=1e-7,
                                           max_iter=500000):
    """ Construct surface density for a planet that has been in existence for a given 
    number of orbits. Helpful wrapper of reconstruct_surface_density"""
    
    zeta = 0
    if m_p == 0:
        zeta = background_vortensity(R, p, h_p)
    else:
        del_zeta = compute_delta_zeta(R, p, h_p, m_p, use_cr_21=use_cr_21)
        del_zeta_del_t = del_zeta * np.abs(R**(-3/2) - 1)/(2 * np.pi)
        # One factor of 2 pi is for time, and the second factor is to adjust for 
        # the incorrect scaling in the original paper
        zeta = background_vortensity(R, p, h_p) + del_zeta_del_t  * orbits * 2 * np.pi
        if use_cr_21:
            zeta = background_vortensity(R, p, h_p) + del_zeta_del_t  * orbits * 4 * np.pi**2

    return reconstruct_surface_density(
        R, p, h_p, zeta, c, max_error,
        max_iter)



def background_vortensity(R, p, h_p):
    """ Calculate the vortensity of an isothermal disk """

    if p == 0:
        return 0.5 * R**(-3/2)
    omega = (R**(-3) - h_p**2 * p/R**2)**(1/2)

    # NOTE THIS HAS BEEN FIXED BUT NOT WRITTEN UP
    # The fix that is

    zeta = R**(p) * (1 - 2 * h_p**2 * p * R) * (R - h_p**2 * R**2)**(-1/2) / (2 * R)
    return zeta
