function zeta = compute_delta_zeta(R, p, h_p, m_p)
    % Basic properties of the system
    % using Omega_k(R_p) = Sigma_0(R_p) = G = M_star = 1
    % Radial position of the planet is one by this definition (given kepler)

    % Compute thermal mass at the planet
    m_th = h_p^3;  % M_th = (H_p/R_p)**3 M_star

    % Define the sound speed next to the planet by the relationship
    % m_th = c_s**3/(omega * G)
    % c_s = m_th^(1/3);
    % m_p_by_m_th = 0
    c_s = h_p;

    % Rescale to be in stellar masses
    % which is easy as we are dealing with units of M_stellar = 1
    % Mass should be given in thermal mass
    m_p_by_m_th = m_p;
    % m_p = m_p * m_th;

    % Unperturbed angular velocity of disk fluid
    % taking to be keplerian
    % omega_0 = np.sqrt(omega_k_squared(R))  # - p * c_s**2)
    % Unperturbed surface density
    sigma_0 = R.^(-p);  % Sigma_p * (R/R_p) ** (-p)

    % Compute scaling factor tau
    tau = r_to_tau(R, m_p_by_m_th, h_p, p);

    % Comptute shock strength
    tau_0 = 1.89 * m_p_by_m_th;  % equation 16
    delta_chi = compute_delta_chi(tau, m_p_by_m_th);

    % Combine to get the vortensity jump
    % Compute B(R) and C(R)
    % CR21 C4
    b_R = sqrt(R.^(p-1) .* abs(R.^(-3/2) - 1));

    tau = abs(tau);
    delta_phi = 1;
    d_tau_d_r = gradient(tau)./(R(2) - R(1)); 
    correction = delta_phi * h_p^ 2 * d_tau_d_r./( ...
        2 * abs(tau - tau_0).^(1/2));

    c_R = sign(R - 1) ./ sqrt( ...
        1 + h_p^(-2) .* R.^2 .* ((R.^(-3/2) - 1 + correction).^2));

    % Split up the equation
    sec_2 = (1 + m_p_by_m_th .* b_R .* delta_chi./(2^(1/4) * h_p^(1/2))).^(-5/2);

    % Note np.gradient allows us to not use ghost points which is nice
    sec_3 = c_R .* gradient(b_R .* delta_chi) ./ gradient(R);

    scaling = c_s * m_p_by_m_th^3 / (2^(7/4) * h_p^(3/2)) * 2 * pi;
    zeta = scaling .* (1./sigma_0) .* b_R.^2 .* delta_chi.^2 .* sec_2 .* sec_3;
end
