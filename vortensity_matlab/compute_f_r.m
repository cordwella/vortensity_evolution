function k_r = compute_f_r(R, m_p, h_p, p)
    
    tau = r_to_tau(R, m_p, h_p, p);
    delta_chi = compute_delta_chi(tau, m_p);

    epsilon = delta_chi .* m_p .* (...
        abs(R.^(-3/2) - 1) ./...
        (2^(1/2) * h_p * R.^(1 - p)) ).^(1/2);
    
    psi_i = (epsilon .* (2 + epsilon) - (2 * (1 + epsilon)) .* log(1 + epsilon))./(2 * (1 + epsilon));
    % plot(R, psi);
    k_r = sign(1 - R.^(-3/2)) .* R .* (h_p^2) .* psi_i * 2 * pi;
end