function chi = compute_delta_chi(tau, m_p)
    % define constants
    tau_0 = 1.89 * m_p;
    tau_squiggle = abs(tau) - tau_0;

    % Compute the inner fit

    A = 2.07;
    tau_b = 0.300;
    alpha_1 = -10.84;
    alpha_2 = 0.505;
    delta = 0.623;

    scaled = tau_squiggle./tau_b;
    inner = (A * (scaled).^(-1 * alpha_1) ...
             .* ( ...
             (1 + (scaled).^(1/delta)).^((alpha_1 - alpha_2)*delta)));

    % Compute the outer fit
    A = 3.11;
    tau_b = 0.181;
    alpha_1 = -8.63;
    alpha_2 = 0.525;
    delta = 0.766;

    scaled = tau_squiggle./tau_b;
    outer = (A * (scaled).^(-1 * alpha_1) .* ((1 + (scaled).^(1/delta) ).^((alpha_1 - alpha_2)*delta)));

    chi = ((inner .* (tau <= 0)) + (outer .* (tau >= 0))) .* (abs(tau) >= tau_0);
end