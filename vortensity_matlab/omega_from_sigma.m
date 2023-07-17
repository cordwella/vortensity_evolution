function omega = omega_from_sigma(r, sigma, dsigma, h_p)
    omega = (r.^(-3) + h_p^2 * dsigma ./ (r .* sigma)).^(1/2);
end