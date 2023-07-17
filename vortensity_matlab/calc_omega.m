function om = calc_omega(r, sigma, h_p)
    ds = gradient(sigma.y, sigma.x);
    om = (r.^(-3) + h_p .* ds ./ (r .* sigma))^(1/2);
end