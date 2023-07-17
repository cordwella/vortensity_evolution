function zeta = background_vortensity(R, p, h_p)

    if p == 0
        zeta = 0.5 * R.^(-3/2);
    else
        zeta = R.^(p) .* (1 - 2 * h_p^2 * p .* R) .* (R - h_p^2 * R.^2).^(-1/2) ./ (2 .* R);
    end
end