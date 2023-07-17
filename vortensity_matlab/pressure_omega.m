function om = pressure_omega(r, p, h_p)
    om = (r.^(-3) - h_p .* p .* r.^(-2)).^(1/2);
end