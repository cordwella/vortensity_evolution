function [sigma, err] = construct_surface_density(R, p, h_p, m_p, orbits)
    c_s = h_p;
    % Simplify due to choice of unit
    w = R(2) - R(1);

    % iteration count
    i = 0;
    
    % NB: Because of how the finite differencing works if the size 
    % of R is too low you will also need to reduce the max relative error
    c = 2/3; max_relative_error = 1e-7; max_iter = 500000;
    % For this you want about 4000 points 
    
    function f = compute_f(zeta_r, with_ghost, r_2)
        sig_i = with_ghost(2:end - 1);
        sig_diff = with_ghost(3:end) - with_ghost(1:end - 2);

        if any(sig_i < 0)
            disp(sig_i)
            error("Error sigma < 0")
        end

        a = (2 .* zeta_r .* sig_i.^2./c_s^2) .* ( ...
             r_2(2:end - 1).^(-3) + c_s^2 .* sig_diff./( ...
                        2 * w .* r_2(2:end - 1) .* sig_i)).^(1/2);

        f = (a - (r_2(2:end - 1).^(-3) .* sig_i./c_s.^2) ...
                + (sig_diff.^2./(4 * w.^2 .* sig_i)) ...
                - (3 .* sig_diff./(2 * w .* r_2(2:end - 1))));
    end
    
    with_ghost_r = [R(1) - w, R, R(end) + w];
    with_ghost = with_ghost_r.^(-p);

    zeta = 0;
    if m_p == 0
        zeta = background_vortensity(R, p, h_p); % + del_zeta;
    else
        del_zeta_del_t = compute_delta_zeta(R, p, h_p, m_p) .* abs(R.^(-3/2) - 1)/(2 * pi);
        zeta = background_vortensity(R, p, h_p) + del_zeta_del_t * orbits * 2 * pi ;
    end

    f = compute_f(zeta, with_ghost, with_ghost_r);

    new_abs_error = abs( ...
        (with_ghost(1:end - 2) - 2 .* with_ghost(2:end - 1) ...
        + with_ghost(3:end)) - f .* w.^2);

    previous_error = sum(new_abs_error);
    new_relative_error = max(abs(new_abs_error .* R.^(p)));
    total_error = previous_error * 0.95;

    disp(new_relative_error)
    disp(i)

    while i < max_iter
        if new_relative_error < max_relative_error
            disp("Convergence found")
            break
        end

        if total_error > 1.1 * previous_error
            disp("Convergence not increasing")
            disp(i)
            break
        end

        % Update
        % with_ghost(3:end) + with_ghost(1:end - 2);
        holder = (1 - c) .* with_ghost(2:end - 1) + (c/2) * ( ...
            (-1 * w^2 .* f) + with_ghost(3:end) + with_ghost(1:end - 2));

        with_ghost(2:end - 1) = holder;

        % Check convergence using a relative error scoring
        f = compute_f(zeta, with_ghost, with_ghost_r);

        previous_error = total_error;

        new_abs_error = abs( ...
            (with_ghost(1:end - 2) - 2 .* with_ghost(2:end - 1) ...
            + with_ghost(3:end)) - f .* w.^2);

        total_error = sum(new_abs_error);
        new_relative_error = max(abs(new_abs_error .* R.^(p)));

        i = i + 1;
    end

    if new_relative_error > max_relative_error
        disp("Solution incomplete")
    end
    disp(i)

    sigma = with_ghost(2:end - 1);
    err = new_relative_error;
end