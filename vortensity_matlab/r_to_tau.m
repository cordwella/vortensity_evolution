function tau = r_to_tau(R, m_p, h_p, p)
    base = sign(R - 1) .* (3/(2^(5/4))) * m_p * h_p^(-5/2);
    
    % Two versions of the integral 
    % computed directly as hypergeometric functions for ease
    % as hyper geometric functions for 
    % each one
    % assume p = 3/2

    fun = @(s) abs(s.^(3/2) - 1).^(3/2) .* s.^((p/2) - (11/4));
    function i = eval(x)
        i = abs(integral(fun, 1, x, 'RelTol', 1e-12));
    end
    
    tau = base .* abs(arrayfun(@(x) eval(x), R));
end