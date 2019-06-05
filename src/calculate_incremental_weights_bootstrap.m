function lv = calculate_incremental_weights_bootstrap(model, y, xp, x, theta)
% Calculate incremental weights for bootstrap proposal

    narginchk(5, 5);
    [Nx, J] = size(x);
    py = model.py;
    if py.fast
        lv = py.logpdf(y*ones(1, J), xp, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = py.logpdf(y, xp(:, j), theta);
        end
    end
end
