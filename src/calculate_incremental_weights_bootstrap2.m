function [v, lv] = calculate_incremental_weights_bootstrap2(y, x, t, model)
% Calculate incremental weights for the bootstrap PF and non-Markovian SSMs
%
% SYNOPSIS
%   [v, lv] = calculate_incremental_weights_bootstrap2(y, x, t, model)
%
% DESCRIPTION
%   Calculates the weight increment for non-Markovian state-space models
%   when using the corresponding bootstrap particle filter (bootstrap_pf2).
%   In particular, it calculates the value of the likelihood
%
%       v = p(y[n] | x[1:n], y[1:n-1])
%
%   (or rather its log).
%
% PARAMETERS
%   y       Ny x n matrix of measurements
%
%   x       Ny x M x n+1 matrix of state trajectories (including x[0])
%
%   t       1 x n+1 vector of sampling times
%
%   model   State-space model
%
% RETURNS
%   v       The incremental weight (non-normalized)
%
%   lv      The log of the incremental weight (non-normalized)
%
% SEE ALSO
%   bootstrap_pf2
%
% VERSION
%   2017-08-11
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Move to general smc library

    % Strip initial values (x[0] and t[0])
    [~, M, n] = size(x);
    x = x(:, :, 2:n);
    t = t(2:n);

    py = model.py;
    if py.fast
        lv = py.logpdf(y, x, t);
    else
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = py.logpdf(y, x(:, m, :), t);
        end
    end
    v = exp(lv-max(lv));
end
