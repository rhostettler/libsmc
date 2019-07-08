function lv = calculate_incremental_weights_bootstrap(y, xp, x, t, model)
% Incremental particle weights for the bootstrap particle filter
%
% SYNOPSIS
%   [v, lv] = CALCULATE_INCREMENTAL_WEIGHTS_BOOTSTRAP(y, xp, t, model)
%
% DESCRIPTION
%   Calculates the incremental particle weights for the bootstrap particle
%   filter. In this case, the incremental weight is given by
%
%       v[n] ~= p(y[n] | x[n]).
%
%   Note that the function actually computes the non-normalized log weights
%   for numerical stability.
%
% PARAMETERS
%   y       Measurement y[n].
%   xp      Particles at time t[n] (i.e. x[n]).
%   x       Particles at time t[n-1] (i.e. x[n-1]).
%   t       Time t[n] or time difference t[n]-t[n-1].
%   model   Model structure.
%
% RETURNS
%   lv      The non-normalized log-weights.
%
% AUTHOR
%   2017-04-07 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(5, 5);
    M = size(xp, 2);
    py = model.py;
    if py.fast
        lv = py.logpdf(y*ones(1, M), xp, t);
    else
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = py.logpdf(y, xp(:, m), t);
        end
    end
end
