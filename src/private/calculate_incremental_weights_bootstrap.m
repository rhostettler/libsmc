function lv = calculate_incremental_weights_bootstrap(y, x, t, model)
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
%   y       Measurement y.
%   x       Particles x[n].
%   t       Timestamp.
%   model   Model structure.
%
% RETURNS
%   lv      The non-normalized log-weights.
%
% AUTHOR
%   2017-04-07 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(4, 4);
    M = size(x, 2);
    py = model.py;
    if py.fast
        lv = py.logpdf(y*ones(1, M), x, t);
    else
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = py.logpdf(y, x(:, m), t);
        end
    end
end
