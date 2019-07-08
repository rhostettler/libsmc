function [xp, lv, q] = sis_update_bootstrap(y, x, theta, model)
% Sequential Imporance Sampling: Bootstrap Update
%
% USAGE
%   [xp, lv] = SIS_UPDATE_BOOTSTRAP(y, xp, x, theta, model)
%
% DESCRIPTION
%   Samples a set of new samples x[n] from the bootstrap importance 
%   distribution, that is, samples
%
%       x[n] ~ p(x[n] | x[n-1])
%
%   and calculates the incremental weights
%
%       lv = log(p(y[n] | x[n]))
%
%   for state-space models of the form
%
%       x[0] ~ p(x[0]),
%       x[n] ~ p(x[n] | x[n-1]),
%       y[n] ~ p(y[n] | x[n]).
%
% PARAMETERS
%   y       Measurement y[n].
%   xp      Particles at time t[n] (i.e. x[n]).
%   x       Particles at time t[n-1] (i.e. x[n-1]).
%   theta   Additional parameters.
%   model   Model structure.
%
% RETURNS
%   xp      The new samples x[n].
%   lv      The non-normalized log-weights.
%
% AUTHOR
%   2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(4, 4);
    [Nx, J] = size(x);
    q = [];

    %% Sample
    px = model.px;
    if px.fast
        xp = px.rand(x, theta);
    else
        xp = zeros(Nx, J);
        for j = 1:J
            xp(:, j) = px.rand(x(:, j), theta);
        end
    end

    %% Calculate incremental weights
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
