function xp = draw_samples(y, x, t, q)
% Sample from the SMC importance density
%
% SYNOPSIS
%   xp = draw_samples(y, x, t, q)
%
% DESCRIPTION
%
%
% PARAMETERS
%   y       Measurement vector y[n]
%
%   x       Samples at x[n-1]
%
%   t       Time t[n] (or time difference t[n]-t[n-1])
%
%   q       Proposal density such that x[n] ~ q(x[n] | x[n-1], y[n])
%
% RETURNS
%   xp      The new samples x[n]
%
% VERSION
%   2017-06-18
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@ltu.se>

% TODO:
%   * Description
%   * Add some comments below
%   * How about non-markovian systems?

    narginchk(4, 4);

    [Nx, M] = size(x);
    if q.fast
        xp = q.rand(y*ones(1, M), x, t);
    else
        xp = zeros(Nx, M);
        for m = 1:M
            xp(:, m) = q.rand(y, x(:, m), t);
        end
    end
end
