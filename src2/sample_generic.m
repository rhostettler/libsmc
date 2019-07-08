function xp = sample_generic(y, x, t, model, q)
% Sample from an arbitrary importance density
%
% USAGE
%   xp = SAMPLE_GENERIC(y, x, t, q)
%
% DESCRIPTION
%   Samples a set of new samples x[n] from the importance distribution 
%   q(x[n]), that is, samples x[n] ~ q(x[n]).
%
% PARAMETERS
%   y       Measurement vector y[n].
%   x       Samples at x[n-1].
%   t       Time t[n] (or time difference t[n]-t[n-1]).
%   model   State-space model structure.
%   q       Proposal density such that x[n] ~ q(x[n] | x[n-1], y[n]).
%
% RETURNS
%   xp      The new samples x[n]
%
% SEE ALSO
%   calculate_incremental_weights_generic
%
% AUTHOR
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

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
