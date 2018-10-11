function xp = sample_bootstrap(y, x, t, model)
% Sample using the bootstrap importance density
%
% USAGE
%   xp = SAMPLE_BOOTSTRAP(y, x, t, model)
%
% DESCRIPTION
%   Samples a set of new samples x[n] from the bootstrap importance 
%   distribution, that is, samples
%
%       x[n] ~ p(x[n] | x[n-1]).
%
% PARAMETERS
%   y       Measurement vector y[n].
%   x       Samples at x[n-1].
%   t       Time t[n] (or time difference t[n]-t[n-1]).
%   model   State-space model structure.
%
% RETURNS
%   xp      The new samples x[n]
%
% AUTHOR
%   2018-10-11 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(4, 4);
    [Nx, M] = size(x);
    px = model.px;
    if px.fast
        xp = px.rand(x, t);
    else
        xp = zeros(Nx, M);
        for m = 1:M
            xp(:, m) = px.rand(x(:, m), t);
        end
    end
end
