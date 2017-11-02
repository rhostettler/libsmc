function xp = sample_bootstrap2(x, t, model)
% Sample from the bootstrap proposal for non-Markovian SSMs
%
% SYNOPSIS
%   xp = sample_bootstrap2(x, t, model)
%
% DESCRIPTION
%   Samples new states from the non-Markovian transition density 
%   p(x[n] | x[0:n-1]).
%
% PARAMETERS
%   x       Nx x M x n matrix of the states x[0:n-1]
%   
%   t       1 x n vector of timestamps
%
%   model   State-space model
%
% RETURNS
%   xp      The samples
%
% SEE ALSO
%   bootstrap_pf2
%
% VERSION
%   2017-08-11
% 
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    [Nx, M] = size(x);
    px = model.px;
    if px.fast
        xp = px.rand(x, t);
    else
        xp = zeros(Nx, M);
        for m = 1:M
            xp(:, m) = px.rand(x(:, m, :), t);
        end
    end
end
