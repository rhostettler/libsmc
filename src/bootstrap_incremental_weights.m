function [v, lv] = bootstrap_incremental_weights(y, xp, t, model)
% Incremental particle weights for the bootstrap particle filter
%
% SYNOPSIS
%   [v, lv] = bootstrap_incremental_weights(y, xp, t, model)
%
% DESCRIPTION
%
% PARAMETERS
%
% RETURNS
%
% VERSION
%   2017-04-07
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

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
    v = exp(lv-max(lv));
end
