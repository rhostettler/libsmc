function [xp, lv] = sis_update_generic(y, x, theta, model, q)
% Sample from an arbitrary importance density
%
% USAGE
%   [xp, lv] = SIS_UPDATE_GENERIC(y, x, theta, model, q)
%
% DESCRIPTION
%   Samples a set of new samples x[n] from the importance distribution 
%   q(x[n]), that is, samples x[n] ~ q(x[n]).
%
% PARAMETERS
%   y       Measurement vector y[n].
%   x       Samples at x[n-1].
%   theta   Parameter (e.g., time or time difference) or other input.
%   model   State-space model structure.
%   q       Proposal density such that x[n] ~ q(x[n] | x[n-1], y[n]).
%
% RETURNS
%   xp      The new samples x[n].
%   lv      The log of the incremental weights.
%
% SEE ALSO
%   sis_update_bootstrap
%
% AUTHOR
%   2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(5, 5);
    [Nx, J] = size(x);
    
    %% Sampling
    if q.fast
        xp = q.rand(y*ones(1, J), x, theta);
    else
        xp = zeros(Nx, J);
        for j = 1:J
            xp(:, j) = q.rand(y, x(:, j), theta);
        end
    end

    %% Incremental weights
    px = model.px;
    py = model.py;    
    if px.fast && py.fast && q.fast
        lv = ( ...
            py.logpdf(y*ones(1, J), xp, theta) ...
            + px.logpdf(xp, x, theta) ...
            - q.logpdf(xp, y*ones(1, J), x, theta) ...
        );
    else
        J = size(xp, 2);
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = ( ...
                py.logpdf(y, xp(:, j), theta) ...
                + px.logpdf(xp(:, j), x(:, j), theta) ...
                - q.logpdf(xp(:, j), y, x(:, j), theta) ...
            );
        end
    end    
end
