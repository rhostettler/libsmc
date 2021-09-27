function [xp, alpha, lqx, lqalpha, qstate] = sample_generic(~, y, x, lw, theta, q, par)
% # Sample from an arbitrary importance density
% ## Usage
% * `[xp, alpha, lqx, lqalpha] = sample_generic(model, y, x, lw, theta, q)`
% * `[xp, alpha, lqx, lqalpha, qstate] = sample_generic(model, y, x, lw, theta, q, par)`
%
% ## Description
% Samples a set of new particles x[n] from the importance distribution 
% q(x[n]).
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `x`: dx-times-J particle matrix at x[n-1].
% * `lw`: Log-weights of x[n-1].
% * `theta`: Additional parameter vector.
% * `q`: Importance density such that x[n] ~ q(x[n] | x[n-1], y[n]).
% * `par`: Struct of additional parameters:
%   - `resample`: Resampling function (default: `resample_ess`).
%
% ## Output
% * `xp`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lqx`: 1-times-J vector of the importance density of the jth sample 
%   `xp`.
% * `lqalpha`: 1-times-J vector of the importance density of the jth
%   ancestor index `alpha`.
% * `qstate`: Sampling algorithm state information.
%
% ## Authors
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

%{
% This file is part of the libsmc Matlab toolbox.
%
% libsmc is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libsmc is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libsmc. If not, see <http://www.gnu.org/licenses/>.
%}

    %% Defaults
    narginchk(6, 7);
    if nargin < 7 || isempty(par)
        par = struct();
    end
    defaults = struct( ...
        'resample', @resample_ess ...
    );
    par = parchk(par, defaults);

    %% Sampling   
    % Sample ancestor indices (resampling)
    [alpha, lqalpha, rstate] = par.resample(lw);
    x = x(:, alpha);

    % Sample new states
    [dx, J] = size(x);
    if q.fast
        xp = q.rand(y*ones(1, J), x, theta);
        lqx = q.logpdf(xp, y*ones(1, J), x, theta);
    else
        xp = zeros(dx, J);
        lqx = zeros(1, J);
        for j = 1:J
            xp(:, j) = q.rand(y, x(:, j), theta);
            lqx(j) = q.logpdf(xp(:, j), y, x(:, j), theta);
        end
    end
    
    qstate = struct('rstate', rstate, 'qj', q);
end
