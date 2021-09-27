function [xp, alpha, lqx, lqalpha, qstate] = sample_bootstrap(model, ~, x, lw, theta, par)
% # Sample from the bootstrap importance density
% ## Usage
% * `[xp, alpha, lqx, lqalpha, qstate] = sample_bootstrap(model, y, x, lw, theta)`
% * `[xp, alpha, lqx, lqalpha, qstate] = sample_bootstrap(model, y, x, lw, theta, par)`
%
% ## Description
% Samples a set of new samples x[n] from the bootstrap importance density,
% that is, samples
%
%     x[n] ~ p(x[n] | x[n-1]).
%
% ## Input
% * `model`: State-space model struct.
% * `y`: Measurement vector y[n].
% * `x`: Samples at x[n-1].
% * `lw`: Log-weights of x[n-1].
% * `theta`: Model parameters.
% * `par`: Struct of additional parameters:
%   - `resample`: Resampling function (default: `resample_ess`).
%
% ## Output
% * `xp`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lq`: 1-times-J vector of the importance density of the jth sample 
%   `xp`.
% * `lqalpha`: 1-times-J vector of the importance density of the jth
%   ancestor index `alpha`.
% * `qstate`: Sampling algorithm state information, see `resample_ess`.
%
% ## Author
% 2018-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
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
    
    % Sample
    [dx, J] = size(x);
    lqx = zeros(1, J);
    px = model.px;
    if px.fast
        xp = px.rand(x, theta);
        lqx = px.logpdf(xp, x, theta);
    else
        xp = zeros(dx, J);
        for j = 1:J
            xp(:, j) = px.rand(x(:, j), theta);
            lqx(j) = px.logpdf(xp(:, j), x(:, j), theta);
        end
    end
    
    qstate = struct('rstate', rstate, 'qj', repmat(model.px, [1, J]));
end
