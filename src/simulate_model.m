function [xs, ys] = simulate_model(model, theta, N)
% # Simulate probabilistic state-space model
% ## Usage
% * `[x, y] = simulate_model(model)`
% * `[x, y] = simulate_model(model, theta, N)`
%
% ## Description
% Simulates a probabilistic state-space model, that is, simulates a state
% trajectory of length `N` and the corresponding measurements. Requires
% that both the transition density and likelihood can be sampled from.
%
% ## Input
% * `model`: State-space model struct.
% * `theta`: Additional parameters to the model (default: `NaN`).
% * `N`: Number of time samples to generate (default: `100`).
%
% ## Output
% * `x`: dx-times-N matrix of simulated states.
% * `y`: dy-times-N matrix of simulated measurements.
%
% ## Authors
% 2019-present -- Roland Hostettler

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
    narginchk(1, 3);
    if nargin < 2 || isempty(theta)
        theta = NaN;
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    
    % Expand
    if size(theta, 2) == 1
        theta = theta*ones(1, N);
    end

    %% Generate data
    % Initialize
    x = model.px0.rand(1);
    y = model.py.rand(x, theta(:, 1)); % Generate a dummy measurement to peek at the dimension dy

    % Preallocate
    dx = size(x, 1);
    dy = size(y, 1);
    ys = zeros(dy, N);
    xs = zeros(dx, N);
    
    % Generate data
    for n = 1:N
        x = model.px.rand(x, theta(:, n));
        y = model.py.rand(x, theta(:, n));
        xs(:, n) = x;
        ys(:, n) = y;
    end
end
