function [xhat, w] = gridf(model, y, theta, x)
% # Grid filter for 1D (nonlinear) state-space models
% ## Usage
% * `xhat = gridf(model, y)`
% * `[xhat, w] = gridf(model, y, theta, x)
% 
% ## Description
% Implements Bayesian filtering (prediction/update) using a set of
% deterministic grid points and numerical integration.
%
% Note: Only scalar (1D) models are supported.
% 
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N measurement data matrix.
% * `theta`: Additional model parameters (default: `NaN`)
% * `x`: 1-times-J vector of (equispaced)  gridpoints (default: 
%   `-1:1e-3:1`).
% 
% ## Output
% * `xhat`: Minimum mean squared error filter state estimate.
% * `w`: J-times-N matrix of numerically estimated posterior pdfs.
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

% TODO: 
% * Prediction and update function assume that px.fast and py.fast are set

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(x)
        warning('Using default grid [-1,1], results may be inaccurate');
        x = -1:1e-3:1;
    end
    
    % Sanity check
    [dx, J] = size(x);
    if dx > 1
        error('gridfilter does not work for state dimensions larger than 1');
    end
    if std(diff(x)) > 1e-9
        error('Grid must be equispaced.');
    end
    if ~model.px.fast || ~model.py.fast
        error('px and py must be fast-evaluatable.');
    end
    
    %% Expand inputs
    [dy, N] = size(y);
    if size(theta, 2) == 1
        theta = theta*ones(1, N);
    end
    y = [NaN*ones(dy, 1), y];
    theta = [NaN*ones(size(theta, 1)), theta];
    N = N+1;
    
    %% Preallocate
    w = zeros(N, J);
    xhat = zeros(dx, N-1);
    deltax = x(2)-x(1);
    
    %% Initialize
    lw = model.px0.logpdf(x, theta);
    
    for n = 2:N
        % Prediction
        lwp = grid_predict(model, x, lw, theta(n));
    
        % Measurement update
        lw = grid_update(model, y(:, n), x, lwp, theta(n));
        
        %% Store
        w(n, :) = exp(lw);
        xhat(:, n-1) = deltax*(w(n, :)*x.');
    end
    
    % Remove leading entry
    w = w(2:N, :);
end

%% Prediction
function lwp = grid_predict(model, x, lw, theta)
    J = size(x, 2);
    deltax = x(2)-x(1);
    lpx = zeros(J, J);

    % lpx is a matrix where each column is for one x[n] grid point.
    % Hence, summing over rows will yield the marginalization
    for j = 1:J
        lpx(:, j) = model.px.logpdf(x(j)*ones(1, J), x, theta).';
    end
    wp = deltax*sum(exp(lpx+lw.'*ones(1, J)));
    lwp = log(wp);
end

%% Measurement update
function lw = grid_update(model, y, x, lwp, theta)
    J = size(x, 2);
    deltax = x(2)-x(1);
    lpy = model.py.logpdf(y*ones(1, J), x, theta);
    lw = lpy + lwp - log(deltax*sum(exp(lpy + lwp)));
end
