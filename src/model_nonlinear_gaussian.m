function model = model_nonlinear_gaussian(f, Q, g, R, m0, P0, Fx, Gx, fast)
% # Nonlinear Gaussian state-space Model
% ## Usage
% * `model = model_nonlinear_gaussian(f, Q, g, R, m0, P0)`
% * `model = model_nonlinear_gaussian(f, Q, g, R, m0, P0, Fx, Gx, fast)`
%
% ## Description
% Defines the model structure for nonlinear state-space models with
% Gaussian process- and measurement noise of the form
%
%     x[0] ~ N(m0, P0),
%     x[n] = f(x[n-1], n) + q[n],
%     y[n] = g(x[n], n) + r[n],
%
% where q[n] ~ N(0, Q[n]) and r[n] ~ N(0, R[n]).
%
% ## Input
% * `f`: Mean function of the dynamic model. Function handle of the form 
%   `@(x, theta)`.
% * `Q`: Process noise covariance, either a dx-times-dx matrix or a
%   function handle of the form `@(x, theta)`.
% * `g`: Mean function of the observation model. Function handle of the 
%   form `@(x, theta)`.
% * `R`: Measurement noise covariance, either a dy-times-dy matrix or a
%   function handle of the form `@(x, theta)`.
% * `Fx`: Jacobian of the dynamic model, function handle `@(x, theta)` 
%   (optional).
% * `Gx`: Jacobian of the measurement model, function handle `@(x, theta)`
%   (optional).
% * `fast`: Boolean flag set to 'true' if the transition density and
%   likelihood (i.e., `f`, `Q`, `g`, and `R`) can evaluate whole dx-times-J
%   particle matrices at once (default: false).
%
% ## Output
% * `model`: The state-space model struct that contains the necessary 
%   fields (i.e., the probabilistic representation of the state-space model
%   px0, px, py).
%
% ## Authors
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

% TODO:
% * The pdf of the initial state is not yet handled by `pdf_mvn`.

    %% Defaults
    narginchk(6, 9);
    if nargin < 7
        Fx = [];
    end
    if nargin < 8
        Gx = [];
    end
    if nargin < 9 || isempty(fast)
        fast = false;
    end
    
    %% Initialize model struct
    % Initial state
    dx = size(m0, 1);
    L0 = chol(P0).';
    px0 = struct();
    px0.rand = @(J) m0*ones(1, J)+L0*randn(dx, J);
    px0.logpdf = @(x, theta) logmvnpdf(x.', m0.', P0).';
    px0.mean = m0;
    px0.cov = P0;
    px0.fast = true;
    dy = size(g(m0, []), 1);
    
    % State transition densiy, likelihood
    px = pdf_mvn(dx, f, Q, Fx, fast);
    py = pdf_mvn(dy, g, R, Gx, fast);
    
    % Complete model
    model = struct('px0', px0, 'px', px, 'py', py);
end
