function [A, b, Omega] = slr_taylor(m, P, theta, g, Gx, R)
% # Taylor series expansion statistical linear regression
% ## Usage
% * `[A, b, Omega] = slr_taylor(m, P, theta, g, Gx, R)`
%
% ## Description
% Statistical linear regression of the (nonlinear, non-Gaussian) likelihood
% p(y[n] |x[n]) with mean E{y[n] | x[n]} and covariance Cov{y[n] | x[n]},
% that is, approximation of the measurement model according to
%
%     y = A*x + b + nu
%
% with `nu ~ N(0, Omega)`.
%
% This implementation uses Taylor-series-based linearization of the
% nonlinear model (as the EKF).
%
% ## Input
% * `m`: Linearization density mean.
% * `P`: Linearization density covariance.
% * `theta`: dtheta-times-1 vector of other parameters.
% * `g`: Mean of the likelihood E{y[n] | x[n]} (function handle 
%   `@(x, theta)`).
% * `Gx`: Jacobian of the mean of the likelihood (function handle 
%   `@(x, theta)`).
% * `R`: Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%   `@(x, theta)`).
%
% ## Ouptut
% * `A`: Slope of the affine approximation.
% * `b`: Intercept of the affine approximation.
% * `Omega`: Residual noise covariance.
%
% ## Authors
% 2018-present -- Roland Hostettler

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
% * Take g, Gx, R from model struct
% * The equations for A, b, and Omega really reduce to G, 0, and R

    %% Defaults
    narginchk(6, 6);
    
    %% Linearization
    % Expectations w.r.t. linearization density
    my = g(m, theta);
    G = Gx(m, theta);
    if isa(R, 'function_handle')
        Py = G*P*G' + R(m, theta);
    else
        Py = G*P*G' + R;
    end
    Pyx = G*P;

    % Calculate linearization w.r.t. linearization density
    % y = A*x + b + v, v ~ N(0, Omega)
    A = Pyx/P;
    b = my - A*m;
    Omega = Py - A*P*A';
end
