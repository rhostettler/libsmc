function [A, b, Omega] = slr_cf(mp, Pp, theta, Ey, Cy, Cyx)
% # Closed-form statistical linear regression
% ## Usage
% * `[A, b, Omega] = slr_sp(mp, Pp, theta, g, R)`
% * `[A, b, Omega] = slr_sp(mp, Pp, theta, g, R, Xi, wm, wc)`
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
% Uses closed-form expression for the mean and covariances.
%
% ## Input
% * `mp`: Linearization density mean.
% * `Pp`: Linearization density covariance.
% * `theta`: dtheta-times-1 vector of other parameters.
% * `Ey`: Closed-form expression of the mean integral (function handle 
%    `@(m, P, theta)`).
% * `Cy`: Closed-form expression of the covariance integral (function 
%   handle `@(m, P, theta)`).
% * `Cyx`: Closed-form expression of the cross-covariance integral 
%   (function handle `@(m, P, theta)`).
%
% ## Ouptut
% * `A`: Slope of the affine approximation.
% * `b`: Intercept of the affine approximation.
% * `Omega`: Residual noise covariance.
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
% * Ey, Cy, Cyx should probably be taken from the 'model' struct in the
% future.

    %% Defaults
    narginchk(6, 6);

    %% Linearization
    A = Cyx(mp, Pp, theta)/Pp;
    b = Ey(mp, Pp, theta) - A*mp;
    Omega = Cy(mp, Pp, theta) - A*Pp*A';
end
