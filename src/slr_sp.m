function [A, b, Omega] = slr_sp(mp, Pp, theta, g, R, Xi, wm, wc)
% # Sigma-point statistical linear regression
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
% Approximation of the moment matching integrals using sigma-points.
%
% ## Input
% * `mp`: Linearization density mean.
% * `Pp`: Linearization density covariance.
% * `theta`: dtheta-times-1 vector of other parameters.
% * `g`: Mean of the likelihood (function handle @(m, P, theta)).
% * `R`: Covariance of the likelihood (function hanlde @(m, P, theta)).
% * `Xi`: Unit sigma-points (default: cubature rule).
% * `wm`: Mean sigma-point weights (default: cubature rule).
% * `wc`: Covariance sigma-point weights (default: cubature rule).
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
% * When Xi is specified, at least wm should be too (and vice-versa)
% * Defaults for sigma-points are missing
% * g, R should be taken from the model struct in the future

    %% Defaults
    narginchk(5, 8);
    if nargin < 6 || isempty(Xi)
        error('Please specify the sigma-points.');
    end
    if nargin < 7 || isempty(wm)
        error('Please specify the sigma-point weights.');
    end
    if nargin < 8 || isempty(wc)
        warning('libsmc:warning', 'Using the same weights for mean and covariance');
        wc = wm;
    end

    %% Linearization
    dx = size(mp, 1);
    dy = size(g(mp, theta), 1);
    I = length(wm);
    Y = zeros(dy, I);
    
    % Generate sigma-points
    Lp = chol(Pp, 'lower');
    X = mp*ones(1, I) + Lp*Xi;

    % Calculate expectations w.r.t. linearziation density
    Ey = zeros(dy, 1);              % E{y}
    Ey2 = zeros(dy, dy);            % E{y*y'}
    EVy_x = zeros(dy, dy);          % E{V{y|x}}
    Eyx = zeros(dy, dx);            % E{y*x'}
    for i = 1:I
        Y(:, i) = g(X(:, i), theta);
        Ey = Ey + wm(i)*Y(:, i);
        Ey2 = Ey2 + wc(i)*(Y(:, i)*Y(:, i)');            
        EVy_x = EVy_x + wc(i)*R(X(:, i), theta);
        Eyx = Eyx + wc(i)*(Y(:, i)*X(:, i)');
    end

    % Calculate (co)variances w.r.t. linearization density
    Vy = Ey2 - (Ey*Ey') + EVy_x;    % V{y}
    Vy = (Vy + Vy')/2;
    Cyx = Eyx - (Ey*mp');           % C{y,x}

    % Calculate linearization w.r.t. linearization density
    % y = A*x + b + nu, nu ~ N(0, Omega)
    A = Cyx/Pp;
    b = Ey - A*mp;
    Omega = Vy - A*Pp*A';
end
