function lv = calculate_weights(model, y, xp, alpha, lq, x, lw, theta)
% # General weight calculation for sequential importance sampling
% ## Usage
% * `lv = calculate_weights(model, y, xp, alpha, lq, x, lw, theta)`
%
% ## Description
% Calculates the non-normalized log-weight for sequential importance
% sampling for an arbitrary proposal density q, that is,
%            _                                                _
%           | p(y[n]|x[n]) p(x[n]|x[n-1]) p(x[1:n-1]|y[1:n-1]) |
%   lv = log| ------------------------------------------------ |
%           |_                   q(x[n])                      _|
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `xp`: dx-times-J matrix of newly drawn particles for the state x[n].
% * `alpha`: 1-times-J vector of ancestor indices for the state x[n].
% * `lq`: 1-times-J vector of importance density evaluations at 
%   {`xp(:, j)`, `alpha(j)`}.
% * `x`: dx-times-J matrix of previous state particles x[n-1].
% * `lw`: 1-times-J matrix of trajectory weights up to n-1.
% * `theta`: Additional parameters.
%
% ## Output
% * `lv`: 1-times-J vector of logarithm of updated weights.
%
% ## Authors
% 2021-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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

    narginchk(8, 8);
    
    % Get ancestor particles and trajectory weight
    x = x(:, alpha);
    lw = lw(alpha);
        
    % Evaluate transition density and likelihood
    J = size(xp, 2);
    lpx = zeros(1, J);
    lpy = lpx;
    if model.px.fast && model.py.fast
        lpx = model.px.logpdf(xp, x, theta);
        lpy = model.py.logpdf(y*ones(1, J), xp, theta);
    else
        for j = 1:J
            lpx(j) = model.px.logpdf(xp(:, j), x(:, j), theta);
            lpy(j) = model.py.logpdf(y, xp(:, j), theta);
        end
    end
    
    %                 Likelihood*Transition*Trajectory
    % Final weight =  --------------------------------
    %                       Importance Density
    lv = lpy + lpx + lw - lq;
end
