function lpy = multiricker_lpy(y, x, theta, beta)
% # Log-likelihood for the multiple Ricker model
% ## Usage
% * `lpy = multiricker_lpy(y, x, theta, beta)`
%
% ## Description
% Wrapper of the log-likelihood for the multiiple Ricker model. Makes it 
% possible to use it% both with and without `theta` for the measurements 
% `y`.
%
% ## Input
% * `y`: dy-times-J or `dx-times-J` measurement vector.
% * `x`: dx-times-J matrix of particles.
% * `theta`: dx-times-1 vector of measurement indices.
% * `beta`: Skewness parameter.
%
% ## Output
% * `lpy`: 1-times-J vector of log-likelihoods.
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

    narginchk(4, 4);
    dy = size(y, 1);
    if sum(theta) == dy
        lpy = sum(loggpoissonpdf(y, exp(x(theta == 1, :)).*(1-beta), beta), 1);
    else
        lpy = sum(loggpoissonpdf(y(theta == 1, :), exp(x(theta == 1, :)).*(1-beta), beta), 1);
    end
end
