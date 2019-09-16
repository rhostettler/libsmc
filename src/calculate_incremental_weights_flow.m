function lv = calculate_incremental_weights_flow(model, y, xp, x, theta, lqx)
% # Incremental weights for Gaussian particle flow OID approximation
% ## Usage
% * `lv = calculate_incremental_weights_flow(model, y, xp, x, theta, q)`
%
% ## Description
% Calculates the incremental particle weights when using the Gaussian
% particle flow OID approximation [1].
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `xp`: dx-times-J matrix of newly drawn particles for the state x[n].
% * `x`: dx-times-J matrix of previous state particles x[n-1].
% * `theta`: Additional parameters.
% * `q`: Importance density struct.
%
% ## Output
% * `lv`: Logarithm of incremental weights.
%
% ## References
% 1. P. Bunch and S. J. Godsill, "Approximations of the optimal importance 
%    density using Gaussian particle flow importance sampling," Journal of 
%    the American Statistical Association, vol. 111, no. 514, pp. 748–762, 
%    2016.
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
% * This is a dummy function right now, and we should properly integrate it
%   into the calculate_incremental_weigths_generic function (see remarks in
%   pf). It might also require rewriting the flow weight calculation
%   slightly (in sample_gaussian_flow()).

    narginchk(6, 6);
    lv = lqx;
end
