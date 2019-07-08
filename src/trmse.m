function [mrmse, varrmse] = trmse(e)
% # Time-averaged root mean squared error
% ## Usage
% * `mrmse = trmse(e)`
% * `[mrmse, varrmse] = trmse(e)`
%
% ## Description
% Calculates the mean (and variance) of the time-averaged root mean squared
% error (RMSE) for a set of Monte Carlo simulations.
% 
% ## Input
% * `e`: dx-times-N-times-L 3D matrix of errors (dx: state dimension; N:
%   number of time samples; L: number of Monte Carlo simulations).
%
% ## Output
% * `mrmse`: Mean of the time-averaged RMSE.
% * `varrmse`: Variance of the time-averaged RMSE.
%
% ## Authors
% 2019-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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

    narginchk(1, 1);
    ermse = sqrt(mean(sum(e.^2, 1), 2));
    mrmse = mean(ermse);
    varrmse = var(ermse);
end
