function dGxdx = multiricker_dGxdx(x, theta)
% # Cell array of second derivatives of multiple Ricker model
% ## Usage
% * `dGxdx = multiricker_dGxdx(x, theta)`
%
% ## Description
% Calculates the matrices of second derivatives as required by
% `sample_gaussian_flow` for its weight integration.
%
% ## Input
% * `x`: dx-times-1 state vector.
% * `theta`: dx-times-1 vector of `0`s and `1`s as indicators of which
%   populations are measured at the current timestep.
%
% ## Output
% * `dGxdx`: 1-times-dx cell array of second derivative matrices.
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

    narginchk(2, 2);
    dx = size(x, 1);        % No. of states
    dy = sum(theta == 1);   % No. of measurements
    iy = 1;                 % Counter to keep track of which measurement we've processed already
    
    dGxdx = cell([1, dx]); 
    for n = 1:dx
        dGxdxn = zeros(dy, dx);
        if theta(n) == 1
            dGxdxn(iy, n) = exp(x(n, :));
            iy = iy+1;
        end
        dGxdx{n} = dGxdxn;
    end
end
