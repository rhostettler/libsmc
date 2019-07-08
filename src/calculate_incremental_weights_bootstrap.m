function lv = calculate_incremental_weights_bootstrap(model, y, xp, ~, theta)
% # Incremental particle weights for the bootstrap particle filter
% ## Usage
% * `lv = calculate_incremental_weights_bootstrap(model, y, xp, ~, theta)`
%
% ## Description
% Calculates the incremental particle weights for the bootstrap particle
% filter. In this case, the incremental weight is given by
%
%     v[n] ~= p(y[n] | x[n]).
%
% Note that the function actually computes the non-normalized log weights
% for numerical stability.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: Measurement y[n].
% * `xp`: Particles at time t[n] (i.e. x[n]).
% * `x`: Particles at time t[n-1] (i.e. x[n-1]).
% * `theta`: Model parameters.
%
% ## Output
% * `lv`: The non-normalized log-weights.
%
% ## Author
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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

    narginchk(5, 5);
    J = size(xp, 2);
    py = model.py;
    if py.fast
        lv = py.logpdf(y*ones(1, J), xp, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = py.logpdf(y, xp(:, j), theta);
        end
    end
end
