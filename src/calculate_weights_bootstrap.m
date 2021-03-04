function lv = calculate_weights_bootstrap(model, y, xp, ~, ~, ~, ~, theta)
% # Incremental particle weights for the bootstrap particle filter
% ## Usage
% * `lv = calculate_weights_bootstrap(model, y, xp, alpha, lq, x, lw, theta)`
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
% * `lv`: The non-normalized log-weights.
%
% ## Author
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
