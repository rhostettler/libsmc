function lv = calculate_incremental_weights_generic(model, y, xp, x, theta, q)
% # General incremental weights for sequential importance sampling
% ## Usage
% * `lv = calculate_incremental_weights_generic(model, y, xp, x, theta, q)`
%
% ## Description
% Calculates the incremental importance weight for sequential importance
% sampling for an arbitrary proposal density q.
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
% ## See also
% sample_generic
%
% ## Authors
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

% TODO:
% * Consider simply expanding q if it's a scalar; check computational
%   drawback.

    narginchk(6, 6);
    J = size(xp, 2);
    lv = zeros(1, J);
    px = model.px;
    py = model.py;
    if px.fast && py.fast && length(q) == 1 && q.fast
        lv = ( ...
            py.logpdf(y*ones(1, J), xp, theta) ...
            + px.logpdf(xp, x, theta) ...
            - q.logpdf(xp, y*ones(1, J), x, theta) ...
        );
    elseif length(q) == J
        for j = 1:J
            lv(j) = ( ...
                py.logpdf(y, xp(:, j), theta) ...
                + px.logpdf(xp(:, j), x(:, j), theta) ...
                - q(j).logpdf(xp(:, j), y, x(:, j), theta) ...
            );
        end
    else
        for j = 1:J
            lv(j) = ( ...
                py.logpdf(y, xp(:, j), theta) ...
                + px.logpdf(xp(:, j), x(:, j), theta) ...
                - q.logpdf(xp(:, j), y, x(:, j), theta) ...
            );
        end
    end
end
