function xp = sample_generic(model, y, x, theta, q)
% # Sample from an arbitrary importance density
% ## Usage
% * `xp = sample_generic(model, y, x, theta, q)`
%
% ## Description
% Samples a set of new particles x[n] from the importance distribution 
% q(x[n]).
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `x`: dx-times-J particle matrix at x[n-1].
% * `theta`: Additional parameter vector.
% * `q`: Importance density such that x[n] ~ q(x[n] | x[n-1], y[n]).
%
% ## Output
% * `xp`: The new samples x[n].
%
% ## See also
% calculate_incremental_weights_generic
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

    narginchk(5, 5);
    [dx, J] = size(x);
    if q.fast
        xp = q.rand(y*ones(1, J), x, theta);
    else
        xp = zeros(dx, J);
        for j = 1:J
            xp(:, j) = q.rand(y, x(:, j), theta);
        end
    end
end
