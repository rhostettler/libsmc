function [xp, lqx, qstate] = sample_bootstrap(model, ~, x, theta)
% # Sample from the bootstrap importance density
% ## Usage
% * `[xp, lqx, qstate] = sample_bootstrap(model, y, x, theta)`
%
% ## Description
% Samples a set of new samples x[n] from the bootstrap importance density,
% that is, samples
%
%     x[n] ~ p(x[n] | x[n-1]).
%
% ## Input
% * `model`: State-space model struct.
% * `y`: Measurement vector y[n].
% * `x`: Samples at x[n-1].
% * `theta`: Model parameters.
%
% ## Output
% * `xp`: The new samples x[n].
% * `lqx`: 1-times-J vector of the importance density evaluated at 
%   `xp(:, j)`.
% * `qstate`: Empty.
%
% ## Author
% 2018-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * Consider rewriting this such that we just define the q used in this
%   case and then pass this to sample_generic, now that we have to redefine
%   the sampling density anyway.

    narginchk(4, 4);
    [dx, J] = size(x);
    lqx = zeros(1, J);
    qstate = [];
    px = model.px;
    if px.fast
        xp = px.rand(x, theta);
        lqx = px.logpdf(xp, x, theta);
    else
        xp = zeros(dx, J);
        for j = 1:J
            xp(:, j) = px.rand(x(:, j), theta);
            lqx(j) = px.logpdf(xp(:, j), x(:, j), theta);
        end
    end
end
