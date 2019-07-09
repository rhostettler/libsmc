function lv = calculate_backward_simulation_weights(model, xs, x, theta)
% # Calculate backward simulation weights
% ## Usage
% * `lv = calculate_backward_simulation_weights(model, xs, x, theta)`
%
% ## Description
% Calculates the backward simulation weights needed in the Monte Carlo
% approximation of the smoothing kernel in forward-filtering
% backward-simulation smoothing [1].
%
% ## Input
% * `model`: State-space model structure.
% * `xs`: dx-times-1 smoothed particle at n+1, that is, x[n+1|N].
% * `x`: dx-times-Jf matrix of filtered particles at n, that is, x[n|n].
% * `theta`: Additional parameters.
% 
% ## Output
% * `lv`: Backward simulation weights.
% 
% ## References
% 1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
%    smoothing with application to audio signal enhancement," IEEE 
%    Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
%    2002.
%
% ## Author
% 2017-present -- Roland Hostettler

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
    J = size(x, 2);
    if model.px.fast
        lv = model.px.logpdf(xs*ones(1, J), x, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
           lv(j) = model.px.logpdf(xs, x(:, j), theta);
        end
    end
end
