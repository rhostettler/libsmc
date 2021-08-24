function lv = calculate_weights_bootstrap(model, y, xp, alpha, lqx, lqalpha, x, lw, theta)
% # Particle weights for the bootstrap particle filter
% ## Usage
% * `lv = calculate_weights_bootstrap(model, y, xp, alpha, lqx, lqalpha, x, lw, theta)`
%
% ## Description
% Calculates the particle weights for the bootstrap particle filter. In 
% this case, the incremental weight is given by
%
%     w[n] ~= p(y[n] | x[n])*w[n-1].
%
% Note that the function actually computes the non-normalized log weights
% for numerical stability.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `xp`: dx-times-J matrix of newly drawn particles for the state x[n].
% * `alpha`: 1-times-J vector of ancestor indices for the state x[n].
% * `lqx`: 1-times-J vector of importance density evaluations for 
%   `xp(:, j)`.
% * `lqalpha`: 1-times-J vector of importance density evaluations for the
%   ancestor indices `alpha(j)`.
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

    narginchk(9, 9);
    J = size(xp, 2);
    
    % Get trajectory weights
    % TODO: Here's the bug. Needs to be sorted out.
    lw = lw(alpha);
    
    % Incremental weights
    if model.py.fast
        lv = model.py.logpdf(y*ones(1, J), xp, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = model.py.logpdf(y, xp(:, j), theta);
        end
    end
    
    % Final weight: Incremental weight + trajectory weight - ancestor index
    % weights
    %
    % N.B.:
    % * If no resampling has taken place, then lqalpha is log(1/J)
    % * If resampling has taken place, then lqalpha is equal to lw
    % Thus, lw cancels if resampling has taken place, not otherwise.
    lv = lv + lw - lqalpha;
end
