function lv = calculate_weights_bootstrap_rbpf(model, y, sp, mzp, Pzp, alpha, ~, lqalpha, ~, ~, ~, lw, theta)
% # Particle weights for the Rao-Blackwellized bootstrap particle filter
% ## Usage
% * `lv = calculate_weights_bootstrap_rbpf(model, y, sp, mzp, Pzp, alpha, lqs, lqalpha, s, mz, Pz, lw, theta)`
%
% ## Description
% Calculates the particle weights for the bootstrap particle filter. In 
% this case, the incremental weights are given by
%
%     w[n] ~= p(y[n] | s[0:n], y[1:n-1])*w[n-1].
%
% Note that the function actually computes the non-normalized log weights
% for numerical stability.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `sp`: ds-times-J matrix of newly drawn particles for the state `s[n]`.
% * `mzp`: dz-times-J matrix of predicted means `mz[n|n-1]`.
% * `Pzp`: dz-times-dz-times-J matrix of predicted covariances `Pz[n|n-1]`.
% * `alpha`: 1-times-J vector of ancestor indices for the state `s[n]`.
% * `lqs`: 1-times-J vector of importance density evaluations for 
%   `sp(:, j)`.
% * `lqalpha`: 1-times-J vector of importance density evaluations for the
%   ancestor indices `alpha(j)`.
% * `s`: ds-times-J matrix of previous state particles `s[n-1]`.
% * `mz`: dz-times-J matrix of previous means `mz[n-1|n-1]`.
% * `Pz`: dz-times-dz-times-J array of previous covariances `Pz[n-1|n-1]`.
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

    narginchk(13, 13);
    if model.pym.fast
        lv = model.pym.logpdf(y, sp, mzp, Pzp, theta);
    else
        J = size(sp, 2);
        lv = zeros(1, J);
        for j = 1:J
            lv(:, j) = model.pym.logpdf(y, sp(:, j), mzp(:, j), Pzp(:, :, j), theta);
        end
    end
    
    % Final weight: Incremental weight + trajectory weight - ancestor index
    % weights
    %
    % N.B.:
    % * If no resampling has taken place, then lqalpha is log(1/J)
    % * If resampling has taken place, then lqalpha is equal to lw
    % Thus, lw cancels if resampling has taken place, not otherwise.
    lv = lv + lw(alpha) - lqalpha;
end
