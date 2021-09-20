function lv = calculate_weights_rbpf(model, y, sp, mzp, Pzp, alpha, lqs, lqalpha, s, mz, Pz, lw, theta)
% # Generic particle weights for the Rao-Blackwellized particle filter
% ## Usage
% * `lv = calculate_weights_bootstrap(model, y, sp, mzp, Pzp, alpha, lqs, lqalpha, s, mz, Pz, lw, theta)`
%
% ## Description
% Calculates the particle weights for the generic Rao-Blackwellized 
% particle filters. In this case, the weights are given by
%
%             p(y[n] | s[0:n], y[1:n-1])*p(s[n] | s[0:n-1], y[1:n-1]) w[n-1]
%     w[n] ~= --------------------------------------------------------------
%                                       q(s[n])
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
    
    % Get ancestor particles, mean/covariance, and trajectory weight
    s = s(:, alpha);
    mz = mz(:, alpha);
    Pz = Pz(:, :, alpha);
    lw = lw(alpha);
        
    % Evaluate transition density and likelihood
    J = size(sp, 2);
    lps = zeros(1, J);
    lpy = lps;
    if model.psm.fast && model.pym.fast
        lps = model.psm.logpdf(sp, s, mz, Pz, theta);
        lpy = model.pym.logpdf(y*ones(1, J), sp, mzp, Pzp, theta);
    else
        for j = 1:J
            lps(j) = model.psm.logpdf(sp(:, j), s(:, j), mz(:, j), Pz(:, :, j), theta);
            lpy(j) = model.pym.logpdf(y, sp(:, j), mzp(:, j), Pzp(:, :, j), theta);
        end
    end
    
    %                 Likelihood*Transition*Trajectory
    % Final weight =  --------------------------------
    %                       Importance Density
    lv = lpy + lps + lw - lqalpha - lqs;
end
