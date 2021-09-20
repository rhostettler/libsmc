function [sp, alpha, lqs, lqalpha, qstate] = sample_bootstrap_rbpf(model, ~, s, mz, Pz, lw, theta, par)
% # Sample from the Rao-Blackwellized bootstrap proposal for CLGSS models
% ## Usage
% * `[sp, alpha, lqs, lqalpha, qstate] = sample_bootstrap_rbpf(model, y, s, mz, Pz, lw, theta, par)`
% 
% ## Description
% Implements the bootstrap importance density for Rao-Blackwellized
% condtionally linear Gaussian state-space models, that is, samples
%
%   s[n] ~ p(s[n] | s[0:n-1], y[1:n-1]).
% 
% ## Input
% * `model`: State-space model structure. In addition to the standard
%   fields, the additional fields for CLGSSMs for RBPFs are required, see
%   `rbpf()`.
% * `y`: dy-times-1 measurement vector.
% * `s`: ds-times-J matrix of particles `s[n-1]`.
% * `mz`: dz-times-J matrix of conditional means.
% * `Pz`: dz-times-dz-times-J array of conditional covariances.
% * `lw`: 1-times-J vector of log-weights `log(w[n-1])`.
% * `theta`: Additional model parameters.
% * `par`: Struct of additional algorithm parameters:
%     - `resample(lw)`: Resampling function.
%
% ## Output
% * `sp`: Newly sampled particles `s[n]`.
% * `alpha`: Ancestor indices.
% * `lqs`: Log of the importance density for each particle.
% * `lqalpha`: Log of the importance density for the ancestor weights.
% * `qstate`: Sampler state.
% 
% ## Authors
% 2021-present -- Roland Hostettler

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

    narginchk(7, 8);
    if nargin < 8 || isempty(par)
        par = struct();
    end
    defaults = struct( ...
        'resample', @resample_ess ...
    );
    par = parchk(par, defaults);
    
    %% Sampling
    % Resample
    [alpha, lqalpha, qstate] = par.resample(lw);
    s = s(:, alpha);
    mz = mz(:, alpha);
    Pz = Pz(:, :, alpha);

    % Sample new particles from the transition density
    if model.psm.fast
        sp = model.psm.rand(s, mz, Pz, theta);
        lqs = model.psm.logpdf(sp, s, mz, Pz, theta);
    else
        [ds, J] = size(s, 2);
        sp = zeros(ds, J);
        lqs = zeros(1, J);
        for j = 1:J
            sp(:, j) = model.psm.rand(s(:, j), mz(:, j), Pz(:, :, j), theta);
            lqs(:, j) = model.psm.logpdf(sp(:, j), s(:, j), mz(:, j), Pz(:, :, j), theta);
        end
    end
end
