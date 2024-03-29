function [beta, state] = sample_backward_simulation2(model, y, xs, xf, lw, theta, par)
% # Backward trajectory simulation sampling for non-Markovian models
% ## Usage
% * `[beta, state] = sample_backward_simulation2(model, y, xs, x, lw, theta, par)`
%
% ## Description
% Samples the indices of the smoothed particle trajectories for
% backward-simulation particle smoothing. Calculates all the kernel weights
% prior to sampling, which corresponds to the original (slow) approach in
% [1] (but for non-Markovian state-space models).
%
% ## Input
% * `model`: State-space model struct.
% * `xs`: dx-times-Js matrix of smoothed particles at n+1, that is, 
%    x[n+1|N].
% * `x`: dx-times-Jf matrix of filtered particles at n, that is, x[n|n].
% * `lw`: dx-times-Jf matrix of log-weights of the filtered particles at n,
%    that is, lw[n|n].
% * `theta`: Additional parameters.
% * `par`: Algorithm parameters:
%     - `lv = calculate_backward_simulation_weights(model, y, xs, xf, theta)`:
%       Backward simulation smoothing kernel. This parameter *must* be
%       specified (default: `[]`).
%
% ## Output
% * `beta`: Sampled indices.
% * `state`: Sampler state (empty).
%
% ## References
% 1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
%    smoothing with application to audio signal enhancement," IEEE 
%    Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
%    2002.
%
% ## Authors
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

% TODO:
% * This is essentially the same code as the regular
%   sample_backward_simulation; check if we can unify the two.

    %% Defaults
    narginchk(6, 7);
    if nargin < 7
        par = struct();
    end
    def = struct( ...
        'calculate_backward_simulation_weights', [] ...
    );
    par = parchk(par, def);
    if isempty(par.calculate_backward_simulation_weights)
        error('Simulation weights function not specified');
    end
    
    %% Sampling
    state = [];
    Js = size(xs, 2);
    Jf = size(xf, 2);
    beta = zeros(1, Js);
    for j = 1:Js
        % Calculate the weights of the backward kernel
        lv = par.calculate_backward_simulation_weights(model, y, shiftdim(xs(:, j, :), 1), xf, theta);
        lwtilde = lw + lv;
        wtilde = exp(lwtilde-max(lwtilde));
        wtilde = wtilde/sum(wtilde);

        % Sample a particle index from the backward kernel to extend the
        % jth backward trajectory
        ir = resample_stratified(log(wtilde));
        beta(j) = ir(randi(Jf, 1));
    end
end
