function [xhat, sys] = pf2(model, y, theta, J, par)
% # Sequential importance sampling for non-Markovian state-space models
% ## Usage
% * `[xhat, sys] = pf2(model, y)
% * `[xhat, sys] = pf2(model, y, theta, J, par)
%
% ## Description
% Sequential importance sampling with resampling particle filter for non-
% Markovian state-space models. Calculates the (degenerate) filtering 
% distribution p(x[1:n] | y[1:n]) for systems where both the state 
% transition density and the likelihood may be non-Markovian of the form
%
%   x[n] ~ p(x[n] | x[0:n-1]),
%   y[n] ~ p(y[n] | x[1:n], y[1:n-1]).
%
% By default, the bootstrap proposal is used.
%
% Note that this is most likely not the most efficient implementation.
%
% ## Inputs
% * `model`: State-space model struct. Must contain the following structs.
%   The function interfaces are the same as for the Markovian models,
%   except that the conditional variables is a dx-times-n matrix.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: dtheta-times-1 vector or dtheta-times-N matrix of additional
%   parameters (default: `[]`).
% * `J`: Number of particles (default: 100).
% * `par`: Struct of additional (optional) parameters:
%      - `[alpha, lw, r] = resample(lw)`: Function handle to the resampling 
%        function. The input `lw` is the log-weights and the function must 
%        return the indices of the resampled particles (`alpha`), the log-
%        weights of the resampled (`lw`) particles, as well as a boolean
%        indicating whether resampling was performed or not (`r`). Default:
%        `@resample_ess`.
%      - `[xp, lqx] = sample(model, y, x, theta)`: Function handle to the
%        sampling function to draw new state vectors. The output are the
%        new samples `xp` as well as the importance density evaluated at
%        each `xp(:, j)` (default: `@sample`; bootstrap proposal, internal
%        function).
%      - `lv = calculate_incremental_weights(model, y, xp, x, theta, lqx)`:
%        Function to calculate the weight increment `lv`. This function 
%        must match the `sample` function (default:
%        `@calculate_incremental_weights`; generic internal function).
%
% ## Outputs
% * `xhat`: Minimum mean squared error filter state estimate (calculated 
%    using the marginal filtering density).
%  * `sys`: Particle system array of structs with the following fields:
%      - `x`: dx-times-J matrix of particles for the marginal filtering 
%        density.
%      - `w`: 1-times-J vector of the particle weights for the marginal
%        filtering density.
%      - `xf`: dx-times-J vector of state trajectory samples.
%      - `alpha`: 1-times-J vector of ancestor indices.
%      - `rstate`: Resampling algorithm state.
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

    %% Parameters and defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    if nargin < 5
        par = struct();
    end
    def = struct( ...
        'resample', @resample_ess, ... % Resampling function
        'sample', @sample ...
    );
    par = parchk(par, def);

    %% Preallocate
    % Since we also store and return the initial state (in 'sys'), a dummy
    % (NaN) measurement is prepended to the measurement matrix so that we 
    % can use consistent indexing in the processing loop.
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    N = N+1;
    
    % Expand 'theta' to the appropriate size, such that we can use
    % 'theta(n)' as an argument to the different functions (if not already
    % expanded).
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    else
        theta = [NaN*ones(dtheta, 1), theta];
    end

    % Preallocate full-state matrix.
    dx = size(model.px0.rand(1), 1);
    xf = zeros(dx, J, N);
    xhat = zeros(dx, N-1);

    % sys-struct, if required
    return_sys = (nargout >= 2);
    if return_sys
        sys = initialize_sys(N, dx, J);
    end
    
    %% Initialize
    xf(:, :, 1) = model.px0.rand(J);
    lw = log(1/J)*ones(1, J);
    if return_sys
        sys(1).x = xf(:, :, 1); % Non-resampled particles
        sys(1).w = exp(lw);     % Non-resampled particle weights
        sys(1).alpha = 1:J;     % Resampling indices
        sys(1).rstate = struct('r', false, 'ess', J);
%         sys(1).qstate = [];
    end
    
    %% Process data
    for n = 2:N
        %% Update
        % (Re-)Sample
        [alpha, lw, rstate] = par.resample(lw);
        xf(:, :, 1:n-1) = xf(:, alpha, 1:n-1);
        [xf(:, :, n), lqx] = par.sample(model, y(:, 1:n), xf(:, :, 1:n-1), theta(1:n));
        
        % Calculate weights
        lv = calculate_incremental_weights(model, y(:, 1:n), xf(:, :, 1:n), theta(1:n), lqx);
        lw = lw+lv;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        %% MMSE estimate
        xhat(:, n-1) = xf(:, :, n)*w.';
        
        %% Store
        if return_sys
            sys(n).x = xf(:, :, n);   % Store particles
            sys(n).w = w;             % Store particle weights
			sys(n).r = rstate;     % Store resampling state
            sys(n).alpha = alpha;     % Store ancestor indices
        end
    end
    
    % 
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end

%% Bootstrap proposal for non-Markovian SSMs
% Default importance density, slow implementation
function [xp, lqx] = sample(model, y, x, theta)
    [Nx, J, ~] = size(x);
    px = model.px;
    xp = zeros(Nx, J);
    lqx = zeros(1, J);
    for j = 1:J
        xj = shiftdim(x(:, j, :), 1);
        xp(:, j) = px.rand(xj, theta);
        lqx(:, j) = px.logpdf(xp(:, j), xj, theta);
    end
end

%% Calculate incremental weights for non-Markovian SSMs for the BPF
% Generic weight calculation function, slow implementation
function lv = calculate_incremental_weights(model, y, x, theta, lqx)
    [~, J, n] = size(x);
    lpy = zeros(1, J);
    lpx = zeros(1, J);
    for j = 1:J
        xj = shiftdim(x(:, j, :), 1);
        lpy(:, j) = model.py.logpdf(y, xj, theta);
        lpx(:, j) = model.px.logpdf(xj(:, n), xj(:, 1:n-1), theta);
    end
    lv = lpy + lpx - lqx;
end
