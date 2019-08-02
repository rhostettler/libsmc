function [xhat, sys] = pf(model, y, theta, J, par)
% # Sequential importance sampling with resampling particle filter
% ## Usage
% * `xhat = pf(model, y)`
% * `[xhat, sys] = pf(model, y, theta, J, par)`
%
% ## Description
% `pf` is a generic sequential importance sampling with resampling particle
% filter (PF). It can be used as anything ranging from the bootstrap PF to
% the auxiliary particle filter.
%
% In its minimal form, a bootstrap particle filter with conditional
% resampling based on the effective sample size with `J = 100` particles is
% used.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: dtheta-times-1 vector or dtheta-times-N matrix of additional
%   parameters (default: `[]`).
% * `J`: Number of particles (default: 100).
% * `par`: Struct of additional (optional) parameters:
%     - `[alpha, lw, r] = resample(lw)`: Function handle to the resampling 
%       function. The input `lw` is the log-weights and the function must 
%       return the indices of the resampled particles (`alpha`), the log-
%       weights of the resampled (`lw`) particles, as well as a boolean
%       indicating whether resampling was performed or not (`r`). Default:
%       `@resample_ess`.
%     - `[xp, q] = sample(model, y, x, theta)`: Function handle to the 
%       sampling function to draw new state vectors. The output are the new
%       samples `xp` as well as the importance density `q` sampled from 
%       (struct or array of structs). Default: `@sample_bootstrap`.
%     - `lv = calculate_incremental_weights(model, y, xp, x, theta)`:
%       Function to calculate the weight increment `lv`. This function must
%       match the `sample` function. Default:
%       `@calculate_incremental_weights_bootstrap`.
%
% ## Output
% * `xhat`: Minimum mean squared error filter state estimate (calculated 
%   using the marginal filtering density).
% * `sys`: Particle system array of structs with the following fields:
%     - `x`: dx-times-J matrix of particles for the marginal filtering 
%       density.
%     - `w`: 1-times-J vector of the particle weights for the marginal
%       filtering density.
%     - `alpha`: 1-times-J vector of ancestor indices.
%     - `rstate`: Resampling algorithm state.
%
% ## Authors
% 2018-present -- Roland Hostettler

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
% * Add possibility of adding output function (see gibbs_pmcmc())
% * Add possibility of calculating arbitrary MC integrals based on the
%   marginal filtering density; defaults to mean.
% * In order to make it more generic (i.e., make it fit for APF), the
%   resampling-function should return the (log-)weights used for resampling
%   and calculate_incremental_weights should take these weights as an
%   input.
% * The interface for the sample() function should be updated such that it
%   returns the log-pdf of the samples, i.e., [xp, lqx, qstate] = sample(),
%   which is then taken as an extra input to
%   calculate_incremental_weights(). (The same type of interface should be
%   made for resample() to address the above comment).
% * Default to calculate_incremental_weights_generic() rather than
%   bootstrap? Seems to be the more common case nowadays.
% * Remove unnecessary incremental weight calculation functions

    %% Defaults
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
        'sample', @sample_bootstrap, ...
        'resample', @resample_ess, ...
        'calculate_incremental_weights', @calculate_incremental_weights_bootstrap ...
    );
    par = parchk(par, def);
%     modelchk(model);

    %% Initialize
    % Sample initial particles
    x = model.px0.rand(J);
    lw = log(1/J)*ones(1, J);
    
    % Since we also store and return the initial state (in 'sys'), a dummy
    % (NaN) measurement is prepended to the measurement matrix so that we 
    % can use consistent indexing in the processing loop.
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta' to the appropriate size, such that we can use
    % 'theta(n)' as an argument to the different functions (if not already
    % expanded).
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];
    
    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    return_sys = (nargout >= 2);
    if return_sys
        sys = initialize_sys(N, dx, J);
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).alpha = 1:J;
        sys(1).rstate = struct('r', false, 'ess', J);
        sys(1).q = [];
    end
    xhat = zeros(dx, N-1);
    
    %% Process Data
    for n = 2:N
        %% Update
        % (Re-)Sample
        [alpha, lw, rstate] = par.resample(lw);
        [xp, q] = par.sample(model, y(:, n), x(:, alpha), theta(:, n));
        
        % Calculate and normalize weights
        lv = par.calculate_incremental_weights(model, y(:, n), xp, x(:, alpha), theta(:, n), q);
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        x = xp;
        
        %% Point Estimate(s)
        % Note: We don't have a state estimate for the initial state (in
        % the filtered version, anyway), thus we save the MMSE estimate in
        % n-1.
        xhat(:, n-1) = x*w';

        %% Store
        if return_sys
            sys(n).x = x;
            sys(n).w = w;
            sys(n).alpha = alpha;
            sys(n).rstate = rstate;
            sys(n).q = q;
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end
