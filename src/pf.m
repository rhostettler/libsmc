function [xhat, sys] = pf(model, y, theta, J, par)
% # Sequential importance sampling with resampling particle filter
% ## Usage
% * `xhat = sisr_pf(model, y)`
% * `[xhat, sys] = sisr_pf(model, y, theta, J, par)`
%
% ## Description
% `sisr_pf` is a generic sequential importance sampling with resampling
% particle filter (PF). It can be used as anything ranging from the 
% bootstrap PF to the auxiliary particle filter.
%
% In its minimal form, a bootstrap particle filter with conditional
% resampling based on the effective sample size with `J = 100` particles is
% used.
%
% ## Input
% **TODO** Update documentation from here on.
%   y       Ny times N matrix of measurements.
%   t       1 times N vector of timestamps.
%   model   State space model structure.
%   M       Number of particles (optional, default: 100).
%   par     Structure of additional (optional) parameters:
%
%           [alpha, lw, r] = resample(lw)
%               Function handle to the resampling function. The argument lw
%               is the log-weights and the must return the indices of the
%               resampled (alpha) particles, the weights of the resampled 
%               (lw) particles, as well as a bool indicating whether
%               resampling was performed or not.
%
% ## Outut
%   xhat    Minimum mean squared error state estimate (calculated using the
%           marginal filtering density).
%   sys     Particle system array of structs with the following fields:
%           
%               xf  Nx times M matrix of particles for the marginal
%                   filtering density.
%               wf  1 times M vector of the particle weights for the
%                   marginal filtering density.
%               af  1 times M vector of ancestor indices.
%               r   Boolean resampling indicator.
%
% ## Authors
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
% * Update documentation
% * Add possibility of adding output function (see gibbs_pmcmc())
% * Add a field to the parameters that can be used to calculate custom
%   'integrals'
% * update documentation
% * 't' is not 'time' anymore but a generic parameter; need proper
%   handling for that.

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
    if size(theta, 2) == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN, theta];
    
    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    if nargout >= 2
        sys = initialize_sys(N, dx, J);
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).alpha = 1:J;
        sys(1).r = false;
        sys(1).q = [];          % TODO: Should this be removed? Check.
        return_sys = true;
    else
        return_sys = false;
    end
    xhat = zeros(dx, N-1);
    
    %% Process Data
    for n = 2:N
        %% Update
        % Sample
        % TODO:
        % * Resampling will have to return the weights used for resampling
        %   in the future (for APF)
        % * par.sample will have to return the density sampled from in the 
        %   future
        [alpha, lw, r] = par.resample(lw);
        xp = par.sample(model, y(:, n), x(:, alpha), theta(n));
        
        % Calculate and normalize weights
        % TODO:
        % * This should take the importance density as an argument
        % * t should be theta
        % * To make this fit for APF, this will have to take the resampling weights into account
        lv = par.calculate_incremental_weights(model, y(:, n), xp, x, theta(n));
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
            sys(n).r = r;
%             sys(n).q = q;
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end
