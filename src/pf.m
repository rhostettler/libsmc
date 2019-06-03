function [xhat, sys] = pf(y, t, model, M, par)
% Particle filter
%
% USAGE
%   xhat = PF(y, t, model)
%   [xhat, sys] = PF(y, t, model, M, par)
%
% DESCRIPTION
%   PF is a generic sequential importance sampling with resampling
%   particle filter.
%
%   By default, a bootstrap particle filter with the following
%   configuration is used:
%
%       - XXXX
%
%   However, essentially every aspect can be changed through additional
%   parameters. These are specified as 
%
%
% PARAMETERS
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
% RETURNS
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
% AUTHORS
%   2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Add possibility of adding output function (see gibbs_pmcmc())
%   * Add a field to the parameters that can be used to calculate custom
%     'integrals'
%   * update documentation
%   * rename to sir_pf or similar again?
%   * 't' is not 'time' anymore but a generic parameter; need proper
%     handling for that.

    %% Defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = struct();
    end
    def = struct(...
        'resample', @resample_ess, ...
        'update', @sis_update_bootstrap ...
    );
    par = parchk(par, def);
    modelchk(model);

    %% Initialize
    x = model.px0.rand(M);
    lw = log(1/M)*ones(1, M);
    
    % Prepend a non-measurement and initial time (zero)
    Ny = size(y, 1);
    y = [NaN*ones(Ny, 1), y];
    t = [0, t];
    
    %% Preallocate
    Nx = size(x, 1);
    N = length(t);
    if nargout >= 2
        sys = initialize_sys(N, Nx, M);
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).alpha = 1:M;
        sys(1).r = false;
        sys(1).q = [];
        return_sys = true;
    else
        return_sys = false;
    end
    xhat = zeros(Nx, N-1);
    
    %% Process Data
    for n = 2:N
        %% Update
        % Resample
        [alpha, lw, r] = par.resample(lw);
        
        % Sample new particles
        [x, lv, q] = par.update(y(:, n), x(:, alpha), t(n), model);
        
        % Calculate and normalize weights
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        
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
            sys(n).q = q;
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys, 1:M);
    end
end
