function [xhat, sys] = sisr_pf(y, t, model, q, M, par)
% Sequential importance sampling w/ resampling particle filter
%
% SYNOPSIS
%   xhat = SISR_PF(y, t, model, q)
%   [xhat, sys] = SISR_PF(y, t, model, q, M, par)
%
% DESCRIPTION
%   SISR_PF is a generic sequential importanc sampling with resampling
%   particle filter, that is, pretty much the most generic SIR-type filter.
%
%   Note that in this implementation, resampling is done before sampling
%   new states from the importance distribution, much like in the auxiliary
%   particle filter (but is different from the auxiliary particle filter in
%   that it generally doesn't make use of adjustment multipliers, even
%   though that can be implemented too by using an appropriate
%   'resampling()' function).
%
% PARAMETERS
%   y       Ny times N matrix of measurements.
%   t       1 times N vector of timestamps.
%   model   State space model structure.
%   q       Importance distribution structure.
%   M       Number of particles (optional, default: 100).
%   par     Structure of additional parameters:
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
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Add possibility of adding output function (see gibbs_pmcmc())
%   * Add a field to the parameters that can be used to calculate custom
%     'integrals'

    %% Defaults
    narginchk(4, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    def = struct(...
        'resample', @resample_ess, ...
        'calculate_incremental_weights', @calculate_incremental_weights_generic ...
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
        return_sys = true;
    end
    xhat = zeros(Nx, N-1);
    
    %% Process Data
    for n = 2:N
        %% Resample
        [alpha, lw, r] = par.resample(lw);

        %% Draw Samples
        xp = sample_q(y(:, n), x(:, alpha), t(n), q);
        
        %% Weights
        lv = par.calculate_incremental_weights(y(:, n), xp, x, t(n), model, q);
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
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys, 1:M);
    end
end
