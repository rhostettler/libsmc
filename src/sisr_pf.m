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

    %% Preliminary Checks
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(4, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'resample', @resample_ess ... % Resampling function
    );
    par = parchk(par, def);
    [px, py, px0] = modelchk(model);

    %% Initialize
    x = px0.rand(M);
    lw = log(1/M)*ones(1, M);
    
    % TODO: Add initial samples to system, increase N by one, change loop
    % form 2:N
    
    %% Preallocate
    Nx = size(x, 1);
    N = length(t);
    if nargout >= 2
        sys = initialize_sys(N, Nx, M);
        return_sys = true;
    end
    xhat = zeros(Nx, N);
    
    %% Process Data
    for n = 1:N
        %% Resample
        [alpha, lw, r] = par.resample(lw, par);

        %% Draw Samples
        xp = sample_q(y(:, n), x(:, alpha), t(n), q);
        
        %% Weights
        lv = calculate_incremental_weights(y(:, n), xp, x, t(n), model, q);
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        x = xp;
        
        %% Point Estimates
        xhat(:, n) = x*w';

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
