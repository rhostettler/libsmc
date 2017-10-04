function [xhat, sys] = bootstrap_pf2(y, t, model, M, par)
% Bootstrap particle filter for non-Markovian state-space models
%
% SYNOPSIS
%   [xhat, sys] = bootstrap_pf2(y, t, model)
%   [xhat, sys] = bootstrap_pf2(y, t, model, M, par)
%
% DESCRIPTION
%   Bootstrap particle filter for non-Markovian state-space models, i.e.
%   calculates the (degenerate) filtering distribution p(x[1:n] | y[1:n])
%   for systems where both the state transition density and the likelihood
%   may be non-Markovian, that is, p(x[n] | x[0:n-1]) and 
%   p(y[n] | x[1:n], y[1:n-1]), respectively.
%
% PARAMETERS
%   y       Ny x N matrix of measurements
%
%   t       1 x N vector of timestamps
%
%   model   State-space model structure
%
%   M       No. of particles (optional, default 100)
%
%   par     Struct of additional parameters with the following fields:
%
%               resample    Resampling function (default: resample_ess)
%
% RETURNS
%   xhat    MMSE estimate of the state (posterior mean)
%
%   sys     Particle system structure containing the following fields:
%
%           
% SEE ALSO
%   bootstrap_pf, ffbsi_ps2
%
% VERSION
%   2017-08-11
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Investigate if the initial state should be stored or not; see below

    %% Parameters and defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = [];
    end
    
    def = struct(...
        'resample', @resample_ess ... % Resampling function
    );
    par = parchk(par, def);
    [~, ~, px0] = modelchk(model);

    %% Preallocate
    Nx = size(px0.rand(1), 1);
    N = size(y, 2);
    xf = zeros(Nx, M, N+1);
    xhat = zeros(Nx, N);
    
    if nargout == 2
        sys = struct();
        sys.alphas = zeros(1, M, N);    % Resampling indices
        sys.x = zeros(Nx, M, N);        % Non-resampled particles
        sys.w = zeros(1, M, N);         % Non-resampled particle weights
        sys.lw = zeros(1, M, N);        % Non-resampled log-weights
        sys.r = zeros(1, N);            % Resampling indicator
    end
    
    if isempty(t)
        t = 1:N;
    end
    
    %% Initialize
    xf(:, :, 1) = px0.rand(M);
    lw = log(1/M)*ones(1, M);
    t = [0, t];
    
    %% Process data
    for k = 2:N+1
        n = k - 1;
        
        % Resample
        [alpha, lw, r] = par.resample(lw, par);
        xf(:, :, 1:k-1) = xf(:, alpha, 1:k-1);
        
        % Draw Samples
        xf(:, :, k) = sample_bootstrap2(xf(:, :, 1:k-1), t(1:k), model);
        
        % Calculate weights
        [~, lv] = calculate_incremental_weights_bootstrap2(y(:, 1:n), xf(:, :, 1:k), t(1:k), model);
        lw = lw+lv;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        % MMS estimate
        xhat(:, n) = xf(:, :, k)*w.';
        
        % Store the particle system
        if nargout == 2
            sys.x(:, :, n) = xf(:, :, k);   % Store particles
            sys.lw(:, :, n) = lw;           % Store particle weights (log)
            sys.w(:, :, n) = w;             % Store particle weights
			sys.r(:, n) = r;                % Store resampling indicator
            sys.alphas(:, :, n) = alpha;    % Store ancestor indices
        end
    end
    
    % Strip initial value from the full trajectories
    % TODO: We might want to keep that, actually
    if nargout == 2
        sys.xf = xf(:, :, 2:N+1);
    end
end
