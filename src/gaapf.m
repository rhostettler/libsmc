function [xhat, sys] = gaapf(y, t, model, M, par)
% Auxiliary Particle filter (Gaussian approximation)
%
% USAGE
%   xhat = GAAPF(y, t, model)
%   [xhat, sys] = GAAPF(y, t, model, M, par)
%
% DESCRIPTION
%   Auxiliary particle filter (APF) that uses a Gaussian approximation for
%   the optimal adjustment multipliers as well as the optimal importance
%   density.
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
%   2018-12-06 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Add possibility of adding output function (see gibbs_pmcmc())
%   * Add a field to the parameters that can be used to calculate custom
%     'integrals'
%   * Update the description to more appropriately reflect what's being
%     done
%   * 

    %% Defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = struct();
    end
    def = struct( ...
        'resample', @resample_ess, ... % TODO: This can probably be used for the sampling function
        'calculate_proposal', [] ...
    );
    par = parchk(par, def);
    modelchk(model);
    if isempty(par.calculate_proposal)
        error('Please specify the moment approximation function.');
    end

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
        return_sys = true;
    else
        return_sys = false;
    end
    xhat = zeros(Nx, N-1);
    
    % preallocate some more    
    Ny = size(y, 1);
    [Nx, J] = size(x);

    px = model.px;
    py = model.py;
    
    mx = zeros(Nx, J);
    Px = zeros(Nx, Nx, J);
    my = zeros(Ny, J);
    Py = zeros(Ny, Ny, J);
    Pxy = zeros(Nx, Ny, J);
    mp = zeros(Nx, J);
    Pp = zeros(Nx, Nx, J);
    
    xp = zeros(Nx, J);
    lv = zeros(1, J);    
    
    %% Process Data
    for n = 2:N
        %% Calculate proposal moments
        % TODO: Can we parallelize / use 'fast' here as well?
        for j = 1:J
            [mp(:, j), Pp(:, j), my(:, j), Py(:, :, j), Pxy(:, :, j)] = par.calculate_proposal(y(:, n), x(:, j), t(n));

            % Calculate the approximation for p(x[n] | x[n-1], y[n])
if 0
            K = Pxy(:, :, j)/Py(:, :, j);
            mp(:, j) = mx(:, j) + K*(y(:, n) - my(:, j));
            Pp(:, :, j) = Px(:, :, j) - K*Py(:, :, j)*K';
end

            % Resampling weights, use the approximation p(y[n] | x[n-1])
            lv(j) = lw(j) + logmvnpdf(y(:, n).', my(:, j).', Py(:, :, j)).';
        end
        v = exp(lv-max(lv));
        v = v/sum(v);

        %% Sample
        % Sample auxiliary variables
        alpha = sysresample(v);

        % Sample state and calculate importance weights
        for j = 1:J
            % Draw a new sample
            xp(:, j) = mp(:, alpha(j)) + chol(Pp(:, :, alpha(j))).'*randn(Nx, 1);

            % Calculate importance weight
            lw(:, j) = ( ...
                py.logpdf(y(:, n), xp(:, j), t(n)) + px.logpdf(xp(:, j), x(:, alpha(j)), t(n)) ...
                - logmvnpdf(xp(:, j).', mp(:, alpha(j)).', Pp(:, :, alpha(j)).').' ...
            );
        end
        x = xp;
        w = exp(lw-max(lw));
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
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys, 1:M);
    end
end
