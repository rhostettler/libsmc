function [x, sys] = cpfas(y, t, model, xt, M, par)
% Conditional Particle Filter with Ancestor Sampling
%
% USAGE
%   x = CPFAS(y, t, model)
%   [x, sys] = CPFAS(y, t, model, xt, q, M, par)
% 
% DESCRIPTION
%   
%
% PARAMETERS
%   y       Measurement data (Ny*N)
%   t       Time vector (1*N)
%   model   Model structure
%   xt      
%   q       Proposal density (optional, default: bootstrap proposal)
%   M       No. of particles (optional, default: 100)
%   par     Additional parameters
%
% RETURNS
%   x       A newly sampled trajectory (Nx*N)
%   sys     Particle system containing the following fields:
%
%               xf  The filtered particles (i.e. particles of the marginal
%                   filtering distribution at time t[n])
%               wf  The weights corresponding to the particles in xf
%               x   (Degenerate) state trajectories
%
%
%
% AUTHOR
%   2018-05-11 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Documentation
%   * Implement rejection sampling-based ancestor index sampling

    %% Defaults
    narginchk(3, 6);
%    modelchk(model);
        
    if nargin < 5 || isempty(M)
        % Default no. of particles
        M = 100;
    end

    % Default parameters (importance density, weights, etc.)
    if nargin < 6
        par = struct();
    end
    def = struct( ...
        'sample', @sample_bootstrap, ...
        'calculate_incremental_weights', @calculate_incremental_weights_bootstrap, ...
        'sample_ancestor_index', @sample_ancestor_index ...
    );
    par = parchk(par, def);

    % Prepend t[0] to t and y
    [Ny, N] = size(y);
    N = N+1;
    t = [0, t];
    y = [NaN*ones(Ny, 1), y];

    %% Initialize seed trajectory
    % TODO: This is still hacky, needs some more attention
    % If no trajectory is given (e.g. for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which should help
    % speed up convergence.
    if nargin < 4 || isempty(xt) || all(all(xt == 0))
        % Default trajectory: Use a regular PF to calculate a degenerate 
        % trajectory (see below)
        % TODO: bootstrap_pf can't cope with extra y and t yet, hence we
        % remove them again.
        % TODO: Should move to apf() once we have implemented that properly
        [~, sys] = bootstrap_pf(y(:, 2:end), t(2:end), model, M);
        beta = sysresample(sys(end).wf);
        j = beta(randi(M, 1));
        % TODO: this is slow; need to figure out a better way
        xt = zeros(size(sys(1).xf, 1), N);
        for n = 1:N
            xt(:, n) = sys(n).xf(:, j);
        end
    end
    
    %% Prepare and preallocate
    Nx = size(model.px0.rand(1), 1);
    
    % Preallocate
    sys = initialize_sys(N, Nx, M);
    
    %% Initialize
    % Draw initial particles
    x = model.px0.rand(M-1);
    x(:, M) = xt(:, 1);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    % Store initial state
    sys(1).x = x;
    sys(1).w = w;
    
    %% Iterate over the data
    for n = 2:N
        %% Sampling
        % Resample, then sample M-1 particles and set the Mth to the seed 
        % trajectory
        alpha = sysresample(w);                         % TODO: Should we be able to change this through par?
        xp = par.sample(y(:, n), x(:, alpha), t(n), model);
        xp(:, M) = xt(:, n);                            % Set Mth particle
        
        % Ancestor index (note: the ancestor weights have to be calculated
        % *inside* the sampling function).
        % TODO: State is used for diagnostics. Not sure if we're going to
        % use it or not, but most likely we'll attach it to sys()
        [alpha(M), state] = par.sample_ancestor_index(xt(:, n), x, t(n), lw, model);
        
        %% Calculate weights
        lw = par.calculate_incremental_weights(y(:, n), xp, x(:, alpha), t(n), model);
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);

        if any(isnan(w)) || any(w == Inf)
            warning('NaN and/or Inf in particle weights.');
        end
        
        % Set particles
        x = xp;

        %% Store
        % TODO: Here we will also store the rejection sampling indicator,
        % once that is properly migrated
        sys(n).x = x;
        sys(n).w = w;
        sys(n).alpha = alpha;
    end    
    
    %% Sample trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    sys = calculate_particle_lineages(sys, j);
    x = cat(2, sys.xf);
        
    if 0
    fprintf('Acceptance rate is %.2f\n', sum(rs)/N);
    end
end
