function [x, sys] = cpfas(model, y, xt, theta, J, par)
% Conditional particle filter with ancestor sampling
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
%   2017-2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Update documentation
%   * t is not t anymore; replace
%   * Merge non-Markovian stuff here; shouldn't be too difficult?
%   * sample_XXX functions should be re-implemented and changed in pf()

    %% Defaults
    narginchk(2, 6);
%    modelchk(model);

    if nargin < 4
        theta = [];
    end

    if nargin < 5 || isempty(J)
        % Default no. of particles
        J = 100;
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

        %% Prepare and preallocate
    % Prepend a NaN measurement (for x[0] where we don't have a 
    % measurement)
    [Ny, N] = size(y);
    N = N+1;
    y = [NaN*ones(Ny, 1), y];

    % Expand theta properly such that we have theta(:, n)
    [~, Nc] = size(theta);
    switch Nc
        case 0
            % Empty theta => create vector of NaNs
            theta = NaN*ones(1, N);
        case 1
            % Single (static parameter), expand to be Nc*N
            theta = theta*ones(1, N);
        case N-1
            theta = [NaN, theta];
        otherwise
            error('Parameter vector must either be empty, M x 1, or M x N');
    end
    
    % Determine state size
    Nx = size(model.px0.rand(1), 1);
    sys = initialize_sys(N, Nx, J);
    
    %% Initialize seed trajectory
    % If no trajectory is given (e.g. for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which helps to
    % speed up convergence.
    if nargin < 3 || isempty(xt) || all(all(xt == 0))
        % Default trajectory: Use a regular PF to calculate a degenerate 
        % trajectory (see below)
        [~, sys] = pf(y(:, 2:N), theta(:, 2:N), model, J);
        
        % Sample trajectory according to the final filter weights
        beta = sysresample(sys(end).wf);
        j = beta(randi(J, 1));
        xf = cat(3, sys.xf);
        xt = squeeze(xf(:, j, :));
    end
        
    %% Initialize
    % Draw initial particles
    x = model.px0.rand(J-1);
    x(:, J) = xt(:, 1);
    w = 1/J*ones(1, J);
    lw = log(1/J)*ones(1, J);
    
    % Store initial state
    sys(1).x = x;
    sys(1).w = w;
    
    %% Iterate over the data
    for n = 2:N
        %% Sampling
        % Resample, then sample J-1 particles and set the Jth to the seed 
        % trajectory
        alpha = sysresample(w);                         % TODO: Should we be able to change this through par?
        xp = par.sample(model, y(:, n), x(:, alpha), theta(:, n));
        xp(:, J) = xt(:, n);                            % Set Jth particle
        
        % Ancestor index (note: the ancestor weights have to be calculated
        % *inside* the sampling function).
        % TODO: State is used for diagnostics. Not sure if we're going to
        % use it or not, but most likely we'll attach it to sys()
        [alpha(J), state] = par.sample_ancestor_index(model, y(:, n), xt(:, n), x, lw, theta(:, n));
        
        %% Calculate weights
        lw = par.calculate_incremental_weights(model, y(:, n), xp, x(:, alpha), theta(:, n));
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
        sys(n).state = state;
    end
    
    sum(cat(1, sys(:).state)) %%%% TODO: Move this away
    
    %% Sample trajectory
    beta = sysresample(w);
    j = beta(randi(J, 1));
    sys = calculate_particle_lineages(sys, j);
    x = cat(2, sys.xf);
end
