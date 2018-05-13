function [x, sys] = cpfas(y, t, model, xt, q, M, par)
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

    %% Parameter check & defaults
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(3, 7);
%    modelchk(model);
    
    if nargin < 5 || isempty(q)
        % Default proposal: Bootstrap
        q = struct();
        q.fast = model.px.fast;
        q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
        q.pdf = @(xp, y, x, t) model.px.pdf(xp, x, t);
        q.rand = @(y, x, t) model.px.rand(x, t);
        q.bootstrap = 1;
    end
    
    if nargin < 6 || isempty(M)
        % Default no. of particles
        M = 100;
    end
    
    if nargin < 7
        % Default parameters
        par = [];
    end
    % No parameters as of yet; may as well change
%     def = struct(...
%         'xt', [] ...      % Default trajectory
%     );
%     par = parchk(par, def);

    %% Initialize seed trajectory
    % If no trajectory is given (e.g. for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which should help
    % speed up convergence.
    if nargin < 4 || isempty(xt) || all(all(xt == 0))
        % Default trajectory: Use a regular PF to calculate a degenerate 
        % trajectory (see below)
        % TODO: I need to see how to properly handle the augmentation of
        % the y and t vectors. Right now, gibbs_pmcmc adds a zero in t, but
        % nothing in y. This seems broken, but removing it at this point
        % looks like it will break parameter sampling.
        [~, sys] = bootstrap_pf(y, t(2:end), model, M);
        beta = sysresample(sys(end).wf);
        j = beta(randi(M, 1));
        % TODO: this is slow; need to figure out a better way
        N = length(sys);
        xt = zeros(size(sys(1).xf, 1), length(sys));
        for n = 1:N
            xt(:, n) = sys(n).xf(:, j);
        end
    end
    
    %% Prepare and preallocate
    Nx = size(model.px0.rand(1), 1);
    [Ny, N] = size(y);
    N = N+1;
    
    % Prepend t[0] to t and y
    t = [0, t];
    y = [NaN*ones(Ny, 1), y];
    
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
        %% Sample particles
        alpha = sysresample(w);
        xp = sample_q(y(:, n), x(:, alpha), t(n), q);   % Draw M-1 particles
        xp(:, M) = xt(:, n);                            % Set Mth particle
        
        %% Sample ancestor index
        % TODO: Make generic (i.e., make so that we take the ancestor
        % sampling function as a parameter)
        if 1
            % Conventional ancestor sampling
            alpha(M) = sample_ancestor_index(xt(:, n), x, t(n), lw, model);
        else
            % Ancestor sampling based on rejection sampling
            [alpha(M), rs(n)] = sample_ancestor_index_rs(xt(:, n), x, t(n), lw, model);
        end
        
        %% Calculate weights
        % Incremental weights
        % TODO: This needs to be taken from the par-structure
        lw = calculate_incremental_weights_generic(y(:, n), xp, x(:, alpha), t(n), model, q);
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

%% Draw ancestor index for seed trajectory
function alpha = sample_ancestor_index(xt, x, t, lw, model)
    M = size(x, 2);
    lv = calculate_ancestor_weights(xt, x, t, lw, model.px);
    v = exp(lv-max(lv));
    v = v/sum(v);
    tmp = sysresample(v);
    alpha = tmp(randi(M, 1));
end

%% 
% TODO:
%   * This is very ad-hoc right now
% 	* We should also implement an adaptive version which uses the
%     newly drawn weight in the proposal
%   * We might as well sample from the prior; in fact, we alread resampled
%     (outside of the function), thus we could sample from the prior by
%     sampling random integers, which would be as efficient.
function [alpha, accepted] = sample_ancestor_index_rs(xt, x, t, lw, model)
    M = size(x, 2);
    J = 10;
    j = 0;
    done = 0;
    lv = lw + log(model.px.rho);
    %lv = calculate_ancestor_weights(xt, x, t, lw, model.px);
    iv = zeros(1, M);
    while ~done
        % Propose sample
        alpha = randi(M, 1);
        
        % Calculate non-normalized weight
%        lv(alpha) = lw(alpha) + model.px.logpdf(xt, x(:, alpha), t);
        lv(alpha) = calculate_ancestor_weights(xt, x(:, alpha), t, lw(alpha), model.px);
        iv(alpha) = 1;
        
        % Calculate upper bound on normalizing constant
        %rho = sum(exp(lv));
        rho = exp(max(lv));
        kappa = 1; % M / M
        
        u = rand(1);
        paccept = (exp(lv(alpha))/(kappa*rho));
        if paccept > 1
            warning('Acceptance probability larger than one, check your bounding constant.');
        end
        accepted = (u < paccept);
        
        j = j+1;
        done = accepted || (j >= J);
    end
    if ~accepted
        % Exhaustive search for the non-calculated ones
        lv(~iv) = calculate_ancestor_weights(xt, x(:, ~iv), t, lw(~iv), model.px);
        v = exp(lv-max(lv));
        v = v/sum(v);
        tmp = sysresample(v);
        alpha = tmp(randi(M, 1));
    end
end

%% Calculate ancestor weigths
function lv = calculate_ancestor_weights(xt, x, t, lw, px)
    M = size(x, 2);
    if px.fast
        lv = lw + px.logpdf(xt*ones(1, M), x, t);
    else
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = lw(m) + px.logpdf(xt, x(:, m), t);
        end
    end
end
