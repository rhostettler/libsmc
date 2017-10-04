function [x, sys] = pgas(y, t, model, q, M, par)
% Particle Gibbs with ancestor sampling for Markovian systems
%
% SYNOPSIS
%   xhat = cpfas(y, t, model, q)
%   [xhat, sys] = cpfas(y, t, model, q, M, par)
% 
% DESCRIPTION
%   
%
% PARAMETERS
%   y       Measurement data (Ny*N)
%
%   t       Time vector (1*N)
%
%   model   Model structure
%
%   q       Proposal density (optional, default: bootstrap proposal)
%
%   M       No. of particles (optional, default: 100)
%
%   par     Additional parameters
%
% RETURNS
%   x       A newly sampled trajectory (Nx*N)
%
%   sys     Particle system containing the following fields:
%
%               xf  The filtered particles (i.e. particles of the marginal
%                   filtering distribution at time t[n])
%               wf  The weights corresponding to the particles in xf
%               x   (Degenerate) state trajectories
%
% VERSION
%   2017-06-18
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Documentation
%   * Implement similar 0:N as in gprbpgas
%   * There seems to be a problem with the naming convention for the
%     samples in the particle system among the different methods (sys.x,
%     sys.w vs sys.xf, sys.wf)
%   * Implement rejection sampling-based ancestor index sampling
%   * Maybe replace par with xt? That seems more appropriate.

    %% Parameter check & defaults
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(3, 6);
    modelchk(model);
    
    % Default proposal
    if nargin < 4 || isempty(q)
        q = struct();
        q.fast = model.px.fast;
        q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
        q.pdf = @(xp, y, x, t) model.px.pdf(xp, x, t);
        q.rand = @(y, x, t) model.px.rand(x, t);
        q.bootstrap = 1;
    end
    
    % Default no. of particles
    if nargin < 5 || isempty(M)
        M = 100;
    end
    
    % Default parameters
    if nargin < 6
        par = [];
    end
    def = struct(...
        'xt', [] ...      % Default trajectory
    );
    par = parchk(par, def);

    %% Initialize seed trajectory
    % If no trajectory is given (e.g. for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which should help
    % speed up convergence.
    xt = par.xt;
    if isempty(xt)
        [~, ~, sys] = bootstrap_pf(y, t, model, M);
        beta = sysresample(sys.wf);
        j = beta(randi(M, 1));
        xt = squeeze(sys.xf(:, j, :));
        
        % TODO: Initial state?
    end
    
    %% Initialize
    % TODO: we should also take xt(0) here
    x = model.px0.rand(M);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    %% Preallocate
    N = size(y, 2);
    Nx = size(x, 1);
    rs = zeros(1, N);
    xf = zeros(Nx, M, N);
    alphaf = zeros(1, M, N);
    wf = zeros(1, M, N);
    
    %% Iterate over the data
    for n = 1:N
        % Draw anscetor indices
        alpha = sysresample(w);
        if 0
            % Conventional ancestor sampling
            alpha(M) = draw_ancestor_index(xt(:, n), x, t(n), lw, model);
        else
            % Ancestor sampling based on rejection sampling
            [alpha(M), rs(n)] = sample_ancestor_index_rs(xt(:, n), x, t(n), lw, model);
        end
        
        % Draw & set particles
        xp = draw_samples(y(:, n), x(:, alpha), t(n), q);
        xp(:, M) = xt(:, n);
        
        % Calculate particle weights
        [~, lw] = calculate_incremental_weights(y(:, n), xp, x(:, alpha), t(n), model, q);

        %% Set particles and weights
        x = xp;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);

        if any(isnan(w)) || any(w == Inf)
            warning('NaN and/or Inf in particle weights.');
        end

        %% Store
        alphaf(:, :, n) = alpha;
        xf(:, :, n) = x;
        wf(:, :, n) = w;
    end
    
    %% Store particle system
    if nargout >= 2
        sys = struct();
        sys.xf = xf;
        sys.wf = wf;
        sys.rs = rs;
    end
    
    %% Sample trajectory
    % Put together trajectories; moved here for performance reasons
    % Walk down the ancestral tree, backwards in time to get the correct
    % lineage.
    alpha = 1:M;
    xf(:, :, N) = xf(:, :, N);
    for n = N-1:-1:1
        xf(:, :, n) = xf(:, alphaf(:, alpha, n+1), n);
        alpha = alphaf(:, alpha, n+1);
    end
    
    % Store the complete trajectories
    if nargout >= 2
        sys.x = xf;
    end
    
    % Draw a trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    x = squeeze(xf(:, j, :));
    
    if 0
    fprintf('Acceptance rate is %.2f\n', sum(rs)/N);
    end
end

%% Draw ancestor index for seed trajectory
function alpha = draw_ancestor_index(xt, x, t, lw, model)
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
