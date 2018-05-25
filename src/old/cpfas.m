function [xhat, Phat, sys] = cpfas(y, t, model, q, M, par)
% Conditional particle filter with ancestor sampling (CPF-AS)
%
% SYNOPSIS
%   xhat = cpfas(y, t, model, q)
%   [xhat, Phat, sys] = cpfas(y, t, model, q, M, par)
% 
% DESCRIPTION
%   
%
% PARAMETERS
%
% RETURNS
%
% SEE ALSO
%   cpfas_ps
%
% VERSION
%   2017-04-07
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Fix documentation
%   * Move 'calculate_incremental_weights()' and possibly 'draw_samples()' 
%     out of function; combine it with 'sisr_pf()'

%% From gibbs_pmcmc => Draws a new state trajectory
% TODO: cpfas needs to be improved as follows to be directly compatible
%   * cpfas does not return the initial state 
%   * cpfas returns all trajectories, but we need to return only one
%   * We need to fix those things before removing this function
error('FIXME, lots to do here!');
if 0
function x = sample_trajectory(y, x, t, theta_f, theta_y, create_model, par)
    % Crate a new model with updated parameters and set seed trajectory
    model = create_model([theta_f; theta_y]);
    par.xt = x;
    
    % Run CPF & draw trajectory
    [~, ~, sys] = cpfas(y, t, model, [], par.M, par);    
    beta = sysresample(sys.wf);
    j = beta(randi(par.M, 1));
    x = squeeze(sys.xf(:, j, :));
end
end

    %% Parameter check & defaults
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(3, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'xt', [], ...      % Default trajectory
        'bootstrap', 0 ... % Use bootstrap proposal?
    );
    par = parchk(par, def);
    [px, ~, px0] = modelchk(model);
    
    %% Default Proposal
    if nargin < 4 || isempty(q)
        q = struct();
        q.fast = model.px.fast;
        q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
        q.pdf = @(xp, y, x, t) px.pdf(xp, x, t);
        q.rand = @(y, x, t) model.px.rand(x, t);
        par.bootstrap = 1;
    end
    
    %% Initialize
    x = px0.rand(M);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    %% Preallocate
    N = size(y, 2);
    Nx = size(x, 1);
    xt = par.xt;
    if isempty(xt)
        [~, ~, sys] = bootstrap_pf(y, t, model, M);
        beta = sysresample(sys.wf);
        j = beta(randi(M, 1));
        xt = squeeze(sys.xf(:, j, :));
    end
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    if nargout == 3
        sys.xf = zeros(Nx, M, N);
        sys.xf2 = zeros(Nx, M, N);
        alphas = zeros(1, M, N);
        sys.x = zeros(Nx, M, N);
        sys.w = zeros(1, M, N);
        sys.lw = zeros(1, M, N);
    end
    
    %% Iterate over the data
    for n = 1:N
        %% Propagate the m = 1, ..., M-1 particles
        % Draw anscetor indices
        alpha = sysresample(w);
if 0
        for m = 1:M
            % Draw new samples
            xp(:, m) = q.rand(y(:, n), x(:, alpha(m)), t(n));

            % Calculate the ancestor weights for the Mth particle
            lv(m) = lw(m) + px.logpdf(xt(:, n), x(:, m), t(n));
            
            % Calculate the particle weights
            [~, lw(m)] = par.incremental_weights(y(:, n), xp(:, m), x(:, alpha(m)), t(n));
        end
end
        xp = draw_samples(y(:, n), x(:, alpha), t(n), q);
        lv = calculate_ancestor_weights(xt(:, n), x, t(n), lw, px);
        [~, lw] = calculate_incremental_weights(y(:, n), xp, x(:, alpha), t(n), model, q, par);
        
        %% Propagate the Mth particle
        % Set particle
        xp(:, M) = xt(:, n);

        % Draw ancestor index
        v = exp(lv-max(lv));
        v = v/sum(v);
        tmp = sysresample(v);
        alpha(M) = tmp(randi(M, 1));
        
        % Calculate the Mth particle weight
        [~, lw(M)] = calculate_incremental_weights(y(:, n), xp(:, M), x(:, alpha(M)), t(n), model, q, par);

        %% Set particles and weights
        x = xp;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        %% Extend trajectories
        if nargout == 3
            sys.x(:, :, n) = x;
            sys.w(:, :, n) = w;
            sys.lw(:, :, n) = lw;
            alphas(:, :, n) = alpha;
%            sys.xf(:, :, 1:n-1) = sys.xf(:, alpha, 1:n-1);
%            sys.xf(:, :, n) = x;
            sys.wf = w;
        end
        
        if any(isnan(w))
            aaa = 1;
        end
    end
    
    %% Post-processing
    % Put together trajectories; moved here for performance reasons
    % Walk down the ancestral tree, backwards in time to get the correct
    % lineage.
    if nargout == 3
        alpha = 1:M;
        sys.xf(:, :, N) = sys.x(:, :, N);
        for n = N-1:-1:1
            sys.xf(:, :, n) = sys.x(:, alphas(:, alpha, n+1), n);
            alpha = alphas(:, alpha, n+1);
        end
    end
end

%%
function xp = draw_samples(y, x, t, q)
    [Nx, M] = size(x);
    if q.fast
        xp = q.rand(y*ones(1, M), x, t);
    else
        xp = zeros(Nx, M);
        for m = 1:M
            xp(:, m) = q.rand(y, x(:, m), t);
        end
    end
end

%% 
function [v, lv] = calculate_incremental_weights(y, xp, x, t, model, q, par)
    if par.bootstrap
        [v, lv] = bootstrap_incremental_weights(y, xp, t, model);
    else
        [v, lv] = sis_incremental_weights(y, xp, x, t, model, q);
    end
end

%% 
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
