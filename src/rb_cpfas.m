function [x, sys] = rb_cpfas(y, t, model, M, par)
% Rao–Blackwellized conditional particle filter for mixed CLGSS models
%
% SYNOPSIS
%   x = rb_cpfas(y, t, model)
%   [x, sys] = rb_cpfas(y, t, model, M, par)
% 
% DESCRIPTION
%   Rao-Blackwellized particle Gibbs with ancestor sampling (conditional
%   particle filter) for mixed conditionally linear Gaussian state space
%   models of the form
%
%       s[n] = g(s[n-1]) + G(s[n-1]) z[n-1] + q^s[n]
%       z[n] = h(s[n-1]) + H(s[n-1]) z[n-1] + q^z[n]
%       y[n] ~ p(y[n] | s[n])
%       x[0] ~ N(m[0], P[0])
%
%   where [q^s[n]; q^z[n]] ~ N(0, Q(s[n-1])).
%
%   Note that the likelihood is assumed to be independent of z[n] and of 
%   arbitrary form, see [1] for details.
%
%   NOTE: In order to speed up the computations, it is assumed that G and H
%   don't depend on s[n-1] and and g and h return vectors if s[n-1] is a 
%   vector.
%
% PARAMETERS
%   y       Measurement data matrix (Ny x N)
%
%   t       Vector of sampling times or time differences between sampling
%           instants. If left empty, 1:N is used.
%
%   model   Struct describing the model. Requires the following fields:
%
%               m0, P0  Initial state and covariance
%               in, il  Index vector for the nonlinear and linear states
%                       such that s = x(in) and z = x(il)
%               fn, Fn  Functions g and G
%               fl, Fl  Functions h and H
%               Q, Qn,  Functions for the process noise covariance
%               Ql, Qnl        
%               py      Struct describing the likelihood
%
%   M       No. of particles to use in the particle filter (optional, 
%           default: 100)
%
%   par     Struct containing additional parameters:
%
%               xt      Seed trajectory (Nx*N+1); calculated using a
%                       regular PF if omitted.
%               Nbar    No. of samples into the future to consider when
%                       calculating the ancestor weights. By default, the
%                       complete future trajectory is considered.
%
% RETURNS
%   x       The sampled trajectory (Nx*N)
%
%   sys     The particle system containing the following fields:
%           
%               x       Matrix of particles (Nx*M*N)
%               w       Matrix with the particle weights (1*M*N)
%               P       Covariance matrices of the linear states (Nx*Nx*N)
%
% SEE ALSO
%   cpfas
%
% REFERENCES
%   [1] R. Hostettler, S. S?rkk?, S. J. Godsill, "Rao-Blackwellized
%       particle MCMC for parameter estimation in spatio-temporal Gaussian
%       processes," 2017, to appear.
%
% VERSION
%   2017-05-24
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * The initialize function should probably be replaced by a RBPF or
%     something.
%   * Should we return the smoothed covariance as well?

% NOTES
%   * Assumes implicitly that G, H, and Q don't depend on s[n-1] which
%     helps speeding up the calculation (parallelizable). The affected
%     functions are 'kf_update', 'calculate_ancesotor_weights', and
%     'draw_samples'.

    %% Defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = [];
    end
    
    [Ny, N] = size(y);
    if isempty(t)
        t = 0:N;
    end
    def = struct( ...
        'xt', [], ...  % Seed trajectory
        'Nbar', N ...  % Truncation length for calculating the ancestor weigths; 
    );
    par = parchk(par, def);
    
    %% Preallocate
    in = model.in;
    il = model.il;
    
    % Dimensions state dimensions and data length
    Ns = length(in);
    Nz = length(il);
    Nx = Ns+Nz;
    N = N+1;                            % +1 to include x[0]
    
    % Stores the full trajectories
    alphaf = zeros(1, M, N);    % Ancestor indices
    wf = zeros(1, M, N);        % Particle weights
    xf = zeros(Nx, M, N);       % Trajectories
    Pf = zeros(Nz, Nz, N);      % Covariances for linear states
    
    %% Seed Trajectory
    xt = par.xt;
    if isempty(xt) || sum(sum(xt)) == 0
        xt = initialize(y, t, model, M);
    end
    sbar = xt(in, :);
    
    %% Initialize
    % Prepend t_0 and no measurement
    t = [0, t];
    y = [NaN*ones(Ny, 1), y];
    
    % Initial samples
    m0 = model.m0;
    P0 = model.P0;
    s = m0(in)*ones(1, M) + chol(P0(in, in)).'*randn(Ns, M);
    s(:, M) = sbar(:, 1);
    z = m0(il)*ones(1, M) + P0(il, in)/P0(in, in)*(s - m0(in)*ones(1, M));
    P = P0(il, il) - P0(il, in)/P0(in, in)*P0(in, il);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    % Store initial
    wf(:, :, 1) = w;
    xf(in, :, 1) = s;
    xf(il, :, 1) = z;
    Pf(:, :, 1) = P;
    
    %% Process Data
    for n = 2:N
        % Draw new particles
        alpha = sysresample(w);
        sp = draw_samples(s(:, alpha), z(:, alpha), P, t(n), model);
        
        % Set Mth particle
        sp(:, M) = sbar(:, n);
        
        % Calculate ancestor weights and draw an index
        lv = calculate_ancestor_weights(sbar(:, n:N), s, z, P, t(n:N), lw, model, par.Nbar);
        v = exp(lv-max(lv));
        v = v/sum(v);
        tmp = sysresample(v);
        alpha(M) = tmp(randi(M, 1));

        % KF update
        [zp, Pp] = kf_update(z(:, alpha), P, sp, s(:, alpha), t(n), model);
                
        % Particle weights
        [~, lw] = bootstrap_incremental_weights(y(:, n), sp, t(n), model);

        % Normalize weights and update particles
        s = sp;
        z = zp;
        P = Pp;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        if any(isnan(w)) || any(w == Inf)
            warning('NaN and/or Inf in particle weights.');
        end
        
        %% Store New Samples & Ancestor Indices
        alphaf(:, :, n) = alpha;
        wf(:, :, n) = w;
        xf(in, :, n) = s;
        xf(il, :, n) = z;
        Pf(:, :, n) = P;
    end
    
    %% Store Particle System
    if nargout >= 2
        sys = struct();
        sys.x = xf;
        sys.w = wf;
        sys.P = Pf;
    end
    
    %% Draw Trajectory
    % Create full trajectories by walking down the ancestral tree backwards
    % in time to get the correct particle lineage.
    alpha = 1:M;
    xf(:, :, N) = xf(:, :, N);
    for n = N-1:-1:1
        xf(:, :, n) = xf(:, alphaf(:, alpha, n+1), n);
        alpha = alphaf(:, alpha, n+1);
    end
        
    % Draw a trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    x = squeeze(xf(:, j, :));
    
    %% Smooth Linear States
    x(il, :) = smooth_linear(x, Pf, t, model);
end

%% Initialization
function x = initialize(y, t, model, M)
% Calculates an initial seed trajectory by running a conventional PF to
% speed up convergence of the (outer) MCMC sampler.
%
% PARAMETERS
%   y       Measurement data
%   t       Time vector
%   model   Model struct
%   M       No. of particles
%
% RETURNS
%   x       The seed trajectory

    %% Preallocate
    in = model.in;
    il = model.il;
        
    Ns = length(in);
    Nz = length(il);
    Nx = Ns+Nz;
    [Ny, N] = size(y);
    N = N+1;
    
    % Stores the full trajectories
    alphaf = zeros(1, M, N);    % Ancestor indices
    xf = zeros(Nx, M, N);       % Trajectories
    
    %% Initialize
    t = [0, t];
    y = [zeros(Ny, 1), y];
    m0 = model.m0;
    P0 = model.P0;
    s = m0(in)*ones(1, M) + chol(P0(in, in)).'*randn(Ns, M);
    z = m0(il)*ones(1, M) + P0(il, in)/P0(in, in)*(s - m0(in)*ones(1, M));
    P = P0(il, il) - P0(il, in)/P0(in, in)*P0(in, il);
    w = 1/M*ones(1, M);
    
    % Store initial
    xf(in, :, 1) = s;
    xf(il, :, 1) = z;
    
    %% Process Data
    for n = 2:N
        % Draw new particles
        alpha = sysresample(w);
        sp = draw_samples(s(:, alpha), z(:, alpha), P, t(n), model);

        % KF update
        [zp, Pp] = kf_update(z(:, alpha), P, sp, s(:, alpha), t(n), model);
                
        % Particle weights
        [~, lw] = bootstrap_incremental_weights(y(:, n), sp, t(n), model);

        % Normalize weights and update particles
        s = sp;
        z = zp;
        P = Pp;
        w = exp(lw-max(lw));
        w = w/sum(w);
                
        % Store New Samples & Ancestor Indices
        alphaf(:, :, n) = alpha;
        xf(in, :, n) = s;
        xf(il, :, n) = z;
    end
    
    %% Sample Trajectory
    % Calculate complete trajectories
    alpha = 1:M;
    xf(:, :, N) = xf(:, :, N);
    for n = N-1:-1:1
        xf(:, :, n) = xf(:, alphaf(:, alpha, n+1), n);
        alpha = alphaf(:, alpha, n+1);
    end
        
    % Draw a trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    x = squeeze(xf(:, j, :));
end

%% RTS Smoothing
function [zs, Ps] = smooth_linear(x, P, t, model)
% Smoothing of linear states conditional on a complete trajectory of
% non-linear states.
% 
% PARAMETERS
%   x       Complete filtered state trajectory (including non-linear and 
%           linear states)
%   P       Covariances of the linear states
%   t       Time vector
%   model   Model struct
%
% RETURNS
%   zs      Smoothed trajectory of linear states
%   Ps      Smoothed covarainces

    %% Initialize
    % Get nonlinear and linear states
    s = x(model.in, :);
    z = x(model.il, :);
    
    % Preallocate
    [Nz, N] = size(z);
    zs = zeros(Nz, N);
    Ps = zeros(Nz, Nz, N);
    
    % Initialize backward pass
    zs(:, N) = z(:, N);
    Ps(:, :, N) = P(:, :, N);
    
    %% Backward Iteration
    for n = N-1:-1:1
        % Prediction from n to n+1
        h = model.fl(s(:, n), t(n+1));
        H = model.Fl(s(:, n), t(n+1));
        Qz = model.Ql(s(:, n), t(n+1));
        
        % RTS update
        zp = h + H*z(:, n);
        S = Qz + H*P(:, :, n)*H';
        L = P(:, :, n)*H'/S;
        zs(:, n) = z(:, n) + L*(zs(:, n+1) - zp);
        Ps(:, :, n) = P(:, :, n) + L*(Ps(:, :, n+1) - S)*L';
    end
end

%% Sampling
function sp = draw_samples(s, z, P, t, model)
% Draws new samples from the marginalized nonlinear dynamics.
%
% PARAMETERS
%   s       Old samples, s[n-1]
%   z       Mean of conditionally linear states at n-1, z[n-1]
%   P       Covariance of contidionally linear states at n-1, P[n-1]
%   t       Time t[n]
%   model   Model struct
%
% RETURNS
%   sp      New samples, s[n]

    [Ns, M] = size(s);
    
    % Get dynamics
    g = model.fn(s, t);
    G = model.Fn(s, t);
    Qs = model.Qn(s, t);
    
    % Marginalization
    m = g + G*z;
    C = Qs + G*P*G';
    
    % Generate samples
    LC = chol(C).';
    sp = m + LC*randn(Ns, M);
end

%% Ancestor Weights
function lv = calculate_ancestor_weights(sbar, s, z, P, t, lw, model, Nbar)
% Calculates the ancestor weights for the seed trajectory.
%
% PARAMETERS
%   sbar    Seed trajectory from n to N, i.e. bar{s}_{n:N} (Nx*(N-n+1))
%   s       Ancestor states s_{n-1} (Ns*M)
%   z       Mean of linear states at n-1 (Nz*M)
%   P       Covariance of linear states at n-1 (Nz*Nz)
%   t       Time from t[n] to t[N] (1*(N-n+1))
%   lw      Log of weights log(w[n-1]) (1*M)
%   model   The model struct
%   Nbar    Horizon of future states to consider
%
% RETURNS
%   lv      Log of ancestor weights log(v[n])

    %% Initialize
    Nbar = min(Nbar, size(sbar, 2));
    M = size(s, 2);
    lv = lw;                    % Initialize ancestor weights with prior
    
    %% First Step
    % These are zbar[n-1], Pbar[n-1]
    zbar = z;
    Pbar = P;
    
    % For convenience
    g = model.fn(s, t(1));
    G = model.Fn(s, t(1));
    h = model.fl(s, t(1));
    H = model.Fl(s, t(1));
    Q = model.Q(s, t(1));
    Qs = Q(model.in, model.in);
    Qz = Q(model.il, model.il);
    Qzs = Q(model.il, model.in);
    
    % Calculate marginalized prediction p(\bar{s}_n | s_{1:n-1}, y_{1:n-1})
    m = g + G*zbar;
    C = Qs + G*Pbar*G';
    lv = lv + logmvnpdf((sbar(:, 1)*ones(1, M)).', m.', C).';
    
    %% KF Update
    % Calculate p(z_n | \bar{s}_n, s_{1:n-1}, y_{1:n-1})
    S = Qs + G*Pbar*G';
    K = (Qzs + H*Pbar*G')/S;
    zbar = h + H*zbar + K*(sbar(:, 1)*ones(1, M) - g - G*zbar);
    Pbar = Qz + H*Pbar*H' - K*S*K';
    
    %% Update for j > n
    for j = 2:Nbar
        % For convenience
        g = model.fn(sbar(:, j-1), t(j))*ones(1, M);
        G = model.Fn(sbar(:, j-1), t(j));
        h = model.fl(sbar(:, j-1), t(j))*ones(1, M);
        H = model.Fl(sbar(:, j-1), t(j));
        Q = model.Q(sbar(:, j-1), t(j));
        Qs = Q(model.in, model.in);
        Qz = Q(model.il, model.il);
        Qzs = Q(model.il, model.in);
        
        % Calculate marginal prediction
        % p(s_j | \bar{s}_{n:j-1}, s_{1:n-1}, y_{1:n-1})
        m = g + G*zbar;
        C = Qs + G*Pbar*G';
        lv = lv + logmvnpdf((sbar(:, j)*ones(1, M)).', m.', C.').';
        
        %% KF Update
        % Calculate p(z_j | \bar{s}_{n:j-1}, s_{1:n-1}, y_{1:n-1})
        S = Qs + G*Pbar*G';
        K = (Qzs + H*Pbar*G')/S;
        zbar = h + H*zbar + K*(sbar(:, j)*ones(1, M) - g - G*zbar);
        Pbar = Qz + H*Pbar*H' - K*S*K';
    end
end

%% Kalman Filter Update
% 
function [zp, Pp] = kf_update(z, P, sp, s, t, model)
% Kalman filter update. Note that this is essentially only a prediction
% since the likelihood doesn't depend on z[n].
%
% PARAMETERS
%   z       State z[n-1]
%   P       Covariance P[n-1]
%   sp      New samples of the non-linear states, s[n]
%   s       Old samples of the non-linear states, s[n-1]
%   t       Time
%   model   The model struct
%
% RETURNS
%   zp      Updated state z[n]
%   Pp      Updated covariance P[n]

    % Get dynamics; none of Q, G, and H depend on s[n-1]; hence, we can
    % calculate everything efficiently. Nothing depends on 
    g = model.fn(s, t);
    G = model.Fn(s, t);
    Qs = model.Qn(s, t);
    h = model.fl(s, t);
    H = model.Fl(s, t);
    Qz = model.Ql(s, t);
    Qzs = model.Qnl(s, t)';

    % Prediction
    S = Qs + G*P*G';
    K = (Qzs + H*P*G')/S;
    zp = h + H*z + K*(sp - g - G*z);
    Pp = Qz + H*P*H' - K*S*K';
    Pp = (Pp+Pp')/2;
end
