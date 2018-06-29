function [x, sys] = rbcpfas2(y, t, model, xt, M, par)
% Rao-Blackwellized conditional particle filter for mixed CLGSS models
%
% SYNOPSIS
%   x = rb_cpfas(y, t, model)
%   [x, sys] = rb_cpfas(y, t, model, xt, M, par)
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
%   xt      Seed trajectory (Nx*N+1); calculated using a regular PF if 
%           omitted.
%
%   M       No. of particles to use in the particle filter (optional, 
%           default: 100)
%
%   par     Struct containing additional parameters:
%
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
%   [1] R. Hostettler, S. Sarkka, S. J. Godsill, "Rao--Blackwellized
%       particle MCMC for parameter estimation in spatio-temporal Gaussian
%       processes," 2017, to appear.
%
% AUTHORS
%   2017-05-24 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Update the references in the documentation (actually, update the
%     documentation entirely).
%   * The initialize function should probably be replaced by a RBPF or
%     something.
%   * Should we return the smoothed covariance as well?
%   * Not sure if we actually need sys.
%   * Simplify the different linear parts in accordance with the new
%     article.
%   * Rewrite. Make such that it only returns 's' (also update the
%     notation), and both z and P are returned in sys.
%   * How about calculating the lineage if we have particle-dependent P? I
%     think the most straight-forward notation will be to include a
%     varargin to calculate_particle_lineage() that takes a list of fields
%     which have to be considered.
%   * Assumes bootstrap proposal
%   * Handling of the linear states is hacky at best
%   * Add possiblity to add sampling function (importance density; see cpfas)
%   * Replace initialize() with a rbsisr or something similar.
%   * Check how much it has diverged from the original function and
%     re-merge. Should be just fine.

% NOTES
%   * Assumes implicitly that G, H, and Q don't depend on s[n-1] which
%     helps speeding up the calculation (parallelizable). The affected
%     functions are 'kf_predict', 'calculate_ancesotor_weights', and
%     'draw_samples'.

    %% Defaults
    narginchk(3, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    
    [Ny, N] = size(y);
    if isempty(t)
        t = 1:N;
    end
    def = struct( ...
        'Nbar', N ...  % Truncation length for calculating the ancestor weigths; 
    );
    par = parchk(par, def);
    
    % Prepend t[0] and pseudo (no) measurement
    t = [0, t];
    y = [NaN*ones(Ny, 1), y];
    
    %% Preallocate
    in = model.in;
    il = model.il;
    
    % State dimensions and data length
    Ns = length(in);
    Nz = length(il);
    Nx = Ns+Nz;
    N = N+1;                            % +1 to include x[0]
    xp = zeros(Nx, M);
        
    %% Seed Trajectory
    if nargin < 4 || isempty(xt) || sum(sum(xt)) == 0
        xt = initialize(y, t, model, M);
    end
    sbar = xt(in, :);
        
    %% Prepare Particle System
    % TODO: Use initialize_sys, store in x, w, and alpha, use xf for full
    % trajectories (see calculate_particle_lineages())
    sys = initialize_sys(N, Nx, M);
    sys(N).P = zeros(Nx, Nx, M);
    
    %% Initialize
    % Calculate backward statistics for ancestor weights
    [alphas, betas, Gammas] = calculate_backward_statistics2(sbar, t, model);
    
    % Initial samples
    m0 = model.m0;
    P0 = model.P0;
    s = m0(in)*ones(1, M) + chol(P0(in, in)).'*randn(Ns, M);
    s(:, M) = sbar(:, 1);
    mz = m0(il)*ones(1, M) + P0(il, in)/P0(in, in)*(s - m0(in)*ones(1, M));
    Pz = P0(il, il) - P0(il, in)/P0(in, in)*P0(in, il);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    % Store initial
    sys(1).w = w;
    sys(1).x(in, :) = s;
    sys(1).x(il, :) = mz;
    sys(1).P = Pz;
    
    %% Process Data
    for n = 2:N        
        % Draw new particles
        alpha = sysresample(w);
        sp = draw_samples(s(:, alpha), mz(:, alpha), Pz, t(n), model);
        
        % Set Mth particle
        sp(:, M) = sbar(:, n);
        
        % Calculate ancestor weights and draw an index
%         lv = calculate_ancestor_weights(sbar(:, n:N), s, mz, Pz, t(n:N), lw, model, par.Nbar);
        lv = calculate_ancestor_weights2(alphas(n), betas(:, n), Gammas(:, :, n), sbar(:, n), s, mz, Pz, t(n), lw, model);
        v = exp(lv-max(lv));
        v = v/sum(v);
        if any(isnan(v)) || any(v == Inf)
            warning('NaN and/or Inf in ancestor weights.');
        end
        tmp = sysresample(v);
        alpha(M) = tmp(randi(M, 1));
        
        % KF update
        [mzp, Pzp] = kf_predict(mz(:, alpha), Pz, sp, s(:, alpha), t(n), model);
                
        % Particle weights
        xp(in, :) = sp;     % TODO: Same workaround here as in the initialization
        xp(il, :) = mzp;
        lw = calculate_incremental_weights_bootstrap(y(:, n), sp, s(:, alpha), t(n), model, []);

        % Normalize weights and update particles
        s = sp;
        mz = mzp;
        Pz = Pzp;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        if any(isnan(w)) || any(w == Inf)
            warning('NaN and/or Inf in particle weights.');
        end
        
        %% Store New Samples & Ancestor Indices
        sys(n).x(in, :) = s;
        sys(n).x(il, :) = mz;
        sys(n).w = w;
        sys(n).alpha = alpha;
        sys(n).P = Pz;
    end
    
    %% Draw Trajectory    
    % Draw a trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    sys = calculate_particle_lineages(sys, j);
    x = cat(2, sys.xf);
    Pz = cat(3, sys.P);
        
    % Smooth Linear States
    [x(il, :), Ps] = rts_smooth(x, Pz, t, model);
    % TODO: This is hacky!
    for n = 1:N
        sys(n).xf(il, :) = x(il, n);
        sys(n).Ps = Ps(:, :, n);
    end
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
        
    % Stores the full trajectories
    alphaf = zeros(1, M, N);    % Ancestor indices
    xf = zeros(Nx, M, N);       % Trajectories
    xp = zeros(Nx, M);
    
    %% Initialize
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
        [zp, Pp] = kf_predict(z(:, alpha), P, sp, s(:, alpha), t(n), model);
                
        % Particle weights
        xp(in, :) = sp;    % TODO: This is a work-around to make py.logpdf work; not really generic!
        xp(il, :) = zp;
        lw = calculate_incremental_weights_bootstrap(y(:, n), sp, s(:, alpha), t(n), model, []);

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
function [mzs, Pzs] = rts_smooth(x, Pz, t, model)
% Smoothing of linear states conditional on a complete trajectory of
% non-linear states.
% 
% PARAMETERS
%   x       Complete filtered state trajectory (including non-linear and 
%           linear states)
%   Pz      Covariances of the linear states
%   t       Time vector
%   model   Model struct
%
% RETURNS
%   zs      Smoothed trajectory of linear states
%   Ps      Smoothed covarainces

    %% Initialize
    % Get nonlinear and linear states
    % TODO: This should not be done here.
    s = x(model.in, :);
    mz = x(model.il, :);
    
    % Preallocate
    [Nz, N] = size(mz);
    Ns = size(s, 1);
    mzs = zeros(Nz, N);
    Pzs = zeros(Nz, Nz, N);
    
    % Initialize backward pass
    mzs(:, N) = mz(:, N);
    Pzs(:, :, N) = Pz(:, :, N);
    
    %% Backward Iteration
    for n = N-1:-1:1
        % Prediction from n to n+1
        g = model.fn(s(:, n), t(n+1));
        G = model.Fn(s(:, n), t(n+1));
        h = model.fl(s(:, n), t(n+1));
        H = model.Fl(s(:, n), t(n+1));
        Q = model.Q(s(:, n), t(n+1));
        mp = [g; h] + [G; H]*mz(:, n);           % m[n+1 | n]
        Pp = [G; H]*Pz(:, :, n)*[G; H]' + Q;     % P[n+1 | n]
        Pp = (Pp + Pp')/2;
        Pc = Pz(:, :, n)*[G; H]';                % P[n,n+1 | n]
        
        % RTS update
        M = Pc/Pp;
        mzs(:, n) = mz(:, n) + M*([s(:, n+1); mzs(:, n+1)] - mp);
        Pdash = [
            zeros(Ns, Ns), zeros(Ns, Nz); 
            zeros(Nz, Ns), Pzs(:, :, n+1);
        ];
        Pzs(:, :, n) = Pz(:, :, n) + M*(Pdash - Pp)*M';
        Pzs(:, :, n) = (Pzs(:, :, n) + Pzs(:, :, n)')/2;
    end
end

%% Sampling
function sp = draw_samples(s, mz, Pz, t, model)
% Draws new samples from the marginalized nonlinear dynamics.
%
% PARAMETERS
%   s       Old samples, s[n-1]
%   mz      Mean of conditionally linear states at n-1, z[n-1]
%   Pz      Covariance of contidionally linear states at n-1, P[n-1]
%   t       Time t[n]
%   model   Model struct
%
% RETURNS
%   sp      New samples, s[n]

    % Get dynamics
    g = model.fn(s, t);
    G = model.Fn(s, t);
    Qs = model.Qn(s, t);
    
    % Marginalization
    msp = g + G*mz;
    Psp = Qs + G*Pz*G';
    Psp = (Psp+Psp')/2;
    
    % Generate samples
    LPsp = chol(Psp).';
    sp = msp + LPsp*randn(size(s));
end

%% 
function [alpha, beta, Gamma] = calculate_backward_statistics2(sbar, t, model)
    N = length(sbar);
    Nz = length(model.il);

    alpha = zeros(1, N);
    alpha(N) = NaN;
    beta = zeros(Nz, N);
    Gamma = zeros(Nz, Nz, N);
    
    % Initialization of the backward recursion
    s = sbar(:, N-1);      % s[n-1]
    sp = sbar(:, N);        % s[n]
    g = model.fn(s, t(N));
    G = model.Fn(s, t(N));
    Qs = model.Qn(s, t(N));
    alpha(N-1) = (sp - g)'/Qs*(sp - g);
    beta(:, N-1) = G'/Qs*(sp - g);
if 0
    Gamma(:, :, N-1) = (G'/Qs*G)\eye(Nz);
else
    Gamma(:, :, N-1) = G'/Qs*G;
end
    
    % Backward recursion
    for i = N-2:-1:1
        % Note on storage: We store alpha[n+1] as denoted in the article in
        % the alpha(n). This because we need alpha[n+1] for caclulating the
        % ancestor weights of s[n] and this thus allows for easier storage
        % management (instead of having empty entries at the beginning and
        % end).
        [alpha(i), beta(:, i), Gamma(:, :, i)] = bf_update2(alpha(i+1), beta(:, i+1), Gamma(:, :, i+1), sbar(:, i+1), sbar(:, i), t(i+1), model);
    end
end

%% Backward filter update
% TODO: Implement more efficiently and robust
function [alpha, beta, Gamma] = bf_update2(alpha, beta, Gamma, sp, s, t, model)
    M = size(s, 2);
    sp = sp*ones(1, M);
    
    Nz = length(model.il);
    g = model.fn(s, t);
    G = model.Fn(s, t);
    Qs = model.Qn(s, t);
    h = model.fl(s, t);
    H = model.Fl(s, t);
    Qz = model.Ql(s, t);
    Qsz = model.Qnl(s, t);
    
    L = Qsz'/Qs;
    htilde = h + L*(sp - g);
    Htilde = H - L*G;
    Qztilde = Qz - L*Qs*L';

if 0
    % (sp-g)'/Qs*(sp-g)
    LQs = chol(Qs).';
    epsilon1 = sum((LQs\(sp-g)).^2, 1);
    
    % (htilde - 2*Gamma*beta)'/(Gamma+Qztilde)*htilde
    epsilon2 = sum(((Gamma+Qztilde)\(htilde - 2*Gamma*beta*ones(1, M))).*htilde, 1);
    
    alpha = ( ...
        alpha + epsilon1 + epsilon2 ...
        - beta'*Gamma/(Gamma+Qztilde)*Qztilde*beta ...
    );

    beta = Htilde'/(Gamma+Qztilde)*(Gamma*beta*ones(1, M) - htilde) + G'/Qs*(sp-g);
    
    Gamma = (G'/Qs*G + Htilde'/(Gamma+Qztilde)*Htilde)\eye(Nz);
        
else
    Gammatilde = Gamma + eye(Nz)/Qztilde;
    
    Ls = chol(Qs).';
    epsilon1 = sum((Ls\(sp - g)).^2, 1);
    Lhtilde = chol((eye(Nz)/Gamma + Qztilde)).';
    epsilon2 = sum((Lhtilde\htilde).^2, 1);
    LGammatilde = chol(Gammatilde).';
    epsilon3 = sum((LGammatilde\beta).^2, 1);
    alpha = ( ...
        alpha + epsilon1 + epsilon2 - epsilon3 ...
        - 2*beta'/Gammatilde/Qztilde*htilde ...
    );

    beta = ( ...
        G'/Qs*(sp-g) + Htilde'/Qztilde/Gammatilde*beta*ones(1, M) ...
        - Htilde'/(eye(Nz)/Gamma + Qztilde)*htilde ...
    );

    Gamma = G'/Qs*G + Htilde'/(eye(Nz)/Gamma + Qztilde)*Htilde;
end
end

function lv = calculate_ancestor_weights2(alpha, beta, Gamma, sp, s, mz, Pz, t, lw, model)
    if isnan(alpha)
        g = model.fn(s, t);
        G = model.Fn(s, t);
        Qs = model.Qn(s, t);
        
        mp = g + G*mz;
        Pp = G*Pz*G' + Qs;
        lv = lw + logmvnpdf(sp.', mp.', Pp.').';
    else
        [alpha, beta, Gamma] = bf_update2(alpha, beta, Gamma, sp, s, t, model);
        
if 0
        % NOTE: These branches produce the same result, verified.
        % beta'*Gamma/(Gamma + Pz)*(Pz*beta + 2*mz)
        epsilon1 = sum(((Gamma + Pz)\Gamma*beta).*(Pz*beta + 2*mz), 1);
        
        % 2*beta'*Gamma/(Gamma + Pz)*mz
%         epsilon2 = 2*sum(((Gamma + Pz)\Gamma*beta).*mz, 1);
        
        % mz'/(Gamma + Pz)*mz
        LGammaPz = chol(Gamma + Pz).';
        epsilon3 = sum((LGammaPz\mz).^2, 1);
        
        lv = lw - 1/2*(alpha - epsilon1 + epsilon3);
        
else
        Nz = size(beta, 1);
        Sigma = Gamma + eye(Nz)/Pz;

        Lz = chol(eye(Nz)/Gamma + Pz).';
        epsilon1 = sum((Lz\mz).^2, 1);
        LSigma = chol(Sigma).';
        epsilon2 = sum((LSigma\beta).^2, 1);
        epsilon3 = 2*sum((LSigma\beta).*(Lz\mz), 1);
        lv = lw - 1/2*(alpha + epsilon1 - epsilon2 - epsilon3);
end
    end
end

%%
%% Ancestor Weights
function lv = calculate_ancestor_weights(sbar, s, mz, Pz, t, lw, model, Nbar)
% Calculates the ancestor weights for the seed trajectory.
%
% PARAMETERS
%   sbar    Seed trajectory from n to N, i.e. bar{s}_{n:N} (Nx*(N-n+1))
%   s       Ancestor states s_{n-1} (Ns*M)
%   mz      Mean of linear states at n-1 (Nz*M)
%   Pz      Covariance of linear states at n-1 (Nz*Nz)
%   t       Time from t[n] to t[N] (1*(N-n+1))
%   lw      Log of weights log(w[n-1]) (1*M)
%   model   The model struct
%   Nbar    Horizon of future states to consider
%
% RETURNS
%   lv      Log of ancestor weights log(v[n])

    %% Initialize
    Nbar = min(Nbar, size(sbar, 2));
    [Ns, M] = size(s);
    Nz = size(mz, 1);
    in = 1:Ns;
    il = Ns+(1:Nz);
    lv = lw;                    % Initialize ancestor weights with prior
    
    %% Recursively calculate weight
    for j = 1:Nbar
        %% Prediction
        g = model.fn(s, t(j));
        G = model.Fn(s, t(j));
        h = model.fl(s, t(j));
        H = model.Fl(s, t(j));
        Q = model.Q(s, t(j));
        
        % Prediction
        mp = [g; h] + [G; H]*mz;
        Pp = [G; H]*Pz*[G; H]' + Q;
        Pp = (Pp+Pp')/2;
        
        % Calculate marginal prediction
        % p(s_j | \bar{s}_{n:j-1}, s_{1:n-1}, y_{1:n-1})
        sp = sbar(:, j)*ones(1, M);
        lv = lv + logmvnpdf(sp.', mp(in, :).', Pp(in, in).').';
        
        %% KF Update
        % Calculate p(z_j | \bar{s}_{n:j-1}, s_{1:n-1}, y_{1:n-1})
        K = Pp(il, in)/Pp(in, in);
        mz = mp(il, :) + K*(sp - mp(in, :));
        Pz = Pp(il, il) - K*Pp(in, in)*K';
        Pz = (Pz + Pz')/2;
        
        % Next iteration's s is this iteration's sp
        s = sp;
    end
end

%% Kalman Filter Update
function [mzp, Pzp] = kf_predict(mz, Pz, sp, s, t, model)
% Kalman filter prediction and pseudo update
%
% PARAMETERS
%   mz      Mean m[n-1 | n-1]
%   Pz      Covariance P[n-1 | n-1]
%   sp      New samples of the non-linear states, s[n]
%   s       Old samples of the non-linear states, s[n-1]
%   t       Time
%   model   The model struct
%
% RETURNS
%   mzp     Updated state m[n | n-1]
%   Pzp     Updated covariance P[n | n-1]

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
    S = Qs + G*Pz*G';
    K = (Qzs + H*Pz*G')/S;
    mzp = h + H*mz + K*(sp - g - G*mz);
    Pzp = Qz + H*Pz*H' - K*S*K';
    Pzp = (Pzp+Pzp')/2;
end
