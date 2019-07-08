function [xhat, sys] = rbpf_mixed(y, t, model, M, par)
% Rao-Blackwellized particle filter for mixed CLGSS models
%
% USAGE
%   xhat = RBPF_MIXED(y, t, model)
%   [xhat, sys] = RBPF_MIXED(y, t, model, M, par)
% 
% DESCRIPTION
%   Rao-Blackwellized particle filter for mixed conditionally linear
%   Gaussian state space models of the form
%
%       s[n] = f(s[n-1]) + F(s[n-1])*z[n-1] + qs[n],
%       z[n] = g(s[n-1]) + G(s[n-1])*z[n-1] + qz[n],
%       y[n] = h(s[n-1]) + H(s[n-1])*z[n-1] + r[n],
%
%   where s[n] is the nonlinear state and z[n] is the linear state.
%
%   The algorithm implemented here is the most generic version and 
%   corresponds to the algorithm for Model 3 in [1].
% 
% PARAMETERS
%
% 
%
% RETURNS
%   xhat    
%
%
% REFERENCES
%   [1] T. B. Schon, F. Gustafsson, P.-J. Nordlund, "Marginalized Particle
%       Filters for Mixed Linear/Nonlinear State-Space Models", IEEE
%       Transactions on Signal Processing", vol. 53, no. 7, July 2005
%
% SEE ALSO
%   
%
% VERSION
%   2017-01-17
% 
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% Todo:
% * Document the convention used here: x corresponds to s, we have the
%   additonal fields m and P
    

    %% Defaults
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5 || isemtpy(par)
        par = struct();
    end
    defaults = struct( ...
        'resampel', @resample_ess, ...
        'sample', @sample_bootstrap, ...
        'calculate_incremental_weights', @calculate_incremental_weights_bootstrap ...
    );
    par = parchk(par, defaults);
    
    %% Prepare
    % Data dimensions, prepend NaN for zero measurement
    [N, Ny] = size(y, 2);
    t = [0, t];
    y = [NaN*ones(Ny, 1), y];
    N = N+1;
    
    % Linear and nonlinear state indices and dimensions
    in = model.in;
    il = model.il;
    Nn = length(in);
    Nl = length(il);
    Nx = Nn+Nl;
    
    % sys structure
    if nargout >= 2
        return_sys = true;
        sys = initialize_sys(N, Nn, M);
    end
    
    %% Initialize
    % Draw initial particles and calculate mean and covariance
    m0 = model.m0;
    P0 = model.P0;
    s = m0(in)*ones(1, N) + chol(P0(in, in)).'*randn(Nn, N);
    lw = log(1/N)*ones(1, N);
    mz = m0(il)*ones(1, N) + P0(il, in)/P0(in, in)*(s - m0(in)*ones(1, N));
    Pz = P0(il, il) - P0(il, in)/P0(in, in)*P0(in, il);

    % Store
    if return_sys
        sys(1).x = s;
        sys(1).w = exp(lw);
        sys(1).m = mz;
        sys(1).P = Pz;
    end
    
    %% Preallocate
    xhat = zeros(Nx, N-1);
    sp = zeros(size(s));
    mzp = zeros(size(mz));
    Pzp = zeros(size(Pz));
    lv = zeros(1, M);

    %% 
    for n = 2:N
        %% Resample
        [alpha, lw, r] = par.resample(lw);
        s = s(:, alpha);
        mz = mz(:, alpha);
        Pz = Pz(:, :, alpha);
        
        %% Propagate
        for m = 1:M            
            % Get vectors/matrices for compactness
            f = model.f(s(:, m), t(n));
            F = model.F(s(:, m), t(n));
            Q = model.Q(s(:, m), t(n));
            
            fn = f(in);
            Fn = F(in, :);
            Gn = eye(Nn);
            Qn = Q(in, in);

            fl = f(il);
            Fl = F(il, :);
            Gl = eye(Nl);
            Ql = Q(il, il);
            
            Qnl = Q(in, il);

            %% Time update
            % Draw samples
            sp = par.sample(y(:, n), s(:, m), mz(:, m), Pz(:, :, m), t(n), model);

            % KF prediction
            Flbar = Fl - Gl*Qnl'/(Gn*Qn)*Fn;
            Qlbar = Ql - Qnl'/Qn*Qnl;
            
            Nt = Fn*Pz(:, :, m)*Fn' + Gn*Qn*Gn';
            Lt = Flbar*Pz(:, :, m)*Fn'/Nt;
            mzp(:, m) = ( ...
                Flbar*mz(:, m) + Gl*Qnl'/(Gn*Qn)*(sp(:, m) - fn) ...
                + fl + Lt*(sp(:, m) - fn - Fn*mz(:, m)) ...
            );
            Pzp(:, :, m) = ( ...
                Flbar*Pz(:, :, m)*Flbar' + Gl*Qlbar*Gl' - Lt*Nt*Lt' ...
            );

            %% Measurement update
            % KF update
            h = model.h(sp(:, m), t(n));
            H = model.H(sp(:, m), t(n));
            R = model.R(sp(:, m), t(n));

            Mn = H*Pzp(:, :, m)*H' + R;
            Kn = Pzp(:, :, m)*H'/Mn;
            mz(:, m) = mzp(:, m) + Kn*(y - h - H*mzp(:, m));
            Pz(:, :, m) = Pzp(:, :, m) - Kn*Mn*Kn';

            % Weights
            lv(:, m) = par.calculate_incremental_weights(y, sp, mzp, Pzp, t(n), model);
        end

        %% Point estimate
        % Normalize the weights
        lw = lw + lv;
        w = exp(lw-max(lw));
        s = sp;
        w = w/sum(w);
        lw = log(w);

        % MMSE
        xhat(in, n-1) = s*w';
        xhat(il, n-1) = mz*w';        
        
        % Store
        if return_sys
            sys(n).x = s;
            sys(n).w = w;
            sys(n).mz = mz;
            sys(n).Pz = Pz;
            sys(n).alpha = alpha;
            sys(n).r = r;
        end
    end
    
    %%
if 0
    % TODO: Calculate particle lineages
    alpha = 1:M;
    xf(:, :, N) = xf(:, :, N);
    for n = N-1:-1:1
        xf(:, :, n) = xf(:, alphas(:, alpha, n+1), n);
        alpha = alphas(:, alpha, n+1);
    end
end
end

%% Bootstrap sampling function
% Overwrites global bootstrap sampling function
function sp = sample_bootstrap(~, s, mz, Pz, t, model)
    in = model.in;
    f = model.f(s, t);
    F = model.F(s, t);
    Q = model.Q(s, t);
    
    fn = f(in);
    Fn = F(in, :);
    Gn = Is;
    Qn = Q(in, in);

    mu = fn + Fn*mz;
    Sigma = Fn*Pz*Fn' + Gn*Qn*Gn';
    sp = mu + chol(Sigma).'*randn(size(s, 1), 1);
end

%% Bootstrap weighing function
% Overwrites global bootstrap weighing function
function lv = calculate_incremental_weights_bootstrap(y, sp, mzp, Pzp, t, model)
    h = model.h(sp, t);
    H = model.C(sp, t);
    R = model.R(sp, t);

    mu = h + H*mzp;
    Sigma = H*Pzp*H' + R;
    lv = logmvnpdf(y.', mu.', Sigma);
end
