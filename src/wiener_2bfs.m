function [xhat, sys] = wiener_2bfs(y, t, model, M, par)
% Two-Filter Bootstrap Particle Smoother for Wiener state-space models
% 
% USAGE
%   xhat = WIENER_2BFS(y, t, model)
%   [xhat, sys] = WIENER_2BFS(y, t, model, M, par, sys)
%
% DESCRIPTION
%   
%
% PARAMETERS
% 
%
% RETURNS
%
% 
% AUTHORS
%   2018-05-18 -- Roland Hostettler <roland.hostettler@aalto.fi>   
   
    %% Defaults & Checks
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = [];
    end
    def = struct();
    par = parchk(par, def);
    
    %% Process data
    t = [0, t];
    y = [NaN*ones(size(y, 1), 1), y];
    sys = filter(y, t, model, M, par);
    [xhat, sys] = smooth(y, t, model, M, par, sys);
end

%% Forward filter
function sys = filter(y, t, model, M, par)
    %% Bootstrap proposal
    q = struct();
    q.fast = model.px.fast;
    q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
    q.rand = @(y, x, t) model.px.rand(x, t);
    
    %% Initialize
    x = model.px0.rand(M);
    lw = log(1/M)*ones(1, M);
    
    %% Preallocate
    Nx = size(x, 1);
    N = length(t);
    sys = initialize_sys(N, Nx, M);
    sys(1).x = x;
    sys(1).w = exp(lw);
    sys(1).alpha = 1:M;
    sys(1).r = false;
    sys(1).mu = model.m0;
    sys(1).Sigma = model.P0;
    
    %% Process data
    for n = 2:N
        % Resample
        [alpha, lw, r] = resample_ess(lw, par);
        
        % Draw new samples
        xp = sample_q(y(:, n), x(:, alpha), t(n), q);
        
        % Calculate weights
        lv = calculate_incremental_weights_bootstrap(y(:, n), xp, x, t(n), model, q);
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        x = xp;
        
        % Calculate prior
        F = model.F(t(n));
        Q = model.Q(t(n));
        sys(n).mu = F*sys(n-1).mu;
        sys(n).Sigma = F*sys(n-1).Sigma*F' + Q;
        
        % Store
        sys(n).x = x;
        sys(n).w = w;
        sys(n).alpha = alpha;
        sys(n).r = r;
    end
end

%% Backward filter and smoothing
function [xhat, Phat, sys] = smooth(y, t, model, M, par, sys, mu_x, Sigma_x)
    py = model.py;

    %% Preallocate
    [Nx, M, N] = size(sys.x);
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    if nargout == 3
        sys.xs = zeros(Nx, M, N);
        sys.wb = zeros(1, M, N);
        sys.lwb = zeros(1, M, N);
        sys.ws = zeros(1, M, N);
        sys.lws = zeros(1, M, N);
    end
    
    % Recurring temporary variables
    % TODO: sort out
    Ny = size(y, 1);
    mu_xb = zeros(Nx, M);
    mu_yb = zeros(Ny, M);
    B = zeros(Nx, Ny, M);
    S = zeros(Ny, Ny, M);
    
    lv = zeros(1, M);

    %% Initialize Backward Filter
    for m = 1:M
        lv(:, m) = logmvnpdf(sys.x(:, m, N).', mu_x(:, N).', Sigma_x(:, :, N)).' ...
            + py.logpdf(y(:, N), sys.x(:, m, N), t(N)) - sys.lw(:, m, N);
    end
    v = exp(lv-max(lv));
    v = v/sum(v); 
    beta = sysresample(v);
    xs = sys.x(:, beta, N);
    lwb = log(1/M)*ones(1, M);
    wb = 1/M*ones(1, M);
    lws = log(1/M)*ones(1, M);
    ws = 1/M*ones(1, M);
    xhat(:, N) = mean(xs, 2);
    
    % Store the particle system
    if nargout == 3
        sys.xs(:, :, N) = xs;
        sys.lwb(:, :, N) = lwb;
        sys.wb(:, :, N) = wb;
        sys.lws(:, :, N) = lws;
        sys.ws(:, :, N) = ws;
    end

    %% Backward-Filter & Smoothing
    for n = N-1:-1:1
        %% Backward Filter
        % Calcualte backward dynamics
        F = model.F(t);
        Q = model.Q(t);
        L = F*Sigma_x(:, :, n)*F' + Q;
        K = Sigma_x(:, :, n)*F'/L;
        mu_xb = mu_x(:, n)*ones(1, M) + K*(xs - (F*mu_x(:, n))*ones(1, M));
        Sigma_xb = Sigma_x(:, :, n) - K*L*K';
        
        % Decompose 
        Nq = rank(Sigma_xb);
        [V, D] = eig(Sigma_xb);
        D = diag(D);
        D(abs(D)/max(abs(D)) < 1e-6) = 0; % To avoid numerical problems, discard eigenvalues that are smaller epsilon*max(abs(D))
        [~, indices] = sort(D, 'descend');
        Phi = diag(D(indices(1:Nq)));
        Gamma = V(:, indices(1:Nq));
        
        % Draw new samples
        xs = mu_xb + Gamma*sqrt(Phi)*randn(Nq, M);
        
        % Weigh
        if py.fast
            lwb = lwb + py.logpdf(y(:, n)*ones(1, M), xs, t(n));
        else
            for m = 1:M
                lwb(m) = lwb(m) + py.logpdf(y(:, n), xs(:, m), t(n));
            end
        end
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);
        lwb = log(wb);
        
        if max(abs(imag(wb))) ~= 0
            aaa = 1;
        end

        %% Smoothing
if 0
        if n > 1
            x = sys.x(:, :, n-1);
            w = sys.w(:, :, n-1);
        else
            x = sys.x0;
            w = 1/M*ones(1, M);
        end
        nu = zeros(1, M);
        for m = 1:M
            nu(m) = w*mvnpdf(xs(:, m).', (F*x).', Q);
%             tmp = lw + logmvnpdf(xs(:, m).', (F*x).', Q).';
%             lnu(m) = log(sum(exp(tmp)));
        end
        lws = lwb - logmvnpdf(xs.', mu_x(:, n).', Sigma_x(:, :, n)).' + log(nu);
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
        lws = log(ws);
        xhat(:, n) = xs*ws.';
end

        %% Store
        if nargout == 3
            sys.xs(:, :, n) = xs;
            sys.lwb(:, :, n) = lwb;
            sys.wb(:, :, n) = wb;
            sys.lws(:, :, n) = lws;
            sys.ws(:, :, n) = ws;
        end
        
        %% Resample Backward Filter
        % TODO: We only need to resample when ESS < Threshold (or similar)
        ir = sysresample(wb);
        xs = xs(:, ir);
        wb = 1/M*ones(1, M);
        lwb = log(1/M)*ones(1, M);
    end
end
