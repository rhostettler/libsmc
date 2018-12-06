function [xhat, sys] = wiener_2bfs(y, t, model, M, par)
% Two-Filter Bootstrap Particle Smoother for Wiener state-space models
% 
% USAGE
%   xhat = WIENER_2BFS(y, t, model)
%   [xhat, sys] = WIENER_2BFS(y, t, model, M, par, sys)
%
% DESCRIPTION
%   Bootstrap particle filter based two filter smoother for Wiener
%   state-space systems as discussed in [1].
%
% PARAMETERS
%   y       Ny*N matrix of measurement data.
%   t       1*N vector of timestamps.
%   model   Wiener state-space model structure.
%   M       Number of particles (default: 100).
%   par     Additional parameters:
%
%               Mt  Resampling threshold (default: M/3).
%
% RETURNS
%   xhat    Smoothed state estimate.
%   sys     Particle system structure (array of structs).
%
% REFERENCES
%   [1] R. Hostettler, "A two filter particle smoother for Wiener state-
%       space systems," in IEEE Conference on Control Applications (CCA),
%       Sydney, Australia, September 2015.
% 
% AUTHORS
%   2018-05-18 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Smoothing of initial state is missing (needs special attention in the
%     smoothed weight section).
%   * There seems to be a bug in the backward filter initialization; too
%     large error there; it seems that this might be related to the
%     ancestor indices, need to check that.
   
    %% Defaults & Checks
    narginchk(3, 5);
    if nargin < 4 || isempty(M)
        M = 100;
    end
    if nargin < 5
        par = [];
    end
    def = struct( ...
        'Mt', M/3 ...   % Resampling threshold
    );
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
function [xhat, sys] = smooth(y, t, model, M, par, sys)
    %% Initialization
    Nx = size(model.m0, 1);
    N = size(y, 2);
    x = sys(N).x;
    lwb = ( ...
        log(sys(N).w) + logmvnpdf(x.', sys(N).mu.', sys(N).Sigma.').' ...
        - log(sys(N-1).w) - logmvnpdf(x.', (model.F(t(N))*sys(N-1).x).', model.Q(t(N)).').' ...
    );
    wb = exp(lwb-max(lwb));
    wb = wb/sum(wb);
    lwb = log(wb);
    
    % Calculate the smoothed weights
    s = zeros(1, M);
    for m = 1:M
        s = s + sys(N-1).w(:, m)*mvnpdf(x.', (model.F(t(N))*sys(N-1).x(:, m)).', model.Q(t(N)).').';
    end
    lws = lwb + log(s) - logmvnpdf(x.', sys(N).mu.', sys(N).Sigma.').';
    ws = exp(lws-max(lws));
    ws = ws/sum(ws);
    
    % Store
    sys(N).xs = x;
    sys(N).ws = ws;
    sys(N).wb = wb;
    sys(N).r = false;

    % Point estimate
    xhat = zeros(Nx, N);
    xhat(:, N) = x*ws';

    %% Backward recursion
    for n = N-1:-1:2
        % Resample
        [alpha, lwb, r] = resample_ess(lwb, par);
        
        % Draw new particles
        F = model.F(t(n+1));
        K = sys(n).Sigma*F'/sys(n+1).Sigma;
        mu = sys(n).mu*ones(1, M) + K*(x(:, alpha) - sys(n+1).mu*ones(1, M));
        Sigma = sys(n).Sigma - K*sys(n+1).Sigma*K';
        Sigma = (Sigma+Sigma')/2;
        xp = mu + chol(Sigma).'*randn(Nx, M);
        
        % Calculate backward filter weights
        lv = calculate_incremental_weights_bootstrap(y(:, n), xp, [], t(n), model, []);
        lwb = lwb+lv;
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);
        lwb = log(wb);
        x = xp;

        % Calculate smoothed weights
        s = zeros(1, M);
        for m = 1:M
            s = s + sys(n-1).w(:, m)*mvnpdf(x.', (model.F(t(n))*sys(n-1).x(:, m)).', model.Q(t(n)).').';
        end
        lws = lwb + log(s) - logmvnpdf(x.', sys(n).mu.', sys(n).Sigma.').';
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
                
        % Point estimate
        xhat(:, n) = x*ws';
        
        % Store the particles and their weights
        sys(n).xs = x;
        sys(n).wb = wb;
        sys(n).ws = ws;
        sys(n).rs = r;
    end
end
