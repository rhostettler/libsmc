function [xhat, Phat, sys] = wiener_2bfs(y, t, model, M, par, sys)
% Two filter particle smoother for degenerate Wiener state space systems
% 
% SYNOPSIS
% 
%
% DESCRIPTION
%   EKF-like linearization for now.
%
% PARAMETERS
% 
%
% RETURNS
%
%
% SEE ALSO
%
%
% VERSION
%   2017-03-28
% 
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>   
   
    %% Defaults & Checks
    narginchk(3, 6);
    
    % Set default particle numbers
    if nargin < 4 || isempty(M)
        M = 100;
    end
    
    % Default parameters
    if nargin < 5
        par = [];
    end
    
    % Check if a filtered particle system is provided
    if nargin < 6 || isempty(sys)
        filter = 0;
    else
        filter = 1;
    end
    
    %% Filter
    % If no filtered system is provided, run a bootstrap PF
    if ~filter
        [~, ~, sys] = bootstrap_pf(y, t, model, M, par);
    end
    
    %% Forward Stats
    % Inefficient, but hey!
    [Nx, ~, N] = size(sys.x);
    mu_x = zeros(Nx, N+1);
    Sigma_x = zeros(Nx, Nx, N+1);
    mu_x(:, 1) = model.m0;
    Sigma_x(:, :, 1) = model.P0;
    for n = 1:N
        F = model.F(t(n));
        Q = model.Q(t(n));
        mu_x(:, n+1) = F*mu_x(:, n);
        Sigma_x(:, :, n+1) = F*Sigma_x(:, :, n)*F' + Q;
    end
    mu_x = mu_x(:, 2:n+1);
    Sigma_x = Sigma_x(:, :, 2:n+1);
    
    %% Smoothing
    switch nargout
        case {0, 1, 2}
            [xhat, Phat] = smooth(y, t, model, M, par, sys, mu_x, Sigma_x);
        case 3
            [xhat, Phat, sys] = smooth(y, t, model, M, par, sys, mu_x, Sigma_x);
        otherwise
            error('Incorrect number of output arguments');
    end
end

%% 
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
