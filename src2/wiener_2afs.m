function [xhat, Phat, sys] = wiener_2afs(y, t, model, Mf, Ms, par, sys)
% Two filter particle smoother for Wiener state space systems
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
    if nargin < 4 || isempty(Mf)
        Mf = 100;
    end
    
    if nargin < 5 || isempty(Ms)
        Ms = Mf;
    end
    
    if Ms ~= Mf
        error('Sorry, not implemented yet.');
    end
    
    % Default parameters
    if nargin < 6
        par = [];
    end
    
    % Check if a filtered particle system is provided
    if nargin < 7 || isempty(sys)
        filter = 0;
    else
        filter = 1;
    end
    
    %% Filter
    % If no filtered system is provided, run a bootstrap PF
    if ~filter
        [~, ~, sys] = wiener_gaapf(y, t, model, Mf, par);
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
            [xhat, Phat] = smooth(y, t, model, Ms, par, sys, mu_x, Sigma_x);
        case 3
            [xhat, Phat, sys] = smooth(y, t, model, Ms, par, sys, mu_x, Sigma_x);
        otherwise
            error('Incorrect number of output arguments');
    end
end

%% 
function [xhat, Phat, sys] = smooth(y, t, model, Ms, par, sys, mu_x, Sigma_x)
    py = model.py;

    %% Preallocate
    [Nx, Mf, N] = size(sys.x);
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    if nargout == 3
        sys.xs = zeros(Nx, Ms, N);
        sys.wb = zeros(1, Ms, N);
        sys.lwb = zeros(1, Ms, N);
        sys.ws = zeros(1, Ms, N);
        sys.lws = zeros(1, Ms, N);
    end
    
    % Recurring temporary variables
    % TODO: sort out
    Ny = size(y, 1);
    mu_xb = zeros(Nx, Ms);
    mu_yb = zeros(Ny, Ms);
    B = zeros(Nx, Ny, Ms);
    S = zeros(Ny, Ny, Ms);
    
    lv = zeros(1, Mf);

    %% Initialize Backward Filter
    for m = 1:Mf
        lv(:, m) = logmvnpdf(sys.x(:, m, N).', mu_x(:, N).', Sigma_x(:, :, N)).' ...
            + py.logpdf(y(:, N), sys.x(:, m, N), t(N)) - sys.lw(:, m, N);
    end
    v = exp(lv-max(lv));
    v = v/sum(v); 
    beta = sysresample(v);
    xs = sys.x(:, beta, N);
    lwb = log(1/Ms)*ones(1, Ms);
    wb = 1/Ms*ones(1, Ms);
    lws = log(1/Ms)*ones(1, Ms);
    ws = 1/Ms*ones(1, Ms);
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
        % Backward Resampling
        F = model.F(t);
        Q = model.Q(t);
        mu = mu_x(:, n);
        Sigma = Sigma_x(:, :, n);
        L = F*Sigma*F' + Q;
        K = Sigma*F'/L;
        Sigma_xb = Sigma - K*L*K';
        for m = 1:Ms
            mu_xb(:, m) = mu + K*(xs(:, m) - F*mu);
            [mu_yb(:, m), S(:, :, m), B(:, :, m)] = calculate_moments(mu_xb(:, m), t, model);
            lv(m) = lwb(m) + logmvnpdf(y(:, n).', mu_yb(:, m).', S(:, :, m)).';
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        lv = log(v);
        beta = sysresample(v);

        % Draw new Samples
        for m = 1:Ms
            % TODO: These things could be calculated in the above for loop
            %       too.
            % Draw a new particle
            K = B(:, :, beta(m))/S(:, :, beta(m));
            mu_xbp = mu_xb(:, beta(m)) + K*(y(:, n) - mu_yb(:, beta(m)));
            C_xbp = Sigma_xb - K*S(:, :, beta(m))*K';
            xs(:, m) = mu_xbp + chol(C_xbp, 'lower')*randn(Nx, 1);

            % Calculate the non-normalized importance weight
            L = B(:, :, beta(m))'/Sigma_xb;
            mu_ybp = mu_yb(:, beta(m)) + L*(xs(:, m) - mu_xb(:, beta(m)));
            C_ybp = S(:, :, beta(m)) - L*Sigma_xb*L';
            
            lwb(:, m) = ( ...
                py.logpdf(y(:, n), xs(:, m), t(n))-logmvnpdf(y(:, n).', mu_ybp.', C_ybp) ...
            );
        end
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);
        lwb = log(wb);

        %% Smoothing
        % TODO: Continue here
        if n > 1
            x = sys.x(:, :, n-1);
            w = sys.w(:, :, n-1);
        else
            x = sys.x0;
            w = 1/Mf*ones(1, Mf);
        end
        nu = zeros(1, Ms);
        for m = 1:Ms
            nu(m) = w*mvnpdf(xs(:, m).', (F*x).', Q);
%             tmp = lw + logmvnpdf(xs(:, m).', (F*x).', Q).';
%             lnu(m) = log(sum(exp(tmp)));
        end
        lws = lwb - logmvnpdf(xs.', mu.', Sigma).' + log(nu);
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
        lws = log(ws);
        xhat(:, n) = xs*ws.';

        %% Store
        if nargout == 3
            sys.xs(:, :, n) = xs;
            sys.lwb(:, :, n) = lwb;
            sys.wb(:, :, n) = wb;
            sys.lws(:, :, n) = lws;
            sys.ws(:, :, n) = ws;
        end
    end
end

%% Moment Matching
% TODO: Ideally, I want to repalce this with a function supplied by 'par',
%       similar as for 'resample' (which, by the way, has to be made more
%       generic itself)
% TODO: This is copy & paste from wiener_gaapf.m
function [yhat, S, B] = calculate_moments(x, t, model)
    Q = model.Q(t);
    R = model.R(t);
    r = zeros(size(R, 1));
    [yhat, Gx, Gr] = model.g(x, r, t);
    S = Gx*Q*Gx' + Gr*R*Gr';
    B = Q*Gx';
end
