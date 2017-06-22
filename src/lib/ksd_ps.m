function [xhat, Phat, sys] = ksd_ps(y, t, model, Mf, Ms, par, sys)
% Kronander-Sch?n-Dahlin marginal particle smoother
%
% SYNOPSIS
%
% DESCRIPTION
%
% PARAMETERS
%
% RETURNS
%
% REFERENCES
%
% VERSION
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults & Checks
    narginchk(3, 7);
    
    % Set default particle numbers
    if nargin < 5 || isempty(Mf)
        Mf = 250;
    end
    if nargin < 5 || isempty(Ms)
        Ms = 100;
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
        [~, ~, sys] = bootstrap_pf(y, t, model, Mf, par);
    end
    
    %% Smooth
    switch nargout
        case {0, 1, 2}
            [xhat, Phat] = smooth(y, t, model, Ms, par, sys);
        case 3
            [xhat, Phat, sys] = smooth(y, t, model, Ms, par, sys);
        otherwise
            error('Incorrect number of output arguments');
    end
end

%%
function [xhat, Phat, sys] = smooth(y, t, model, Ms, par, sys)
    %% Preliminaries
    [Nx, Mf, N] = size(sys.x);
    px = model.px;
    py = model.py;
    
    %% Initialize Variables
    % Preallocate storage for the particle system (if required)
    if nargout == 3
        sys.xs = zeros(Nx, Ms, N);
        sys.ws = zeros(1, Ms, N);
        sys.lws = zeros(1, Ms, N);
    end
    
    % Preallocate for estimates
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);

    %% Initialize Backward Pass
    ir = sysresample(sys.w(:, :, N));
    %b = ir(randi(Mf, 1, Ms));
    b = ir(randperm(Mf, Ms));
    xs = sys.x(:, b, N);
    xhat(:, N) = mean(xs, 2);
    lws = log(1/Ms)*ones(1, Ms);
    if nargout == 3
        sys.xs(:, :, N) = xs;
        sys.lws(:, :, N) = lws;
        sys.ws(:, :, N) = 1/Ms*ones(1, Ms);
    end
    
    %% Backward Iteration
    for n = N-1:-1:1
        % Sample ancestors
        lw = -Inf*ones(1, Ms);
        [indf, locf] = ismember(xs.', sys.x(:, :, n+1).', 'rows');
        lw(indf) = sys.lw(:, locf, n+1);
        if py.fast
            lv = lws + py.logpdf(y(:, n+1)*ones(1, Ms), xs, t(n+1)) - lw;
        else
            lv = zeros(1, Ms);
            for m = 1:Ms
                lv(m) = lws(m) + py.logpdf(y(:, n+1), xs(:, m), t(n+1)) - lw(m);
            end
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        b = sysresample(v);
        xp = xs(:, b);
        
        % Sample new particles
        ir = sysresample(sys.w(:, :, n));
        %a = ir(randi(Mf, 1, Ms));
        a = ir(randperm(Mf, Ms));
        xs = sys.x(:, a, n);
                
        % Calculate weights
        [ws, lws] = calculate_smoothed_weights(xp, xs, t(n+1), px);
        
        %% Estimate
        xhat(:, n) = xs*ws';
if 0
        % TODO: Not implemented yet. (We should have something like
        % 'mc_mean'
        for m = 1:Ms
            Phat(:, :, n) = Phat(:, :, n) + w(:, m);
        end
end
        
        %% Store
        if nargout == 3
            sys.xs(:, :, n) = xs;
            sys.ws(:, :, n) = ws;
            sys.lws(:, :, n) = lws;
        end
    end
end

%% Smoothed Weights
function [ws, lws] = calculate_smoothed_weights(xp, x, t, px)
    %% Calculate Weights
    if px.fast
        lws = px.logpdf(xp, x, t);
    else
        Ms = size(xp, 2);
        lws = zeros(1, Ms);
        for m = 1:Ms
            lws(:, m) = px.logpdf(xp(:, m), x(:, m), t);
        end
    end

    %% Normalize
    ws = exp(lws-max(lws));
    ws = ws/sum(ws);
    lws = log(ws);
end
