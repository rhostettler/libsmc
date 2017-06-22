function [xhat, Phat, sys] = sisr_pf(y, t, model, q, M, par)
% Sequential importance sampling w/ resampling particle filter
%
% SYNOPSIS
%   [xhat, Phat] = sisr_pf(y, t, model, q)
%   [xhat, Phat, sys] = sisr_pf(y, t, model, q, M, par)
%
% DESCRIPTION
%   
%
% PARAMETERS
%   
%
% VERSION
%   2017-03-23
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Replace weighing function (see Wiener-apfs)
%   * Use global calculate_incremental_weights() instead

    %% Preliminary Checks
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(4, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'resample', @resample_ess, ... % Resampling function
        'bootstrap', false ...
    );
    par = parchk(par, def);
    [px, py, px0] = modelchk(model);

    %% Initialize
    x = px0.rand(M);
    lw = log(1/M)*ones(1, M);
    
    %% Preallocate
    Nx = size(x, 1);
    N = length(t);
    if nargout == 3
        alphas = zeros(1, M, N);    % Resampling indices
        sys.x = zeros(Nx, M, N);    % Non-resampled particles
        sys.xf = zeros(Nx, M, N);   % Full trajectories
        sys.w = zeros(1, M, N);     % Non-resampled particle weights
        sys.lw = zeros(1, M, N);    % Non-resampled log-weights
        sys.r = zeros(1, N);        % Resampling indicator
    end
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    
    %% Process Data
    for n = 1:N
        %% Resample
        [alpha, lw, r] = par.resample(lw, par);

        %% Draw Samples
        xp = draw_samples(y(:, n), x(:, alpha), t(n), q);
        
        %% Weights
        [~, lv] = calculate_incremental_weights(y(:, n), xp, x, t(n), px, py, q, par);
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        x = xp;
        
        %% Point Estimates
        xhat(:, n) = x*w';
if 0
        % Takes too much time; disabled for now.
        for m = 1:M
            Phat(:, :, n) = Phat(:, :, n) + w(m)*((xp(:, m)-xhat(:, n))*(xp(:, m)-xhat(:, n))');
        end
end

        %% Store
        if nargout == 3
            sys.x(:, :, n) = x;
            sys.w(:, :, n) = w;
            sys.lw(:, :, n) = lw;
            sys.r(:, n) = r;
            alphas(:, :, n) = alpha;
        end
    end
    
    %% Post-processing
    if nargout == 3
        % Put together trajectories: Walk down the ancestral tree, 
        % backwards in time to get the correct lineage.
        alpha = 1:M;
        sys.xf(:, :, N) = sys.x(:, :, N);
        for n = N-1:-1:1
            sys.xf(:, :, n) = sys.x(:, alphas(:, alpha, n+1), n);
            alpha = alphas(:, alpha, n+1);
        end
        sys.wf = sys.w(:, :, N);
    end
end

%% New Samples
% function xp = draw_samples(y, x, t, q)
%     M = size(x, 2);
%     if q.fast
%         xp = q.rand(y*ones(1, M), x, t);
%     else
%         xp = zeros(size(x));
%         for m = 1:M
%             xp(:, m) = q.rand(y, x(:, m), t);
%         end
%     end
% end

%% Incremental Particle Weight
function [v, lv] = calculate_incremental_weights(y, xp, x, t, px, py, q, par)
    M = size(xp, 2);
    if par.bootstrap
        if py.fast
            lv = py.logpdf(y*ones(1, M), xp, t);
        else
            lv = zeros(1, M);
            for m = 1:M
                lv(m) = py.logpdf(y, xp(:, m), t);
            end
        end
    else
        if px.fast && py.fast && q.fast
            lv = ( ...
                py.logpdf(y*ones(1, M), xp, t) ...
                + px.logpdf(xp, x, t) ...
                - q.logpdf(xp, y*ones(1, M), x, t) ...
            );
        else
            M = size(xp, 2);
            lv = zeros(1, M);
            for m = 1:M
                lv(m) = ( ...
                    py.logpdf(y, xp(:, m), t) ...
                    + px.logpdf(xp(:, m), x(:, m), t) ...
                    - q.logpdf(xp(:, m), y, x(:, m), t) ...
                );
            end
        end
    end
    v = exp(lv-max(lv));
end
