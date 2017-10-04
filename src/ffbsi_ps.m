function [xhat, Phat, sys] = ffbsi_ps(y, t, model, Mf, Ms, par, sys)
% Forward filtering backward simulation particle smoother
% 
% SYNOPSIS
%   [xhat, Phat] = ffbsi_ps(y, t, model)
%   [xhat, Phat, sys] = ffbsi_ps(y, t, model, Mf, Ms, par, sys)
%
% DESCRIPTION
%   
%
% PROPERTIES
% 
%
% METHODS
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

% TODO
%   * Update documentation
%   * Implement rejection sampling w/ adaptive stopping
%   * Check how I can merge that with other backward simulation smoothers,
%     e.g. ksd_ps (they use exactly the same logic in the beginning, only
%     the smooth()-function is different

    %% Defaults & Checks
    narginchk(3, 7);
    
    % Set default particle numbers
    if nargin < 4 || isempty(Mf)
        Mf = 250;
    end
    if nargin < 5 || isempty(Ms)
        Ms = 100;
    end
    
    % Default parameters
    if nargin < 6
        par = [];
    end
    
    %% Filter
    % If no filtered system is provided, run a bootstrap PF
    if nargin < 7 || isempty(sys)
        [~, ~, sys] = bootstrap_pf(y, t, model, Mf, par);
    end
    
    %% Backward simulation
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
    px = model.px;
    
    %% Preallocate
    [Nx, Mf, N] = size(sys.x);
    xhat = zeros(Nx, N);
    Phat = zeros(Nx, Nx, N);
    if nargout == 3
        sys.xs = zeros(Nx, Ms, N);
        sys.ws = 1/Ms*ones(1, Ms, N);
        sys.lws = log(1/Ms)*ones(1, Ms, N);
    end
        
    %% Initialize
    % TODO: Use randi() instead of randperm?
    ir = sysresample(sys.w(:, :, N));
    b = ir(randperm(Mf, Ms));
    xs = sys.x(:, b, N);
    xhat(:, N) = mean(xs, 2);
            
    %% Store
    if nargout == 3
        sys.xs(:, :, N) = xs;
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        %% Iteration
        % j -> trajectory to expand
        % i -> candidate particles
        for j = 1:Ms
            % Compute the backward smoothing weights
            if px.fast
                lwb = sys.lw(:, :, n) + px.logpdf(xs(:, j)*ones(1, Mf), sys.x(:, :, n), t(n+1));
            else
                lwb = zeros(1, Mf);
                for i = 1:Mf
                   lwb(i) = sys.lw(:, i, n) + px.logpdf(xs(:, j), sys.x(:, i, n), t(n+1));
                end
            end
            wb = exp(lwb-max(lwb));
            wb = wb/sum(wb);

            % Draw a new particle from the categorical distribution and
            % extend the trajectory
            ir = sysresample(wb);
            b = ir(randi(Mf, 1));
            xs(:, j) = sys.x(:, b, n);
        end

        %% Estimate & Store
        xhat(:, n) = mean(xs, 2);
        if nargout == 3
            sys.xs(:, :, n) = xs;
        end
    end
end
