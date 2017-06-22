function [x, sys] = wiener_cpfas(y, t, model, M, par)
% 

% TODO:
%   * Needs to be simplified in the same way as all the other cpf-things

    %% Parameter check & defaults
    % Check that we get the correct no. of parameters and a well-defined
    % model so that we can detect model problems already here.
    narginchk(3, 6);
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'xt', [] ...      % Default trajectory
    );
    par = parchk(par, def);
    [px, py, px0] = modelchk(model);
        
    %% Initialize
    x = px0.rand(M);
    w = 1/M*ones(1, M);
    lw = log(1/M)*ones(1, M);
    
    %% Preallocate
    [Ny, N] = size(y);
    Nx = size(x, 1);
    xt = par.xt;
    if isempty(xt)
        xt = zeros(Nx, N);
    end
    xf = zeros(Nx, M, N);
    wf = zeros(1, M, N);
    alphaf = zeros(1, M, N);

    xn = zeros(Nx, M);      % Stores the predicted state at each iteration
    xhat_p = zeros(Nx, M);  % Matched moments from here on down
    C = zeros(Nx, Nx, M);
    yhat_p = zeros(Ny, M);
    B = zeros(Nx, Ny, M);
    S = zeros(Ny, Ny, M);
    lv = zeros(1, M);

    %% Iterate over the Data
    for n = 1:N
        F = model.F(t(n));
        Q = model.Q(t(n));

        %% Calculate the importance distribution's moments
        for m = 1:M
            xp = F*x(:, m);
            [yhat_p(:, m), S(:, :, m), B(:, :, m)] = calculate_moments(xp, t(n), model);
            K = B(:, :, m)/S(:, :, m);
            xhat_p(:, m) = xp + K*(y(:, n) - yhat_p(:, m));
            C(:, :, m) = Q - K*S(:, :, m)*K';
            lv(m) = lw(m) + logmvnpdf(y(:, n).', yhat_p(:, m).', S(:, :, m)).';
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        lv = log(v);

        %% Propagate M-1 Particles
        alpha = sysresample(v);
        for m = 1:M
            % Propagate the particles (observe that the Mth particle
            % will be overwritten below since this will be from the
            % predefined trajectory)
            xn(:, m) = xhat_p(:, alpha(m)) ...
                + chol(C(:, :, alpha(m)), 'lower')*randn(Nx, 1);
            K = B(:, :, alpha(m))'/Q;
            mu = yhat_p(:, alpha(m)) + K*(xn(:, m) - F*x(:, alpha(m)));
            Sigma = S(:, :, alpha(m)) - K*Q*K';
            
            % Update particle weight
            lw(:, m) = py.logpdf(y(:, n), xn(:, m), t(n)) - logmvnpdf(y(:, n), mu, Sigma);            
        end

        %% Propagate the Mth Particle
        xn(:, M) = xt(:, n);
        
        % Calculate ancestor weights
        % TODO: Should this really be lv and not lw (before recalculating though)?
        lva = lv + px.logpdf(xt(:, n)*ones(1, M), x, t(n));
        va = exp(lva-max(lva));
        va = va/sum(va);
        tmp = sysresample(va);
        alpha(M) = tmp(randi(M, 1));
        lw(:, M) = py.logpdf(y(:, n), xn(:, M), t(n)) - logmvnpdf(y(:, n), mu, Sigma);

        %% Set particles and weights
        x = xn;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        %% Extend trajectories
        xf(:, :, n) = x;
        wf(:, :, n) = w;
        alphaf(:, :, n) = alpha;
    end
    
    %% 
    if nargout >= 2
        sys = struct();
        sys.xf = xf;
        sys.wf = wf;
    end
    
    %% Post-processing
    % Put together trajectories; moved here for performance reasons
    % Walk down the ancestral tree, backwards in time to get the correct
    % lineage.
    alpha = 1:M;
    xf(:, :, N) = xf(:, :, N);
    for n = N-1:-1:1
        xf(:, :, n) = xf(:, alphaf(:, alpha, n+1), n);
        alpha = alphaf(:, alpha, n+1);
    end
    
    % Store the complete trajectories
    if nargout >= 2
        sys.x = xf;
    end
    
    % Draw a trajectory
    beta = sysresample(w);
    j = beta(randi(M, 1));
    x = squeeze(xf(:, j, :));
end
