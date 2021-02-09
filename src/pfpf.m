function [xhat, sys] = pfpf(model, y, theta, J, par)
% # Particle flow particle filter
% ## Usage
% * `xhat = pfpf(model, y)`
% * `[xhat, sys] = pfpf(model, y, theta, J, par)`
%
% ## Description
% Particle flow particle filter with invertible particle flow according to
% [1].
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: dtheta-times-1 vector or dtheta-times-N matrix of additional
%   parameters (default: `[]`).
% * `J`: Number of particles (default: 100).
% * `par`: Struct of additional (optional) parameters:
%     - `[alpha, lw, r] = resample(lw)`: Function handle to the resampling 
%       function. The input `lw` is the log-weights and the function must 
%       return the indices of the resampled particles (`alpha`), the log-
%       weights of the resampled (`lw`) particles, as well as a boolean
%       indicating whether resampling was performed or not (`r`). Default:
%       `@resample_ess`.
%     - `L`: Number of integration steps.
%
% ## Output
% * `xhat`: Minimum mean squared error filter state estimate (calculated 
%   using the marginal filtering density).
% * `sys`: Particle system array of structs with the following fields:
%     - `x`: dx-times-J matrix of particles for the marginal filtering 
%       density.
%     - `w`: 1-times-J vector of the particle weights for the marginal
%       filtering density.
%      - `xf`: dx-times-J vector of state trajectory samples.
%     - `alpha`: 1-times-J vector of ancestor indices.
%     - `rstate`: Resampling algorithm state.
%
% ## References
% 1. Y. Li and M. Coates, “Particle filtering with invertible particle 
%    flow,” IEEE Transactions on Signal Processing, vol. 65, no. 15, pp. 
%    4102–4116, August 2017.
%
% ## Authors
% 2019-present -- Roland Hostettler

%{
% This file is part of the libsmc Matlab toolbox.
%
% libsmc is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libsmc is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libsmc. If not, see <http://www.gnu.org/licenses/>.
%}

% TODO:
% * Code optimizations should be considered at some point (at least the
%   no. of loops in the sampling function can be reduced.
% * Enable choice of flow type (EDH/LEDH) through par.
% * Integrate this into the `pf` function.

    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    if nargin < 5
        par = struct();
    end
    def = struct( ...
        'L', 10, ...
        'resample', @resample_ess, ...
        'ukf', [] ...
    );
    par = parchk(par, def);
%     modelchk(model);

    %% Initialize
    % Sample initial particles
    x = model.px0.rand(J);
    lw = log(1/J)*ones(1, J);
    P = repmat(model.px0.P, [1, 1, J]);
    
    % Since we also store and return the initial state (in 'sys'), a dummy
    % (NaN) measurement is prepended to the measurement matrix so that we 
    % can use consistent indexing in the processing loop.
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta' to the appropriate size, such that we can use
    % 'theta(n)' as an argument to the different functions (if not already
    % expanded).
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];
    
    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    return_sys = (nargout >= 2);
    if return_sys
        sys = initialize_sys(N, dx, J);
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).alpha = 1:J;
        sys(1).rstate = struct('r', false, 'ess', J);
        sys(1).qstate = [];
    end
    xhat = zeros(dx, N-1);
    
    %% Process Data
    for n = 2:N
        %% Update
        % (Re-)Sample
        [alpha, lw, rstate] = par.resample(lw);
        [xp, lqx, P, qstate] = sample_ledh(model, y(:, n), x(:, alpha), theta(:, n), lw, P(:, :, alpha), par);
        
        % Calculate and normalize weights
        lv = calculate_incremental_weights_generic(model, y(:, n), xp, x(:, alpha), theta(:, n), lqx);
        lw = lw+lv;
        lw = lw-max(lw);
        w = exp(lw);
        w = w/sum(w);
        lw = log(w);
        x = xp;
        if any(~isfinite(w))
            warning('libsmc:warning', 'NaN/Inf in particle weights.');
        end
        
        %% Point Estimate(s)
        % Note: We don't have a state estimate for the initial state (in
        % the filtered version, anyway), thus we save the MMSE estimate in
        % n-1.
        xhat(:, n-1) = x*w';

        %% Store
        if return_sys
            sys(n).x = x;
            sys(n).w = w;
            sys(n).alpha = alpha;
            sys(n).rstate = rstate;
            sys(n).qstate = qstate;
        end
    end
    
    %% Calculate Joint Filtering Density
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end

%% Global EDH flow
function [xp, lqx, P, qstate] = sample_edh(model, y, x, theta, lw, P, par)
    % TODO: I think we can get rid of the eta0 and eta1 the way it is
    % implemented now (directly operate on xp, also rename etabar to xbar
    % or something.
    
    [dx, J] = size(x);
    qstate = [];
    xp = zeros(dx, J);
    lqx = zeros(1, J);

    f = model.px.m;
    Q = model.px.P;
    
    g = model.py.m;
    G = model.py.dm;
    R = model.py.P;
    
    alpha = par.ukf(1);
    beta = par.ukf(2);
    kappa = par.ukf(3);
    
    % UKF prediction
    w = exp(lw-max(lw));
    w = w/sum(w);
    m = x*w.';
    [mp, Pp] = ukf_predict1(m, P, f, Q, theta, alpha, beta, kappa);
    
    Ix = eye(dx);

    % Step size and integration steps
    epsilon = 1/par.L;
    lambda = (1:par.L)/par.L;
    
    %% Sample each particle
    % Auxiliary variable
    eta0bar = f(m, theta);

    eta0 = zeros(dx, J);
    for j = 1:J
        %% Propagate particles
        eta0(:, j) = model.px.rand(x(:, j), theta);
        lqx(:, j) = model.px.logpdf(eta0(:, j), x(:, j), theta);
    end
    eta1 = eta0;
        
    %% Particle flow
    etabar = eta0bar;
    for l = 1:par.L
        % Calculate Al and bl with the linearzation at etabar using
        % lambda(l)
        Gl = G(etabar, theta);
        el = g(etabar, theta) - Gl*etabar;
        Al = -1/2*P*Gl'/(lambda(l)*Gl*P*Gl' + R)*Gl;
        bl = (Ix + 2*lambda(l)*Al)*((Ix + lambda(l)*Al)*Pp*Gl'/R*(y - el) + Al*eta0bar);
        
        % Propagate auxiliary variable
        etabar = etabar + epsilon*(Al*etabar + bl);
        
        % Propagate particles
        for j = 1:J
            eta1(:, j) = eta1(:, j) + epsilon*(Al*eta1(:, j) + bl);
            if l == par.L
                lqx(:, j) = lqx(:, j) - model.px.logpdf(eta1(:, j), x(:, j), theta);
            end
        end
    end
    
    %% Set particles and calculate proposal weight
    xp = eta1;
    
    % UKF update
    [~, P] = ukf_update1(mp, Pp, y, g, R, theta, alpha, beta, kappa);
end

%% Local EDH flow
function [xp, lqx, P, qstate] = sample_ledh(model, y, x, theta, ~, P, par)
    qstate = [];
    [dx, J] = size(x);
    
    % Get function handles
    f = model.px.m;
    F = model.px.dm;
    Q = model.px.P;
    g = model.py.m;
    G = model.py.dm;
    R = model.py.P;
    
    if isfield(model.py, 'ytr')
        y = model.py.ytr(y, theta);
    end
    
    % Preallocate
    lgamma = ones(1, J);
    lpeta0 = zeros(1, J);
    etabar = zeros(dx, J);
    eta0 = zeros(dx, J);
    mp = zeros(dx, J);
    Pp = zeros(dx, dx, J);
    Ix = eye(dx);

    % generate lambdas; exponential spacing as discussed in Li & Coates
    % (2017)
    lambda_ratio = 1.2;
    lambda = (1 - lambda_ratio)/(1 - lambda_ratio^par.L);
    epsilon = lambda*lambda_ratio.^(0:par.L-1);
    lambda = cumsum(epsilon);

    for j = 1:J
        % Apply ukf/ekf prediction to calculate Pp (and also P; moved here
        % to save an extra iteration over J)
        [mp(:, j), Pp(:, :, j)] = ekf_predict1(x(:, j), P(:, :, j), F, Q, f, [], theta);
        if isa(R, 'function_handle')
            Rj = R(mp(:, j), theta);
        else
            Rj = R;
        end
        [~, P(:, :, j)] = ekf_update1(mp(:, j), Pp(:, :, j), y, G, Rj, g, [], theta);
        etabar(:, j) = f(x(:, j), theta);
        eta0(:, j) = model.px.rand(x(:, j), theta);
        lpeta0(:, j) = model.px.logpdf(eta0(:, j), x(:, j), theta);
    end
    
    eta1 = eta0;
    eta0bar = etabar;
    for l = 1:par.L
        lambdal = lambda(l);
        epsilonl = epsilon(l);
        for j = 1:J
            % Calculate Ajl and bjl from (13) and (14) at etabarj, but only
            % if Pp and Rj are full rank (indicative of problems for some
            % models)
            % TODO: Consider replacing with covariance regularization as in
            % the original code (might be problematic for singular
            % measurement models, though).
            Pj = Pp(:, :, j);            
            [~, ndP] = chol(Pj, 'lower');
            if isa(R, 'function_handle')
                Rj = R(etabar(:, j), theta);
            else
                Rj = R;
            end
            [~, ndR] = chol(Rj, 'lower');
            if ndP == 0 && ndR == 0
                Gj = G(etabar(:, j), theta);
                ej = g(etabar(:, j), theta) - Gj*etabar(:, j);
                Ajl = -1/2*Pj*Gj'/(lambdal*Gj*Pj*Gj' + Rj)*Gj;
                bjl = (Ix + 2*lambdal*Ajl)*( ...
                    (Ix + lambdal*Ajl)*Pj*Gj'/Rj*(y - ej) + Ajl*eta0bar(:, j) ...
                );

                % Migrate axiliary variables and particles
                etabar(:, j) = etabar(:, j) + epsilonl*(Ajl*etabar(:, j) + bjl);
                eta1(:, j) = eta1(:, j) + epsilonl*(Ajl*eta1(:, j) + bjl);

                % Calculate weight factor
                lgamma(:, j) = lgamma(:, j) + log(abs(det(Ix + epsilonl*Ajl)));
            else
                warning('libsmc:warning', 'Covariance not positive definite, not moving particles.');
            end
        end
    end
    
    % Set particles and calculate proposal weights
    xp = eta1;
    lqx = lpeta0 - lgamma;
end
