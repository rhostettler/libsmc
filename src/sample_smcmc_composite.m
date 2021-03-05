function [xn, alpha, lq, qstate] = sample_smcmc_composite(model, y, x, ~, theta, par)
% # Sample from a composite sequential MCMC kernel
% ## Usage
% * `[xn, alpha, lq, qstate] = sample_smcmc_composite(model, y, x, lw, theta, par)`
%
% ## Description
% Composite sequential Markov chain Monte Carlo kernel. In particular,
% samples are drawn according to the following strategy (`alpha`: ancestor 
% index, i.e., trajectory `x[0:n-1]`; `xn`: new sample):
%
% 1. Jointly sample {`alpha(j)`, `xn(j)`};
% 2. Use Metropolis-within-Gibbs to refine `alpha(j)` given `xn(j)`;
% 3. Use Metropolis-within-Gibbs to refine `xn(j)` given `alpha(j)`.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: Measurement vector y[n].
% * `x`: Samples at x[n-1].
% * `lw`: Log-weights of x[n-1] (unused; for compatibility only).
% * `theta`: Model parameters.
% * `par`: Struct of additional parameters:
%   - `Jburnin`: No. of burn-in samples (default: 10 % of J).
%   - `Jmixing`: No. of mixing samples (default: 0).
%   - `L`: No. of refinement steps in the Metropolis-within-Gibbs steps
%     (default: 1).
%   - `epsilon`: MALA integration step (default: 0.1).
% 
% ## Output
% * `xn`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lq`: 1-times-J vector of the importance density of the jth sample.
% * `qstate`: Sampling algorithm state information.
%
% ## Author
% 2020-present -- Roland Hostettler

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
% * Add possibility of changing joint draw (independent but with different 
%   draw for x[n]).
% * Add possibility of choosing refiniment kernels (MALA, GRW, etc.)

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    [dx, J] = size(x);
    def = struct( ...
        'Jburnin', round(0.1*J), ...
        'Jmixing', 1, ...
        'L', 1, ...                 % No. of refinement (Metropolis-within-Gibbs) steps
        'epsilon', 0.1 ...          % Integration step size
    );
    par = parchk(par, def);
    
    % Calculate the no. of Monte Carlo iterations
    Jmcmc = par.Jburnin + 1 + (J-1)*par.Jmixing;
    
    % Parameter defaults
    L = par.L;
    epsilon = par.epsilon;
    Sigma = eye(dx);                % Pre-conditioning matrix
    LSigma = chol(Sigma).';
    
    % Preallocate
    alpha = zeros(1, Jmcmc+1);      % Ancestor indices
    xn = zeros(dx, Jmcmc+1);        % New samples
    rate_alpha = zeros(1, Jmcmc+1); % Acceptance rate for alpha
    rate_x = zeros(1, Jmcmc+1);     % Acceptance rate for x

    %% Chain initialization
    % Sample chain initialization
    alpha(1) = randi(J, 1);
    xn(:, 1) = model.px.rand(x(:, alpha(1)), theta);

    %% Sample from the chain
    % Note: From 2 to +1 b/c of the xn(:, 1) containing the initial value
    for j = 2:Jmcmc+1
        %% Independent joint prior SMCMC draw
        alphap = randi(J, 1);
        xp = model.px.rand(x(:, alphap), theta);
        
        % Metropolis-Hastings accept/reject
        lpp = model.py.logpdf(y, xp, theta);
        lp = model.py.logpdf(y, xn(:, j-1), theta);
        rho = min(1, exp(lpp-lp));
        u = rand(1);
        if u <= rho
            alpha(j) = alphap;
            xn(:, j) = xp;
        else
            alpha(j) = alpha(j-1);
            xn(:, j) = xn(:, j-1);
        end
        
        %% Refinement of x[n-1]
        % Sample ancestor index using Metropolis-within-Gibbs
        p = @(alpha) model.px.logpdf(xn(:, j), x(:, alpha), theta);
        q = struct( ...
            'rand', @(xa) randi(J, 1), ...
            'logpdf', @(xap, xa) -log(J) ...
        );
        [alphap, rate_alpha(:, j)] = metropolis_hastings(p, alpha(j), q, L);
        alpha(j) = alphap(L);

        %% Refinement of x[n]
        % Metropolis-within-Gibbs
        p = @(xp) model.py.logpdf(y, xp, theta) + model.px.logpdf(xp, x(:, alpha(j)), theta);

        % TODO: Make this configurable
        if 1
            % MALA
            G = @(xp) ( ...
                -model.py.loghessian(y, xp, theta) ...
                - model.px.loghessian(xp, x(:, alpha(j)), theta) ...
            );
            f = @(xp) xp + 1/2*epsilon*(G(xp)\Sigma)*( ...
                model.py.loggradient(y, xp, theta) ...
                + model.px.loggradient(xp, x(:, alpha(j)), theta) ...
            );
            q = struct( ...
                'rand', @(x) f(x) + chol(G(x)).'\(sqrt(epsilon)*LSigma.'*randn(dx, 1)), ...
                'logpdf', @(xp, x) logmvnpdf(xp.', f(x).', G(x)\(epsilon*Sigma)).' ...
            );
        else
            % Random walk
            q = struct( ...
                'rand', @(x) x + LSigma.'*randn(dx, 1), ...
                'logpdf', @(xp, x) 0 ...
            );
        end
    
        [xp, rate_x(:, j)] = metropolis_hastings(p, xn(:, j), q, L);
        xn(:, j) = xp(:, L);
    end
    
    %% Strip burn-in and mixing
    j = 1+(par.Jburnin+1:par.Jmixing:Jmcmc);
    alpha = alpha(j);
    xn = xn(:, j);
    lq = -log(J)*ones(1, J);
    
    qstate = struct('rate_alpha', rate_alpha, 'rate_x', rate_x);
end
