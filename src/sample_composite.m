function [xn, alpha, qstate] = sample_composite(model, y, x, theta, Jmcmc, par)
% # Sample from a composite sequential MCMC kernel
% ## Usage
%
% ## Description
%
% ## Input
%
% ## Output
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
% * Add proper documentation; should clearly describe how sampling is
%   performed to make sure that we remember later on.
% * Add proper parameter handling:
%   * Should we strip burnin and such already here?
%   * We should be able to choose the refinement kernel.
% * Sample `alpha` instead of `xa`, and also return the ancestor indices
%   such that we can calculate full posteriors in smcmc (or, if merged, in
%   pf).
% * If we're going to make it compatible with `pf` (i.e., we remove smcmc),
%   we'll have to make sure that we comply with that interface.
% * Allow for different joint draws (independent but with different draw
%   for x[n]).

    %% Defaults and variable initialization
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct( ...
        'L', 1, ...              % No. of refinement (Metropolis-within-Gibbs) steps
        'epsilon', 0.1 ...       % Integration step size
    );
    par = parchk(par, def);
        
    [dx, J] = size(x);       % Determine state dimension and no. of particles
    
    L = par.L;
    epsilon = par.epsilon;
    Sigma = eye(dx);            % Pre-conditioning matrix
    LSigma = chol(Sigma).';
    
    xa = zeros(dx, Jmcmc+1);    % Ancestor
    xn = zeros(dx, Jmcmc+1);    % New sample
    
    % Acceptance rates for the refinement steps for ancestor indices and
    % samples
    rate_x = zeros(1, Jmcmc+1);
    rate_alpha = zeros(1, Jmcmc+1);

    %% Chain initialization
    % Variables:
    % * xn/alphan stores the newly drawn states
    % * xp/alphap stores the proposed state
    % * x contains the samples from the previous iteration (n-1)

    % Sample chain initialization
    xa(:, 1) = x(:, randi(J, 1));
    xn(:, 1) = model.px.rand(xa(:, 1), theta);

    %% Sample from the chain
    % Note: From 2 to +1 b/c of the xn(:, 1) containing the initial value
    for j = 2:Jmcmc+1
        %% Independent joint prior SMCMC draw
        % TODO: Can we formulate this such that we use metropolis_hastings?
        % Prior joint draw
        xap = x(:, randi(J, 1));
        xp = model.px.rand(xap, theta);
        
        % Metropolis-Hastings accept/reject
        lpp = model.py.logpdf(y, xp, theta);
        lp = model.py.logpdf(y, xn(:, j-1), theta);
        rho = min(1, exp(lpp-lp));
        u = rand(1);
        if u <= rho
            xn(:, j) = xp;
            xa(:, j) = xap;
        else
            xn(:, j) = xn(:, j-1);
            xa(:, j) = xa(:, j-1);
        end
        
        %% Refinement of x[n-1]
        % Metropolis-within-Gibbs
        % TODO: Reformulate this as sampling over alphas (ancestor indices)
        p = @(xa) model.px.logpdf(xn(:, j), xa, theta);
        q = struct( ...
            'rand', @(xa) x(:, randi(J, 1)), ...
            'logpdf', @(xap, xa) -log(J) ...
        );
        [xap, rate_alpha(:, j)] = metropolis_hastings(p, xa(:, j), q, L);
        xa(:, j) = xap(:, L);

        %% Refinement of x[n]
        % Metropolis-within-Gibbs
        p = @(x) model.py.logpdf(y, x, theta) + model.px.logpdf(x, xa(:, j), theta);

        if 1
            % MALA
            G = @(x) -model.py.loghessian(y, x, theta) - model.px.loghessian(x, xa(:, j), theta);
            f = @(x) x + 1/2*epsilon*(G(x)\Sigma)*(model.py.loggradient(y, x, theta) + model.px.loggradient(x, xa(:, j), theta));
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
    
    alpha = 1:Jmcmc+1; % TODO: this is not correct, see TODO above.
    
    %% 
    if nargout > 2
        qstate = struct('rate_alpha', rate_alpha, 'rate_x', rate_x);
    end
end
