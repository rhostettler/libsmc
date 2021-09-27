function [xn, alpha, lqx, lqalpha, qstate] = sample_smcmc_bootstrap(model, y, x, ~, theta, par)
% # Sample from the bootstrap (prior) sequential MCMC kernel
% ## Usage
% * `[xn, alpha, lq, qstate] = sample_smcmc_bootstrap(model, y, x, lw, theta, par)`
%
% ## Description
% Prior (bootstrap) sequential Markov chain Monte Carlo kernel. Samples are
% drawn 
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
% 
% ## Output
% * `xp`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lqx`: 1-times-J vector of the importance density of the jth sample 
%   `xp`.
% * `lqalpha`: 1-times-J vector of the importance density of the jth
%   ancestor index `alpha`.
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

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    [dx, J] = size(x);
    def = struct( ...
        'Jburnin', round(0.1*J), ...
        'Jmixing', 1 ...
    );
    par = parchk(par, def);
    
    % No. of accepted samples
    naccept = 0;
    
    % Calculate the no. of Monte Carlo iterations
    Jmcmc = par.Jburnin + 1 + (J-1)*par.Jmixing;
    
    % Preallocate
    alpha = zeros(1, Jmcmc+1);
    xn = zeros(dx, Jmcmc+1);
    
    %% Initialization
    alpha(1) = randi(J, 1);
    xn(:, 1) = model.px.rand(x(:, alpha(1)), theta);
    lpy = model.py.logpdf(y, xn(:, 1), theta);

    %% Sample from the chain
    % N.B.: From 2 to +1 b/c of the xn(:, 1) containing the initial value
    for j = 2:Jmcmc+1
        % Sample from kernel (independent MH based on prior)
        % "Bootstrap SMCMC"
        alphap = randi(J, 1);
        xp = model.px.rand(x(:, alphap), theta);

        % Calculate acceptance probability
        lpyp = model.py.logpdf(y, xp, theta);            
        rho = min(1, exp(lpyp - lpy));

        % Accept/reject
        u = rand(1);
        if u < rho
            xn(:, j) = xp;
            alpha(j) = alphap;
            lpy = lpyp;
            naccept = naccept + 1;
        else
            xn(:, j) = xn(:, j-1);
            alpha(:, j) = alpha(:, j-1);
        end
    end
    
    %% Remove burn-in and mixing
    j = 1+(par.Jburnin+1:par.Jmixing:Jmcmc);
    alpha = alpha(j);
    xn = xn(:, j);
    lqx = -log(J)*ones(1, J);
    lqalpha = lqx;
    
    qstate = struct('rate', naccept/Jmcmc);
end
