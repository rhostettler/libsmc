function [thetas, rate] = metropolis_hastings(p, theta0, q, K, par)
% # Metropolis-Hastings Markov chain Monte Carlo sampler
% ## Usage
% * `theta = metropolis_hastings(p, theta0)`
% * `[theta, rate] = metropolis_hastings(p, theta0, q, K, par)`
%
% ## Description
% Metropolis-Hastings Markov chain Monte Carlo sampler for drawing samples
% from a target distribution.
%
% ## Input
% * `p`: Function handle `@(theta)`of the logarithm of the target
%   distribution.
% * `theta0`: Initial parameter guess (starting point of the Markov chain).
% * `q`: Struct defining the proposal chain. The following fields must be
%   present:
%     - `rand`: Function handle `@(theta)` to draw random samples `theta_p
%       ~ q(theta_p | theta)`.
%     - `logpdf`: Logarithm of the probability density function of the 
%       proposal distribution (function handle `@(theta_p, theta)`.
%
%    Providing a proposal is optional but highly recommended. By default, a
%    random walk proposal `N(theta, I)` is used.
% * `K`: No. of samples to draw (default: 100).
% * `par`: Struct with additional parameters:
%     - `Kburnin`: No. of samples to discard for burn-in (default: 0).
%     - `Kmixing`: No. of samples to discard for improving mixing (default: 
%       1).
%
% ## Output
% * `thetas`: The samples.
% * `rate`: Acceptance rate.
%
% ## Authors
% 2017-present -- Roland Hostettler

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

    %% Parameters & Defaults
    Ntheta = size(theta0(:), 1);
    narginchk(2, 5);
    if nargin < 3 || isempty(q)
        % Create a default proposal distribution
        q.rand = @(theta) theta + randn(Ntheta, 1);
        q.logpdf = @(theta_p, theta) 1;
    end
    if nargin < 4 || isempty(K)
        K = 100;
    end
    if nargin < 5
        par = struct();
    end
    
    def = struct(...
        'Kburnin', 0, ...
        'Kmixing', 1, ...
        'show_progress', [] ...
    );
    par = parchk(par, def);
    
    % Total no. of samples to draw from the chain
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;
    Kaccepted = 0;
    
    %% Sampling
    thetas = zeros(Ntheta, Kmcmc);
    theta = theta0;
    lp = p(theta);
    for k = 1:Kmcmc
        % Propose new sample & calcualte acceptance probability
        theta_p = q.rand(theta);
        
        lp_p = p(theta_p);
        lq = q.logpdf(theta, theta_p);
        lq_p = q.logpdf(theta_p, theta);
        alpha = min(1, exp(lq-lq_p + lp_p-lp));

        % Accept with probability alpha, otherwise keep the old value
        u = rand(1);
        if u <= alpha
            theta = theta_p;
            lp = lp_p;
            Kaccepted = Kaccepted + 1;
        end
        thetas(:, k) = theta;
        
        if ~isempty(par.show_progress)
            par.show_progress(k/Kmcmc, thetas(:, 1:k));
        end
    end

    %% Post-processing
    thetas = thetas(:, par.Kburnin+1:par.Kmixing:Kmcmc);
    rate = Kaccepted/Kmcmc;
end
