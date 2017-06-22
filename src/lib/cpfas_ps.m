function [xhat, sys] = cpfas_ps(y, t, model, q, M, K, par)
% Markov chain Monte Carlo smoothing based on the CPF-AS algorithm
%
% SYNOPSIS
%   xhat = cpfas_ps(y, t, model)
%   [xhat, Phat, sys] = cpfas_ps(y, t, model, q, M, K, par)
% 
% DESCRIPTION
%
% PARAMETERS
%   y   Measurement data matrix where each column is a measurement vector.
%
%   t   Time vector (optional, default: 1:N)
%
%   model
%       State-space model structure, containing the following the
%       probabilistic model representation with structures px0, px, and py
%       for the initial, state transition, and measurement densities,
%       respectively. The structures are:
%           
%       px0.rand(M)
%           Random initial state generator.
%
%       px.fast
%           Boolean variable indicating whether 'rand()' and 'pdf()'/
%           'logpdf()' can evaluate the complete particle set through one 
%           function call.
%
%       px.rand(x, t)
%           Function to draw samples from p(x[n] | x[n-1]).
%
%       px.pdf(xp, x, t)
%           Evaluates the PDF p(x[n] | x[n-1]).
%
%       px.logpdf(xp, x, t)
%           Evaluates the log-PDF log(p(x[n] | x[n-1])).
%
%       py.fast
%           Boolean variable indicating whether 'pdf()' and 'logpdf()' can
%           evaluate the complete particle set through one function call.
%
%       py.pdf(y, x, t)
%           Evaluates the likelihood p(y[n] | x[n]).
%
%       py.logpdf(y, x, t)
%           Evaluates the log-likelihood log(p(y[n] | x[n])).
%
%   q   Proposal distribution from which to draw new particles (default: px
%       (bootstrap)). Structure containing the following functions:
%
%       q.fast
%           Boolean variable indicating whether 'rand()' and
%           'pdf()'/'logpdf()' can handle complete particle sets at once.
%
%       q.rand(y, x, t)
%           Draw samples from the proposal q(x[n] | y[n], x[n-1]).
%
%       q.pdf(xp, y, x, t)
%           Evaluates the PDF q(x[n] | y[n], x[n-1]).
%
%       q.logpdf(xp, y, x, t)
%           Evaluates the log-PDF log(q(x[n] | y[n], x[n-1]).
%
%   M   Number of particles to use in the CPF-AS algorithm, default: 100.
%
%   K   Number of trajectoriy draws, default: 5.
%
%   par Additional parameters
%
% 
% RETURNS
%
%
% REFERENCES
%
% SEE ALSO
%   cpfas
%
% VERSION
%   2017-04-07
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Documentation
%   * Adjust so that we can omit q (=> bootstrap proposal)
%   * Calculate Phat
%   * xs does not adhere to the indexing order used in the other particle
%     filters (i.e. rather than Nx, M, N it is Nx, N, M here.)
%   * Move to pgas as the default filter (cpfas will be removed soon)
%   * Accommodate 0:N rather than 1:N (but we still only return 1:N)
%   * Clean up

    %% Defaults
    narginchk(4, 7);
    N = size(y, 2);
    if isempty(t)
        t = 1:N;
    end
    if nargin < 5 || isempty(M)
        M = 100;
    end
    if nargin < 6 || isempty(K)
        K = 5;
    end
    if nargin < 7
        par = [];
    end
    
    % Default parameters
    def = struct( ...
        'filter', @(y, t, model, q, M, par) cpfas(y, t, model, q, M, par), ...
        'xt', [], ...       % Default trajectory  
        'Kburnin', 10, ...   % No. of burn-in samples
        'Kmixing', 2 ...    % No. of mixing (picks every Nmixing-th trajectory)
    );
    par = parchk(par, def);

    %% Preallocation
    % No. of required MCMC iterations
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;
    Nx = size(model.px0.rand(1), 1);
    
    % xs stores the MCMC trajectories
    xs = zeros(Nx, N, Kmcmc);

    %% Estimation
    for k = 1:Kmcmc
        % Sample a trajectory
        xs(:, :, k) = par.filter(y, t, model, q, M, par);
        par.xt = xs(:, :, k);
    end

    %% Remove Burn-in & Improve Mixing
    % Mixing: We choose every Nth trajectory, e.g. if Nmixing = 10,
    % there are 9 trajectories discarded.
    xs = xs(:, :, par.Kburnin+1:Kmcmc);
    xs = xs(:, :, 1:par.Kmixing:Kmcmc-par.Kburnin);
    xhat = mean(xs, 3);
    
    if nargout >= 2
        sys.xs = xs;
    end
end
