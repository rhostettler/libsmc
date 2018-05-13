function [x, theta, sys] = gibbs_pmcmc(y, t, model, theta0, K, par)
% Particle Gibbs Markov chain Monte Carlo sampler
%
% SYNOPSIS
%   [x, theta] = gibbs_pmcmc(y, t, model, K, theta0, par)
%
% DESCRIPTION
%   Particle Markov chain Monte Carlo Gibbs sampler for generating samples
%   from the posterior distribution p(theta, x[0:N] | y[1:N]). The sampler
%   uses the Gibbs approach, which samples each parameter condtionally on
%   the others in turn. In particular, the method first samples from
%
%       x[0:N] ~ p(x[0:N] | theta, y[1:N])
%
%   and then from
%
%       theta ~ p(theta | x[0:N], y[1:N]).
%
%   Sampling is achieved by calling two user-defined functions (supplied as
%   fields in the 'par' argument):
%
%       1. sample_states(), and
%       2. sample_parameters().
%
%   If the former is not specified, cpfas is used by default and if the
%   latter is not specified, no parameters are sampled.
%   
% PARAMETERS
%   y       Measurement data matrix Ny*N
%   t       Time vector (default: 1:N)
%   model(theta)
%           Function handle to construct the model, takes the parameter
%           values theta as an argument.
%
%   K       No. of MCMC samples to generate (optional, default: 100)
%   theta0  Initial guess of the model parameters
%   par     Additional parameters:
%
%               Kburnin No. of burn-in samples (default: 0)
%               Kmixing No. of samples for improving the mixing
%                       (default: 1)
%
%               x = sample_states(y, t, x, theta, model)
%                       Function to sample the states (default: cpfas).
% 
%               [theta, state] = sample_parameters(y, t, x, theta, model, state)
%                       Function to sample the model parameters (default:
%                       []). In addition to the newly sampled parameters,
%                       the function may also return a state variable which
%                       stores the sampler's state.
%
%               show_progress(p, x, theta)
%                       Function to display or otherwise illustrate the
%                       progress (default: []). The parameters are the
%                       progress of the sampling in [0,1], and the so-far
%                       sampled trajectories and parameters.
% 
% RETURNS
%   x       Trajectory samples (Nx times N times K)
%   theta   Parameter samples (Ntheta times K)
%
% SEE ALSO
%   cpfas, rb_cpfas
% 
% REFERENCES
%   [1] C. Andrieu, A. Doucet, and R. Holenstein, "Particle Markov chain
%       Monte Carlo methods," Journal of the Royal Statistical Society: 
%       Series B (Statistical Methodology), vol. 72, no. 3, pp. 269???342, 
%       2010.
%
%   [2] F. Lindsten, M. I. Jordan, and T. B. Schon, "Particle Gibbs with
%       ancestor sampling," Journal of Machine Learning Research, vol. 15, 
%       pp. 2145-2184, 2014.
% 
% AUTHORS
%   2017-10-04 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * There's still confusion about using both 'create_model' and theta in
%     the different methods. That should be sorted out somehow by the model
%     things
%   * Returning particle system is broken-ish

    %% Defaults
    narginchk(3, 6);
    if nargin < 4
        theta0 = [];
    end
    if nargin < 5 || isempty(K)
        K = 10;
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'Kburnin', 0, ...
        'Kmixing', 1, ...
        'sample_states', @(y, t, x, ~, ~) cpfas(y, t, model, x), ...
        'sample_parameters', [], ...
        'show_progress', [] ...
    );
    par = parchk(par, def);
    
    % Calculate no. of runs
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;

    %% Initialize
    t = [0, t];
    
    % State of Metropolis-within-Gibbs sampler
    state = [];
    
    % Preallocate
    Ntheta = size(theta0, 1);
    x = [];
    theta = [theta0, zeros(Ntheta, Kmcmc)];

    %% MCMC sampling
    for k = 2:Kmcmc+1
        % Sample trajectory
        % TODO: sys should be handled correctly!
        if ~isempty(par.sample_states)
            [xtmp, tmp] = par.sample_states(y, t, x(:, :, k-1), theta(:, 1:k-1), model(theta(:, k-1)));
            if k == 2
                x = zeros([size(xtmp), Kmcmc+1]);
                sys = repmat(tmp, [1, Kmcmc+1]);
            end
            sys(:, k) = tmp;
            x(:, :, k) = xtmp;
        else
            error('No state sampling method provided.');
        end
        
        % Sample parameters
        if ~isempty(par.sample_parameters)
            [theta(:, k), state] = par.sample_parameters(y, t, x(:, :, k), theta(:, 1:k-1), model, state);
        else
            theta(:, k) = theta(:, k-1);
        end
        
        % Show progress
        if ~isempty(par.show_progress)
            par.show_progress((k-1)/Kmcmc, x(:, :, 1:k), theta(:, 1:k));
        end
    end
    
    %% Post-processing
    % Strip burn-in, and mixing
    x = x(:, :, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    sys = sys(:, par.Kburnin+2:Kmcmc+1);
    if ~isempty(theta)
        theta = theta(:, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
end
