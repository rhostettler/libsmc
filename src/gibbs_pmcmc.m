function [xs, thetas, sys] = gibbs_pmcmc(model, y, theta0, lambda, K, par)
% Particle Gibbs Markov chain Monte Carlo sampler
%
% SYNOPSIS
%   [x, theta] = GIBBS_PMCMC(y, t, model, K, theta0, par)
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
%       Series B (Statistical Methodology), vol. 72, no. 3, pp. 269-342, 
%       2010.
%
%   [2] F. Lindsten, M. I. Jordan, and T. B. Schon, "Particle Gibbs with
%       ancestor sampling," Journal of Machine Learning Research, vol. 15, 
%       pp. 2145-2184, 2014.
% 
% AUTHORS
%   2017-2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * There's still confusion about using both 'create_model' and theta in
%     the different methods. That should be sorted out somehow by the model
%     things
%   * If there are no parameters to sample, thetas() can not be used as an
%     input to sample_states.
%   * Update docs
%   * Update interface of sample_parameters

    %% Defaults
    narginchk(2, 6);
    if nargin < 3
        theta0 = [];
    end
    if nargin < 4
        lambda = [];
    end
    if nargin < 5 || isempty(K)
        K = 10;
    end
    if nargin < 6
        par = struct();
    end
    
    % Default parameters
    def = struct(...
        'Kburnin', 0, ...
        'Kmixing', 1, ...
        'sample_states', @(y, xtilde, theta, lambda) cpfas(model(theta(:, end)), y, xtilde, lambda), ...
        'sample_parameters', [], ...
        'show_progress', [] ...
    );
    par = parchk(par, def);
    
    % Calculate no. of runs
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;

    %% Initialize    
    % State of Metropolis-within-Gibbs sampler
    state = [];
    
    % Preallocate
    x = [];
    xs = [];
    if isempty(theta0)
        theta0 = NaN;
    end
    thetas = theta0*ones(1, Kmcmc+1);

    %% MCMC sampling
    for k = 2:Kmcmc+1
        % Sample trajectory
        % TODO: More hacks here...
        if ~isempty(par.sample_states)    
            [x, tmp] = par.sample_states(y, x, thetas(:, 1:k-1), lambda);
            if k == 2
                xs = zeros([size(x), Kmcmc+1]);
                sys = cell(1, Kmcmc+1);
            end
            xs(:, :, k) = x;
            sys{k} = tmp;
        else
            error('No state sampling method provided.');
        end
        
        % Sample parameters
        % TODO: Update interaface on sample_parameters
        if ~isempty(par.sample_parameters)
            [thetas(:, k), state] = par.sample_parameters(y, lambda, xs(:, :, k), thetas(:, 1:k-1), model, state);
        end
        
        % Show progress
        if ~isempty(par.show_progress)
            par.show_progress((k-1)/Kmcmc, xs(:, :, 1:k), thetas(:, 1:k));
        end
    end
    
    %% Post-processing
    % Strip burn-in, and mixing
    xs = xs(:, :, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    if ~isempty(thetas)
        thetas = thetas(:, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
    if nargout >= 3
        sys = sys(par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
end
