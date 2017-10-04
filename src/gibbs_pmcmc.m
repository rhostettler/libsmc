function [x, theta, sys] = gibbs_pmcmc(y, t, create_model, K, theta0, par)
% Particle Gibbs Markov chain Monte Carlo
%
% SYNOPSIS
%   [x, theta] = gibbs_pmcmc(y, t, model, K, theta0, par)
%
% DESCRIPTION
%   Particle Markov chain Monte Carlo-based 
%   
% PARAMETERS
%   y       Measurement data matrix Ny*N
%
%   t       Time vector (default: 1:N)
%
%   model   Function handle to construct the model; takes one argument, the
%           parameters theta = [theta_f, theta_g]
%
%   K       No. of MCMC samples to generate (optional, default: 100)
%
%   theta0
%           Initial guess of the model parameters
%
%   par     Struct of additional parameters:
%
%               Kburnin         No. of burn-in samples (default: 0)
%               Kmixing         No. of samples for improving the mixing
%                               default: 1)
%               M                   No. of samples to use in the particle
%                                   filter (default: 100)
%               sample_states       Function to sample the states (default:
%                                   cpfas)
%               sample_parameters   Function to sample the model parameters
%                                   (default: []).
%
%           The interface for 'sample_states' is
%
%               sample_states(y, t, model, M).
%
%           The interface for 'sample_theta_f' and 'sample_theta_y' is
%
%               sample_theta_X()
% 
% RETURNS
%   x       Trajectory samples
%
%   theta   Parameter samples
%
% SEE ALSO
%   cpfas, rb_cpfas
% 
% REFERENCES
%   [1] our new paper
%
%   [2] pgas paper
%
% VERSION
%   2017-10-04
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Documentation update (complete the section on the parameter sampling
%     functions and mention that they are given the complete sample
%     trajectories)
%   * Clean up internal interfaces and parameter handling. A super mess
%     now.
%   * Lots of dirty hacks here; need to clean it up and make generic
%   * See the various TODO's throughout the code.
%   * Make generic, i.e. move the sample_trajectory function out of the
%     general sampler as well
%   * This could potentially be made quite general -- Call it something
%     like gibbs_pmcmc that takes a single function for sampling the
%     parameters (par.sample_theta) and one for sampling the states
%     (par.sample_states). Then we can use it virtually for everything.
%     Make it so that the state and parameter sampling functions don't take
%     a 'par' structure here so we can clean up the parameters structure as
%     well and have no confilicting fields.
%   * Include an output function that is called upon every iteration for
%     progress monitoring

    % TODO: Make this some kind of switch or rather a parameter that we
    % supply. This should make the hole sampler really flexible; even
    % better: We take the state-sampling function from the parameters as
    % well!
    rb = 1;

    %% Defaults
    % TODO: Needs to be updated once the interface is finalized
    narginchk(3, 6);
    if nargin < 4 || isempty(K)
        K = 10;
    end
    if nargin < 5
        theta0 = [];
    end
    if nargin < 6
        par = [];
    end
    
    % Default parameters
    def = struct(...
        'Kburnin', 0, ...
        'Kmixing', 1, ...
        'M', 100, ...
        'sample_states', ...
        'sample_parameters', [] ...
    );
    par = parchk(par, def);
    
    % Calculate no. of runs
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;

    %% Initialize
    % State of Metropolis-within-Gibbs sampler
    % TODO: This is not handled very nicely.
    state = [];
    
    % Preallocate
    model = create_model([theta0_f; theta0_y]);
    Nx = size(model.px0.rand(1), 1);
    N = size(y, 2);
    Ntheta = size(theta0, 1);
    theta = [theta0_f, zeros(Ntheta, Kmcmc)];
    if rb
        x = zeros(Nx, N+1, Kmcmc);
    else
        x = zeros(Nx, N, Kmcmc);
    end

    %% MCMC sampling
    for k = 2:Kmcmc+1
        % Sample trajectory
        % TODO: Take state sampling function from par
        if rb
        x(:, :, k) = sample_states(y, x(:, :, k-1), t, theta_f(:, k-1), theta_y(:, k-1), create_model, par);
        else
        x(:, :, k) = sample_trajectory(y, x(:, :, k-1), t, theta_f(:, k-1), theta_y(:, k-1), create_model, par);
        end
        
        % Sample parameters
        if ~isempty(par.sample_parameters)            
            % TODO: Move out of function somehow
            tt = [0, t];
            [theta(:, k), state] = par.sample_theta_f(y, x(:, :, k), tt, theta_f(:, 1:k-1), theta_y(:, 1:k-1), create_model, state, par);
        end
    end
    
    %% Post-processing
    % Strip initial values, burn-in, and mixing
    if rb
        x = x(:, 2:N+1, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    else
        x = x(:, :, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
    if ~isempty(theta)
        theta = theta(:, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
end

%% Draws a new state trajectory
% TODO: This can be moved out once we take the filter from the parameters;
%       this requires a slight update of cpfas and (?) to include the
%       initial state. While talking about that... CPFAS should probably be
%       renamed to something else; maybe just pgas
function x = sample_states(y, x, t, theta_f, theta_y, create_model, par)
    % Crate a new model with updated parameters, set seed trajectory, and
    % sample a trajectory
    model = create_model([theta_f; theta_y]);
    par.xt = x;
    x = gprbpgas(y, t, model, par.M, par);
end

%% Draws a new state trajectory
% TODO: This is the old code, non-Rao-Blackwellized; will be removed sooner
% or later.
function x = sample_trajectory(y, x, t, theta_f, theta_y, create_model, par)
    % Crate a new model with updated parameters and set seed trajectory
    model = create_model([theta_f; theta_y]);
    par.xt = x;
    
    % Run CPF & draw trajectory
    [~, ~, sys] = cpfas(y, t, model, [], par.M, par);    
    beta = sysresample(sys.wf);
    j = beta(randi(par.M, 1));
    x = squeeze(sys.xf(:, j, :));
end
