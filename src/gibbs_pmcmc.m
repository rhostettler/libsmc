function [xs, thetas, sys] = gibbs_pmcmc(model, y, theta0, lambda, K, J, par)
% # Particle Gibbs Markov chain Monte Carlo sampler
% ## Usage
% * `x = gibbs_pmcmc(model, y)`
% * `[x, theta, sys] = gibbs_pmcmc(model, y, theta0, lambda, K, J, par)`
%
% ## Description
% Particle Markov chain Monte Carlo Gibbs sampler for generating samples
% from the posterior distribution p(theta, x[0:N] | y[1:N]). The sampler
% uses the Gibbs approach, which samples each parameter condtionally on
% the others in turn. In particular, the method first samples from
%
%     x[0:N] ~ p(x[0:N] | theta, y[1:N])
%
% and then from
%
%     theta ~ p(theta | x[0:N], y[1:N]).
%
% Sampling is achieved by calling two user-defined functions (supplied as
% fields in the 'par' argument):
%
% 1. `sample_states()`, and
% 2. `sample_parameters()`.
%
% If the former is not specified, cpfas is used by default and if the
% latter is not specified, no parameters are sampled.
%   
% ## Input
% * `model`: Function handle of the form @(theta) to construct the state-
%   space model struct. **N.B.**: This is likely to change in the future as
%   parameter handling should not be directed to a "constructor".
% * `y`: dy-times-N measurement matrix.
% * `theta0`: Initial guess of the model parameters (if any; default: 
%   `NaN`).
% * `lambda`: Set of static (known) parameters (default: `[]`).
% * `K`: No. of MCMC samples to generate (default: `10`).
% * `J`: No. of particles to use in the particle filter (default: `100`).
% * `par`: Additional parameters:
%     - `Kburnin`: No. of burn-in samples (removed after sampling; default:
%       `0`).
%     - `Kmixing`: No. of samples for improving the mixing (removed after
%       sampling; default: `1`).
%     - `x = sample_states(model, y, x, lambda)`: Function to sample the 
%       states (default: `@cpfas`).
%     - `[theta, state] = sample_parameters(y, t, x, theta, model, state)`:
%       Function to sample the parameters (default: `[]`). In addition to 
%       the newly sampled parameters, the function may also return a state 
%       variable which stores the sampler's state (useful for adaptive 
%       sampling).
%     - `show_progress(p, x, theta)`: Function to display or otherwise 
%       illustrate the progress (default: `[]`). The parameters are the
%       progress of the sampling in [0,1], and the so-far sampled 
%       trajectories and parameters.
% 
% ## Output
% * `x`: dx-times-N-times-K array of trajectory samples.
% * `theta`: dtheta-times-K matrix of parameter samples.
% * `sys`: Cell array of particle systems. *Warning: This should only be
%   used if absolutely necessary since storing the particle systems takes a
%   lot of memory and may cause Matlab to crash.*
%
% ## References
% 1. C. Andrieu, A. Doucet, and R. Holenstein, "Particle Markov chain
%    Monte Carlo methods," Journal of the Royal Statistical Society: 
%    Series B (Statistical Methodology), vol. 72, no. 3, pp. 269-342, 2010
%
% 2. F. Lindsten, M. I. Jordan, and T. B. Schon, "Particle Gibbs with
%    ancestor sampling," Journal of Machine Learning Research, vol. 15, 
%    pp. 2145-2184, 2014.
% 
% ## Authors
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * There's still confusion about using both 'create_model' and theta in
%   the different methods. That should be sorted out somehow by the model
%   things.
%   => In practice, this is solved by the px0, px, py's which now take a
%   generic set of parameters. Now we only have to concatenate these
%   properly; but make sure to mention how these are concatenated in the
%   documentation.
%   Also note that this method should be agnostic to the 'lambda', i.e., 
%   they should not be part of the interface. Instead, one should pass
%   these to cpfas() directly, e.g. through concatenation or similar. This
%   should be fairly straight forward to implement in practice but might
%   break a lot of things. **SHOULD BE SORTED OUT ASAP** (i.e., when
%   rewriting the GP-PMCMC things. Also requires rewriting
%   example_cpfas_rs_sv.m).
% * Update interface of sample_parameters

    %% Defaults
    narginchk(2, 7);
    if nargin < 3 || isempty(theta0)
        theta0 = NaN;
    end
    if nargin < 4
        lambda = [];
    end
    if nargin < 5 || isempty(K)
        K = 10;
    end
    if nargin < 6 || isempty(J)
        J = 100;
    end
    if nargin < 7
        par = struct();
    end
    
    % Default parameters
    def = struct(...
        'Kburnin', 0, ...
        'Kmixing', 1, ...
        'sample_states', @cpfas, ...
        'sample_parameters', [], ...
        'show_progress', [] ...
    );
    par = parchk(par, def);
    
    % Calculate no. of runs
    Kmcmc = par.Kburnin + 1+(K-1)*par.Kmixing;

    %% Initialize and preallocate
    % Initialize state trajectory and state of parameter sampler (both
    % empty). The later is used for, for example, adaptive proposals.
    x = [];
    state = [];
    
    % Preallocate output matrices/structures
    tmp = model(theta0);
    dx = size(tmp.px0.rand(1), 1);
    N = size(y, 2) + 1;
    
    xs = zeros(dx, N, Kmcmc+1);         % Sampled trajectories
    thetas = theta0*ones(1, Kmcmc+1);   % Sampled parameters    
    
    return_sys = (nargout >= 3);        % Cell array of particle systems.
    if return_sys                       % Only used when explicitly 
        sys = cell(1, Kmcmc+1);         % requested to avoid Matlab 
    end                                 % crashes.

    %% MCMC sampling
    for k = 2:Kmcmc+1
        % Sample trajectory
        if ~isempty(par.sample_states)
            model_k = model(thetas(:, k-1));
            if return_sys
                [x, sys{k}] = par.sample_states(model_k, y, x, lambda, J);
            else
                x = par.sample_states(model_k, y, x, lambda, J);
            end
            xs(:, :, k) = x;
        else
            error('No state sampling method provided.');
        end
        
        % Sample parameters
        % TODO: Update interface on sample_parameters
        if ~isempty(par.sample_parameters)
            [thetas(:, k), state] = par.sample_parameters(y, lambda, xs(:, :, k), thetas(:, 1:k-1), model, state);
        end
        
        % Show progress
        if ~isempty(par.show_progress)
            par.show_progress((k-1)/Kmcmc, xs(:, :, 1:k), thetas(:, 1:k));
        end
    end
    
    %% Post-processing
    % Strip burn-in and mixing
    xs = xs(:, :, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    if ~isempty(thetas)
        thetas = thetas(:, par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
    if return_sys
        sys = sys(par.Kburnin+2:par.Kmixing:Kmcmc+1);
    end
end
