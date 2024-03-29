function [x, sys] = cpfas(model, y, xtilde, theta, J, par)
% # Conditional particle filter with ancestor sampling
% ## Usage
% * `x = cpfas(model, y)`
% * `[x, sys] = cpfas(model, y, xtilde, theta, J, par)`
% 
% ## Description
% Conditional particle filter with ancestor sampling (CPF-AS), a
% fast-mixing, Markov kernel invariant over state trajectories x[0:N]. See
% [1,2] for details.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N measurement matrix y[1:N].
% * `xtilde`: dx-times-N seed trajectory xtilde[0:N] (optional, a bootstrap
%   particle filter is used to generate a seed trajectory if omitted).
% * `theta`: Additional parameters (optional).
% * `J`: Number of particles (default: `100`).
% * `par`: Additional algorithm parameters:
%     - `[xp, alpha, lqx, lqalpha, qstate] = sample(model, y, x, lw, theta)`:
%       Function to sample new particles (used for the J-1 particles; 
%       default: `@sample_bootstrap`).
%     - `lw = calculate_weights(model, y, xp, alpha, lqx, lqalpha, x, lw, theta)`: 
%       Function to calculate the incremental particle weights (must match 
%       the sampling function defined above; default: 
%       `@calculate_weights`).
%     - `[alpha, state] = sample_ancestor_index(model, y, xtilde, x, lw, theta)`:
%       Function to sample the ancestor indices (default:
%       `@sample_ancestor_index`).
%
% ## Output
% * `x`: The newly sampled trajectory (dx-times-N).
% * `sys`: Struct of the particle system containing:
%     - `x`: Particles of the marginal filtering density (not complete
%       trajectories).
%     - `w`: Particle weights of the marginal filtering density
%       corresponding to x.
%     - `alpha`: Ancestor indices.
%     - `qstate`: Internal state of the sampling algorithm, see the 
%       corresponding algorithms for details.
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

% TODO
% * Consider merging the non-Markovian version into this function

    %% Defaults
    narginchk(2, 6);
    if nargin < 4 || isempty(theta)
        theta = NaN;
    end
    if nargin < 5 || isempty(J)
        J = 100;
    end

    % Default parameters (importance density, weights, etc.)
    if nargin < 6
        par = struct();
    end
    par_sampler = struct( ...
        'resample', @resample_stratified ...
    );
    def = struct( ...
        'sample', @(model, y, x, lw, theta) sample_bootstrap(model, y, x, lw, theta, par_sampler), ...
        'calculate_weights', @calculate_weights, ...
        'sample_ancestor_index', @sample_ancestor_index ...
    );
    par = parchk(par, def);
    
    %% Initialize seed trajectory
    % If no trajectory is given (e.g., for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which helps to
    % speed up convergence.
    % 
    % **Note**: Better trajectories can be drawn
    % using "better" PFs, but that should be done outside of this function
    % instead.
    if nargin < 3 || isempty(xtilde) || all(all(xtilde == 0))
        % Run bootstrap pf and sample a trajectory according to the filter
        % weights
        [~, tmp] = pf(model, y, theta, J);
        beta = resample_stratified(log(tmp(end).wf));
        j = beta(randi(J, 1));
        xf = cat(3, tmp.xf);
        [dx, ~, N] = size(xf);
        xtilde = reshape(xf(:, j, :), [dx, N]);        
    end
    
    %% Initialize
    % Draw initial particles
    x = model.px0.rand(J-1);
    x(:, J) = xtilde(:, 1);
    lw = log(1/J)*ones(1, J);

    % Prepend a NaN measurement (for x[0] where we don't have a 
    % measurement)
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];

    % Expand theta properly such that we have theta(:, n)
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];

    %% Prepare and preallocate    
    % Determine state size
    dx = size(x, 1);
    N = N+1;
    sys = initialize_sys(N, dx, J);
    sys(1).x = x;
    sys(1).w = exp(lw);
    sys(1).alpha = 1:J;
    sys(1).qstate = struct('rstate', [], 'qj', [], 'astate', []);
    
    %% Iterate over the data
    for n = 2:N
        %% Sampling
        % Resample, then sample J-1 particles and set the Jth to the seed 
        % trajectory
        [xp, alpha, lqx, lqalpha, qstate] = par.sample(model, y, x, lw, theta(:, n));
        xp(:, J) = xtilde(:, n);
        
        % Ancestor index (note: the ancestor weights have to be calculated
        % *inside* the sampling function).
        [alpha(J), astate] = par.sample_ancestor_index(model, y(:, n), xtilde(:, n), x, lw, theta(:, n));
        qstate.astate = astate;
                
        %% Calculate weights
        % Evaulate the proposal density for the seed trajectory sample
        % (this is not done when sampling, for obvious reasons)
        lqx(J) = qstate.qj(alpha(J)).logpdf(xp(:, J), x(:, alpha(J)), theta(:, n));
        lqalpha(J) = lw(alpha(J));

        lw = par.calculate_weights(model, y(:, n), xp, alpha, lqx, lqalpha, x, lw, theta(:, n));
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        x = xp;

        if any(isnan(w)) || any(w == Inf)
            warning('NaN and/or Inf in particle weights.');
        end
        
        %% Store
        sys(n).x = x;
        sys(n).w = w;
        sys(n).alpha = alpha;
        sys(n).qstate = qstate;
    end
    
    %% Sample trajectory
    beta = resample_stratified(log(w));
    j = beta(randi(J, 1));
    tmp = calculate_particle_lineages(sys, j);
    x = cat(2, tmp.xf);
end
