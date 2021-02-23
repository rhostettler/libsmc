function [xhat, sys] = smcmc(model, y, theta, J, par)
% # Sequential Markov chain Monte Carlo
% ## Usage
% * 
%
% ## Description
% 
%
% ## Input
% * 
%
% ## Output
% * 
%
% ## Authors
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
% * Figure out a good way of making this generic. How and where should we
%   encapsulate the kernel? This can probably be re-integrated into PF, if
%   we combine the par.resample() and par.sample() functions. Then we can
%   also integrate the APF into the PF.
% * Documentation

    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    if nargin < 5
        par = struct();
    end
    def = struct( ...
        'sample', @sample_prior, ...
        'Jburnin', round(0.1*J), ...
        'Jmixing', 1 ...
    );
    par = parchk(par, def);
    
    % Calculate the no. of Monte Carlo iterations
    Jmcmc = par.Jburnin + 1 + (J-1)*par.Jmixing;
    
    %% Initialize
    % Sample initial particles
    x = model.px0.rand(J);
    
    % Since we also store and return the initial state (in 'sys'), a dummy
    % (NaN) measurement is prepended to the measurement matrix so that we 
    % can use consistent indexing in the processing loop.
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta' to the appropriate size, such that we can use
    % 'theta(:, n)' as an argument to the different functions (if not 
    % already expanded).
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];
    
    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    return_sys = (nargout >= 2);
    if return_sys
        sys = initialize_sys(N, dx, J);
        sys(1).x = x;
        sys(1).alpha = 1:J;
        sys(1).rate = 1;
    end
    xhat = zeros(dx, N-1);
    
    %% Process data
    for n = 2:N
        [xp, alphap, qstate] = par.sample(model, y(:, n), x, theta(:, n), Jmcmc);

        %% Post-processing
        % Remove burn-in and mixing
        j = 1+(par.Jburnin+1:par.Jmixing:Jmcmc);
        alpha = alphap(:, j);
        x = xp(:, j);

        % Point estimate (MMSE)
        xhat(:, n-1) = mean(x, 2);
        
        %% Store
        if return_sys
            sys(n).alpha = alpha;
            sys(n).x = x;
            sys(n).qstate = qstate;
        end
    end
    
    %% Calculate joint filtering density
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end

%%
function [xn, alphan, qstate] = sample_prior(model, y, x, theta, Jmcmc)
% TODO: Move this out of smcmc

    [dx, J] = size(x);
    xn = zeros(dx, Jmcmc);
    
    %% Chain initialization
    % Variables:
    % * xn/alphan: Accepted states for x[n] and ancestor indices alpha[n].
    % * xp/alphap: Proposed states and ancestor indices.
    % * x contains the samples from the previous iteration (n-1)

    % Reset the number of accepted samples
    naccept = 0;

    % Sample chain initialization
    alphap = randi(J, 1);
    xn(:, 1) = model.px.rand(x(:, alphap), theta);
    alphan(:, 1) = alphap;
    lpy = model.py.logpdf(y, xn(:, 1), theta);

    %% Sample from the chain
    % N.B.: From 2 to +1 b/c of the xn(:, 1) containing the initial 
    % value
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
            alphan(:, j) = alphap;
            lpy = lpyp;
            naccept = naccept + 1;
        else
            xn(:, j) = xn(:, j-1);
            alphan(:, j) = alphan(:, j-1);
        end
    end
    
    qstate = struct('rate', naccept/Jmcmc);
end
