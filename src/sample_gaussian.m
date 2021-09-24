function [xp, alpha, lqx, lqalpha, qstate] = sample_gaussian(model, y, x, lw, theta, par)
% # Gaussian approximation to optimal importance density
% ## Usage
% * `[xp, alpha] = sample_gaussian(model, y, x, lw, theta, slr)`
% * `[xp, alpha, lqx, lqalpha, qstate] = sample_gaussian(model, y, x, lw, theta, slr, L, kappa, epsilon)`
%
% ## Description
% Calculates the Gaussian approximation of the joint distribution
%                                    _    _    _  _    _         _
%                               / | x[n] |  | mx |  |  Px   Pxy | \
%     p(x[n], y[n] | x[n-1]) = N\ |_y[n]_|; |_my_|, |_ Pxy' Py _| /
%
% using generalized statistical linear regression. From this, an 
% approximation of the optimal importance density is calculated according
% to
%
%     p(x[n] | y[n], x[n-1]) ~= N(x[n]; mp, Pp)
%
% where
%
%     K = Pxy/Py
%     mp = mx + K*(y - my)
%     Pp = Px - K*Py*K'.
%
% This approximation is then used for sampling new states.
%
% This update step implements also iterated and generalized variants.
% Iterated variants, also called posterior linearization, iteratively
% calculates the importance density by refining the approximation using
% the previous approximation as the linearization density. Generalized
% means that it can be used with likelihoods where the likelihood is not
% defined by a functional mean (e.g., Poisson likelihoods). In this case,
% the conditional moments E{y[n] | x[n]} and Cov{y[n] | x[n]} are used.
%
% Approximative statistical linear regression is used for calculating the 
% moments of the joint approximation. The linearization method has to be
% specified as a parameter (the function handle `slr`).
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `x`: dx-times-J matrix of particles for the state x[n-1].
% * `lw`: Log-weights of x[n-1].
% * `theta`: dtheta-times-1 vector of other parameters.
% * `par`: Additional parameters:
%   - `resample`: Resampling function (default: `resample_ess`).
%   - `L`: Number of iterations (default: `1`).
%   - `kappa`: Tail probability for gating (default: `1`).
%   - `epsilon`: Threshold for KL-convergence criterion (default: `1e-2`).
%   - `slr`: Function to perform the statistical linear regression, e.g.
%     `slr_cf` or `slr_sp` (function handle `@(mp, Pp, theta)`; default: 
%     `@slr_sp`).
%
% ## Output
% * `xp`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lq`: 1-times-J vector of the importance density of the jth sample 
%   `xp`.
% * `lqalpha`: 1-times-J vector of the importance density of the jth
%   ancestor index `alpha`.
% * `qstate`: Sampling algorithm state information (resampling and
%   proposal).
%
% ## Authors
% 2019-present -- Roland Hostettler

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
% * Add the possibility to use this for APF-sampling
    
    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    defaults = struct( ...
        'L', 1, ...                     % No. of iterations
        'kappa', 1, ...                 % Probability for the detection of poor approximations
        'epsilon', 1e-2, ...            % Convergence tolerance
        'slr', @slr_sp, ...             % Use sigma-point-based SLR by default
        'resample', @resample_ess ...   % Resampling function
    );
    par = parchk(par, defaults);
    
    %% Resampling   
    % Sample ancestor indices (resampling)
    [alpha, lqalpha, rstate] = par.resample(lw);
    x = x(:, alpha);
    
    %% Sample
    % Preallocate
    L = par.L;
    [dx, J] = size(x);
    xp = zeros(dx, J);
    lqx = zeros(1, J);
    qj = struct( ...
        'fast', false, ...
        'rand', @(y, x, theta) [], ...
        'logpdf', @(xp, y, x, theta) [], ...
        'mean', [], ...
        'cov', [], ...
        'dkl', [], ...
        'l', [] ...
    );
    qstate = struct('rstate', rstate, 'qj', repmat(qj, [1, J]));
    dkls = zeros(1, L);
    mps = zeros(dx, L+1);
    Pps = zeros(dx, dx, L+1);
        
    % Divergence detection threshold
    gamma = chi2inv(par.kappa, size(y, 1));
    
    % Get mean and covariance functions
    f = model.px.mean;
    Q = model.px.cov;
    
    % For all particles...
    for j = 1:J
        %% Calculate proposal
        % Initialize
        mx = f(x(:, j), theta);
        if isa(Q, 'function_handle')
            Px = Q(x(:, j), theta);
        else
            Px = Q;
        end
        mp = mx;
        Pp = Px;
        Lp = chol(Pp, 'lower');
        mps(:, 1) = mp;
        Pps(:, :, 1) = Pp;

        % Iterations
        l = 0;
        done = false;
        while ~done
            % Update iteration counter
            l = l + 1;
                        
            % Calculate linearization w.r.t. linearization density
            % y = A*x + b + nu, nu ~ N(0, Omega)
            [A, b, Omega] = par.slr(mp, Pp, theta);

            % Moments of y of the joint approximation
            my = A*mx + b;
            Py = A*Px*A' + Omega;
            Py = (Py + Py')/2;
            Pxy = Px*A';

            % Posterior of x given y
            K = Pxy/Py;
            mt = mx + K*(y - my);
            Pt = Px - K*Py*K';
            Pt = (Pt + Pt')/2;
            
            % Check if posterior update was successful, if not, exit loop
            % and use the previous best approximation
            [Lt, nd] = chol(Pt, 'lower');
            if nd || ((y - my)'/Py*(y - my) >= gamma)
                done = true;
                warning('libsmc:warning', 'Posterior approximation failed (l = %d), sampling from prior.', l);
            else
                % Change in KL divergence
                dkls(l) = (trace(Pt\Pp) - log(det(Pp)) + log(det(Pt)) - dx + (mt - mp)'/Pt*(mt - mp))/2;
                done = dkls(l) < par.epsilon;
                
                mp = mt;
                Pp = Pt;
                Lp = Lt;
            end
            
            % Store current linearization
            mps(:, l+1) = mp;
            Pps(:, :, l+1) = Pp;
            
            % Convergence criteria and tests
            done = (l >= L) || done;
        end
        
        %% Sample
        % TODO: Adding mps, Pps, dkls is inconsistent with respect to the 
        % pdf struct, but we'll keep it for now.
        qj = struct( ...
            'fast', false, ...
            'rand', @(y, x, theta) mp + Lp*randn(dx, 1), ...
            'logpdf', @(xp, y, x, theta) logmvnpdf(xp.', mp.', Pp).', ...
            'mean', mps, ...
            'cov', Pps, ...
            'dkl', dkls, ...
            'l', l ...
        );
        xp(:, j) = qj.rand(y, x(:, j), theta);
        lqx(:, j) = qj.logpdf(xp(:, j), y, x(:, j), theta);
        qstate.qj(j) = qj;
    end
end
