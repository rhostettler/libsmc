function [xp, q] = sample_gaussian_sp(model, y, x, theta, f, Q, g, R, L, Xi, wm, wc)
% # Gaussian approximation to optimal importance density w/ sigma-points
% ## Usage
% * `[xp, q] = sample_gaussian_sp(model, y, x, theta, f, Q, g, R)`
% * `[xp, q] = sample_gaussian_sp(model, y, x, theta, f, Q, g, R, L, Xi, wm, wc)`
%
% ## Description
% Calculates the Gaussian approximation of the joint distribution
%                                    _    _    _  _    _         _
%                               / | x[n] |  | mx |  |  Px   Pxy | \
%     p(x[n], y[n] | x[n-1]) = N\ |_y[n]_|; |_my_|, |_ Pxy' Py _| /
%
% using generalized statistical linear regression and sigma-points. From 
% this, an approximation of the optimal importance density is calculated
% according to
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
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `x`: dx-times-J matrix of particles for the state x[n-1].
% * `theta`: dtheta-times-1 vector of other parameters.
% * `f`: Mean of the dynamic model E{x[n] | x[n-1]} (function handle 
%   @(x, theta)).
% * `Q`: Covariance of the dynamic model Cov{x[n] | x[n-1]} (function
%   handle @(x, theta)).
% * `g`: Mean of the likelihood (function handle @(m, P, theta)).
% * `R`: Covariance of the likelihood (function hanlde @(m, P, theta)).
% * `L`: Number of iterations (default: 1).
% * `Xi`: Unit sigma-points (default: cubature rule).
% * `wm`: Mean sigma-point weights (default: cubature rule).
% * `wc`: Covariance sigma-point weights (default: cubature rule).
%
% ## Ouptut
% * `xp`: New samples.
% * `q`: Array of structs where the jth entry corresponds to the importance
%   density of the jth particle.
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
%   * When Xi is specified, at least wm should be too (and vice-versa)
%   * Defaults for sigma-points are missing
%   * Add 'gamma' as a parameter

    %% Defaults
    narginchk(8, 12);
    [dx, J] = size(x);
    dy = size(y, 1);
    if nargin < 9 || isempty(L)
        L = 1;
    end
    if nargin < 10 || isempty(Xi)
        error('Please specify the sigma-points.');
    end
    if nargin < 11 || isempty(wm)
        error('Please specify the sigma-point weights.');
    end
    if nargin < 12 || isempty(wc)
        warning('libsmc:warning', 'Using the same weights for mean and covariance');
        wc = wm;
    end
    
%     gamma = chi2inv(0.99, Ny);
    gamma = chi2inv(1, dy);

    %% Sample and calculate incremental weights
    % Preallocate
    xp = zeros(dx, J);
    lv = zeros(1, J);
    I = length(wm);
    Y = zeros(dy, I);
    qj = struct('fast', false', 'rand', @(y, x, theta) [], 'logpdf', @(xp, y, x, theta) [], 'mp', [], 'Pp', []);
    q = repmat(qj, [1, J]);
    mps = zeros(dx, L+1);
    Pps = zeros(dx, dx, L+1);
    
    % For all particles...
    for j = 1:J
        %% Calculate proposal    
        % Initialize
        mx = f(x(:, j), theta);
        Px = Q(x(:, j), theta);
        mp = mx;
        Pp = Px;
        Lp = chol(Pp, 'lower');
        mps(:, 1) = mp;
        Pps(:, :, 1) = Pp;

        % Iterations
        l = 0;
        done = false;
        while ~done
            % Generate sigma-points
            X = mp*ones(1, I) + Lp*Xi;

            % Calculate expectations w.r.t. linearziation density
            Ey = zeros(dy, 1);              % E{y}
            Ey2 = zeros(dy, dy);            % E{y*y'}
            EVy_x = zeros(dy, dy);          % E{V{y|x}}
            Eyx = zeros(dy, dx);            % E{y*x'}
            for i = 1:I
                Y(:, i) = g(X(:, i), theta);
                Ey = Ey + wm(i)*Y(:, i);
                Ey2 = Ey2 + wc(i)*(Y(:, i)*Y(:, i)');            
                EVy_x = EVy_x + wc(i)*R(X(:, i), theta);
                Eyx = Eyx + wc(i)*(Y(:, i)*X(:, i)');
            end

            % Calculate (co)variances w.r.t. linearization density
            Vy = Ey2 - (Ey*Ey') + EVy_x;    % V{y}
            Vy = (Vy + Vy')/2;
            Cyx = Eyx - (Ey*mp');           % C{y,x}

            % Calculate linearization w.r.t. linearization density
            % y = A*x + b + nu, nu ~ N(0, Omega)
            A = Cyx/Pp;
            b = Ey - A*mp;
            Omega = Vy - A*Pp*A';

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
                done = false;
                mp = mt;
                Pp = Pt;
                Lp = Lt;
            end
            
            % Store current linearization
            mps(:, l+2) = mp;
            Pps(:, :, l+2) = Pp;
            
            % Convergence criteria and tests
            l = l + 1;
            done = (l >= L) || done;
        end
        
        %% Sample and calculate incremental weight
        qj = struct( ...
            'fast', false, ...
            'rand', @(y, x, theta) mp + Lp*randn(dx, 1), ...
            'logpdf', @(xp, y, x, theta) logmvnpdf(xp.', mp.', Pp).', ...
            ... % TODO: Adding mps and Pps is inconsistent, but we'll keep it for now.
            'mp', mps, ...
            'Pp', Pps ...
        );
        xp(:, j) = qj.rand(y, x(:, j), theta);
        q(j) = qj;
    end
end
