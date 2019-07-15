function [xp, q] = sample_gaussian_cf(model, y, x, theta, f, Q, Ey, Cy, Cyx, L)
% # Gaussian approximation to optimal importance density (closed-form)
% ## Usage
% * `[xp, q] = sample_gaussian_cf(model, y, x, theta, f, Q, Ey, Cy, Cyx)`
% * `[xp, q] = sample_gaussian_cf(model, y, x, theta, f, Q, Ey, Cy, Cyx, L)`
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
% This approximation requires closed-form expressions for the moments 
% E{y}, Cov{y}, and Cov{y, x}, that is, no Taylor series approximation or 
% sigma-point integration is used. See below for these approximations.
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
% * `Ey`: Mean E{y} (function handle @(m, P, theta)).
% * `Cy`: Covariance Cov{y} (function hanlde @(m, P, theta)).
% * `Cyx`: Covariance Cov{y, x} (function handle @(m, P, theta)).
% * `L`: Number of iterations (default: 1).
%
% ## Output
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

    %% Defaults
    narginchk(9, 10);
    [dx, J] = size(x);
    if nargin < 10 || isempty(L)
        L = 1;
    end

    %% Sample and calculate incremental weights
    % Preallocate
    xp = zeros(dx, J);
    qj = struct('fast', false, 'rand', @(y, x, theta) [], 'logpdf', @(xp, y, x, theta) [], 'mp', [], 'Pp', []);
    q = repmat(qj, [1, J]);
        
    % For all particles...
    for j = 1:J
        %% Calculate proposal    
        % Initialize
        mx = f(x(:, j), theta);
        Px = Q(x(:, j), theta);
        mp = mx;
        Pp = Px;
        Lp = chol(Pp, 'lower');
        
        % Iterations
        l = 0;
        done = false;
        while ~done
            % Calculate linearization w.r.t. linearization density
            % y = A*x + b + nu, nu ~ N(0, Omega)
            A = Cyx(mp, Pp, theta)/Pp;
            b = Ey(mp, Pp, theta) - A*mp;
            Omega = Cy(mp, Pp, theta) - A*Pp*A';

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
            if nd
                done = true;
                warning('libsmc:warning', 'Posterior approximation failed, sampling from prior.');
            else
                done = false;
                mp = mt;
                Pp = Pt;
                Lp = Lt;
            end
            
            l = l + 1;            
            done = (l >= L) || done;
        end
        
        %% Sample
        qj = struct( ...
            'fast', false, ...
            'rand', @(y, x, theta) mp + Lp*randn(dx, 1), ...
            'logpdf', @(xp, y, x, theta) logmvnpdf(xp.', mp.', Pp).', ...
            ... % TODO: This is inconsistent, but we'll keep it for now.
            'mp', mp, ...   
            'Pp', Pp ...
        );
        xp(:, j) = qj.rand(y, x(:, j), theta);
        q(j) = qj;
    end
end
