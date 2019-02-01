function [xp, lv] = sis_update_gaussian_cf(y, x, theta, model, f, Q, Ey, Cy, Cyx, L)
% Gaussian Approximation to Optimal Importance Distribution (Closed-Form)
%
% USAGE
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_CF(y, x, theta, model, f, Q, Ey, Cy, Cyx)
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_CF(y, x, theta, model, f, Q, Ey, Cy, Cyx, L)
%
% DESCRIPTION
%   Calculates the Gaussian approximation of the joint distribution
%                                    _    _    _  _    _         _
%                                 / | x[n] |  | mx |  |  Px   Pxy | \
%       p(x[n], y[n] | x[n-1]) = N\ |_y[n]_|; |_my_|, |_ Pxy' Py _| /
%
%   using generalized statistical linear regression and sigma-points. From 
%   this, an approximation of the optimal importance density is calculated
%   according to
%
%       p(x[n] | y[n], x[n-1]) ~= N(x[n]; mp, Pp)
%
%   where
%
%       K = Pxy/Py
%       mp = mx + K*(y - my)
%       Pp = Px - K*Py*K'.
%
%   This approximation is then used for sampling new states.
%
%   This update step implements also iterated and generalized variants.
%   Iterated variants, also called posterior linearization, iteratively
%   calculates the importance density by refining the approximation using
%   the previous approximation as the linearization density. Generalized
%   means that it can be used with likelihoods where the likelihood is not
%   defined by a functional mean (e.g., Poisson likelihoods). In this case,
%   the conditional moments E{y[n] | x[n]} and Cov{y[n] | x[n]} are used.
%
%   This approximation requires closed-form expressions for the moments 
%   E{y}, Cov{y}, and Cov{y, x}, that is, no Taylor series approximation or 
%   sigma-point integration is used. See below for these approximations.
%
% PARAMETERS
%   y       Measurement y[n]
%   x       State x[n-1]
%   theta   Other parameters (e.g., time).
%   f       Mean of the dynamic model E{x[n] | x[n-1]} (function handle 
%           @(x, theta))
%   Q       Covariance of the dynamic model Cov{x[n] | x[n-1]} (function 
%           handle @(x, theta))
%   Ey      Mean E{y} (function handle @(m, P, theta))
%   Cy      Covariance Cov{y} (function hanlde @(m, P, theta))
%   Cyx     Covariance Cov{y, x} (function handle @(m, P, theta))
%   L       Number of iterations (optional, default: 1)
%
% RETURNS
%   xp      New samples.
%   lv      Log of the incremental weights, that is,
%
%               lv = log(p(y[n] | x[n])) + log(p(x[n] | x[n-1]))
%                   - log(q(x[n])),
%
%           where q(x[n]) is the proposal.
%
% AUTHOR
%   2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(9, 10);
    [Nx, J] = size(x);
    if nargin < 10 || isempty(L)
        L = 1;
    end

    %% Sample and calculate incremental weights
    % Preallocate
    xp = zeros(Nx, J);
    lv = zeros(1, J);
    
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
        
        %% Sample and calculate incremental weight
        % Sample
        xp(:, j) = mp + Lp*randn(Nx, 1);
        
        % Incremental importance weight
        py = model.py;
        px = model.px;
        lv(j) = ( ...
            py.logpdf(y, xp(:, j), theta) + px.logpdf(xp(:, j), x(:, j), theta) ...
            - logmvnpdf(xp(:, j).', mp.', Pp.').' ...
        );
    end
end
