function [xp, lv] = sis_update_gaussian_sp(y, x, theta, model, f, Q, g, R, L, Xi, wm, wc)
% Gaussian Approximation to Optimal Importance Distribution w/ Sigma-Points
%
% USAGE
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_SP(y, x, theta, model, f, Q, g, R)
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_SP(y, x, theta, model, f, Q, g, R, L, Xi, wm, wc)
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
% PARAMETERS
%   y       Measurement y[n]
%   x       State x[n-1]
%   theta   Other parameters (e.g., time).
%   f       Mean of the dynamic model E{x[n] | x[n-1]} (function handle 
%           @(x, theta))
%   Q       Covariance of the dynamic model Cov{x[n] | x[n-1]} (function 
%           handle @(x, theta))
%   g       Mean of the likelihood E{y[n] | x[n]} (function handle @(x, 
%           theta))
%   R       Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%           @(x, theta))
%   L       Number of iterations (optional, default: 1)
%   Xi      Unit sigma-points (optional, default: cubature rule)
%   wm      Mean sigma-point weights (optional, default: cubature rule)
%   wc      Covariance sigma-point weights (optional, default: cubature
%           rule)
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
%   2018-2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * When Xi is specified, at least wm should be too (and vice-versa)
%   * Defaults for sigma-points are missing
%   * Add 'gamma' as a parameter

    %% Defaults
    narginchk(8, 12);
    [Nx, J] = size(x);
    Ny = size(y, 1);
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
    gamma = chi2inv(1, Ny);

    %% Sample and calculate incremental weights
    % Preallocate
    xp = zeros(Nx, J);
    lv = zeros(1, J);
    I = length(wm);
    Y = zeros(Ny, I);
    
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
            % Generate sigma-points
            X = mp*ones(1, I) + Lp*Xi;

            % Calculate expectations w.r.t. linearziation density
            Ey = zeros(Ny, 1);              % E{y}
            Ey2 = zeros(Ny, Ny);            % E{y*y'}
            EVy_x = zeros(Ny, Ny);          % E{V{y|x}}
            Eyx = zeros(Ny, Nx);            % E{y*x'}
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
            
            l = l + 1;            
            done = (l >= L) || done;
        end
        
        %% Sample and calculate incremental weight
        xp(:, j) = mp + Lp*randn(Nx, 1);
        py = model.py;
        px = model.px;
        lv(j) = ( ...
            py.logpdf(y, xp(:, j), theta) + px.logpdf(xp(:, j), x(:, j), theta) ...
            - logmvnpdf(xp(:, j).', mp.', Pp.').' ...
        );
    end
end
