function [xp, lv] = sis_update_gaussian_taylor(y, x, theta, model, f, Q, g, Gx, R, L)
% SIS Update w/ Gauss-Taylor Approximation of Optimal Importance Density
%
% USAGE
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_TAYLOR(y, x, theta, model, f, Q, g, Gx, R)
%   [xp, lv] = SIS_UPDATE_GAUSSIAN_TAYLOR(y, x, theta, model, f, Q, g, Gx, R, L)
%
% DESCRIPTION
%
% PARAMETERS
%   Calculates the Gaussian approximation of the joint distribution
%                                    _    _    _  _    _         _
%                                 / | x[n] |  | mx |  |  Px   Pxy | \
%       p(x[n], y[n] | x[n-1]) = N\ |_y[n]_|; |_my_|, |_ Pxy' Py _| /
%
%   using generalized statistical linear regression and first order Taylor 
%   series approximation. From this, an approximation of the optimal 
%   importance density is calculated according to
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
%   Gx      Jacobian of the mean of the lieklihood (function handle @(x,
%           theta)
%   R       Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%           @(x, theta))
%   L       Number of iterations (optional, default: 1)
%   Xi      Unit sigma-points (optional, default: cubature rule)
%   wm      Mean sigma-point weights (optional, default: cubature rule)
%   wc      Covariance sigma-point weights (optional, default: cubature
%           rule)
%
% RETURNS
%   xp      New samples for x[n].
%   lv      Logarithm of the incremental weights.
%
% AUTHOR
%   2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Implement iterated variants

    narginchk(9, 10);
    if nargin < 10 || isempty(L)
        L = 1;
    elseif L > 1
        error('Iterated variants not implemented yet.');
    end
    [Nx, J] = size(x);
    xp = zeros(Nx, J);
    lv = zeros(1, J);
    
    for j = 1:J
        %% Calculate proposal
        %for l = 1:L
            fj = f(x(:, j), theta);
            Gxj = Gx(x(:, j), theta);
            S = Gxj*Q*Gxj' + R;
            K = Q*Gxj'/S;
            mp = fj + K*(y - g(fj));
            Pp = Q - K*S*K';
        %end
       
        %% Sample and calculate weight
        xp(:, j) = mp + chol(Pp).'*randn(Nx, 1);
        lv(j) = ( ...
            model.py.logpdf(y, xp(:, j), theta) ...
            + model.px.logpdf(xp(:, j), x(:, j), theta) ...
            - logmvnpdf(xp(:, j).', mp.', Pp.').' ...
        );
    end
end
