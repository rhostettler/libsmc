function [xp, lv, q] = sis_update_gaussian_taylor(y, x, theta, model, f, Q, g, Gx, R, L)
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
%   This implementation uses Taylor-series-based linearization of the
%   nonlinear model (as the (I)EKF).
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
%   Gx      Jacobian of the mean of the likelihood (function handle @(x,
%           theta)
%   R       Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%           @(x, theta))
%   L       Number of iterations (optional, default: 1)
%
% RETURNS
%   xp      New samples for x[n].
%   lv      Logarithm of the incremental weights.
%
% AUTHOR
%   2018-2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(9, 10);
    if nargin < 10 || isempty(L)
        L = 1;
    end
    [Nx, J] = size(x);
    xp = zeros(Nx, J);
    lv = zeros(1, J);
    qout = (nargout >= 3);
    if qout
        q = repmat(struct('mp', zeros(Nx, L+1), 'Pp', zeros(Nx, Nx, L+1)), [1, J]);
    end

    gamma = chi2inv(0.99, size(y, 1));
    
    %% Update
    % For all particles...
    for j = 1:J
        %% Calculate proposal
        % Initialize
        mx = f(x(:, j), theta);     % E{x[n] | x[n-1]}
        Px = Q(x(:, j), theta);     % Cov{x[n] | x[n-1]}
        mxp = mx;                    % Initial linearization density
        Pxp = Px;
        
        if qout
            q(j).mp(:, 1) = mxp;
            q(j).Pp(:, :, 1) = Pxp;
        end
        
        % IEKF-like approximation
        l = 0;
        done = false;
        while ~done
            % Expectations w.r.t. linearization density
            myp = g(mxp, theta);
            Gxj = Gx(mxp, theta);
            Pyp = Gxj*Pxp*Gxj' + R(mxp, theta);
            Pyxp = Gxj*Pxp;
                        
            % Calculate linearization w.r.t. linearization density
            % y = A*x + b + v, v ~ N(0, Omega)
            A = Pyxp/Pxp;
            b = myp - A*mxp;
            Omega = Pyp - A*Pxp*A';

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
            [~, nd] = chol(Pt, 'lower');
            if nd || ((y - my)'/Py*(y - my) >= gamma)
                done = true;
                warning('libsmc:warning', 'Posterior approximation failed (l = %d; posterior covariance negative definite), sampling from prior.', l);
            else
                done = false;
                mxp = mt;
                Pxp = Pt;
            end

            if qout
                q(j).mp(:, l+2) = mxp;
                q(j).Pp(:, :, l+2) = Pxp;
            end
            
            l = l + 1;            
            done = (l >= L) || done;
        end
       
        %% Sample and calculate weight        
        xp(:, j) = mxp + chol(Pxp).'*randn(Nx, 1);
        lv(j) = ( ...
            model.py.logpdf(y, xp(:, j), theta) ...
            + model.px.logpdf(xp(:, j), x(:, j), theta) ...
            - logmvnpdf(xp(:, j).', mxp.', Pxp.').' ...
        );
    end
end
