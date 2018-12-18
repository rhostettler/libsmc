function [mp, Pp, my, Py] = calculate_gaussian_proposal_sp(y, x, t, Ex, Vx, Ey_x, Vy_x, L, Xi, wm, wc)
% Gaussian Approximation to Optimal Importance Distribution w/ Sigma-Points
%
% USAGE
%   [mp, Pp] = CALCULATE_GAUSSIAN_PROPOSAL_SP(y, x, t, Ex, Vx, Ey_x, Vy_x)
%   [mp, Pp, my, Py] = CALCULATE_GAUSSIAN_PROPOSAL_SP(y, x, t, Ex, Vx, Ey_x, Vy_x, L, Xi, wm, wc)
%
% DESCRIPTION
%   Calculates the Gaussian approximation of the joint distribution
%                                    _    _    _  _    _         _
%                                 / | x[n] |  | mx |  |  Px   Pxy | \
%       p(x[n], y[n] | x[n-1]) = N\ |_y[n]_|; |_my_|, |_ Pxy' Py _| /
%
%   using generalized statistical linear regression and sigma-points.
%
%   From this, the approximate importance distribution
%
%       p(x[n] | y[n], x[n-1]) = N(x[n]; mp, Pp)
%
%   where
%
%       K = Pxy/Py
%       mp = mx + K*(y - my)
%       Pp = Px - K*Py*K'
%       
%   is calculated. Additionally, the moments my, Py of the predictive
%   density
%
%       p(y[n] | x[n-1]) = N(y[n]; my, Py)
%
%   are also returned, which can be used to calculate adjustment
%   multipliers in the auxiliary particle filter.
%
% PARAMETERS
%   y       Measurement y[n]
%   x       State x[n-1]
%   t       Time (or other parameter)
%   Ex      Mean of the dynamic model E{x[n] | x[n-1]} (function handle 
%           @(x, t))
%   Vx      Covariance of the dynamic model Cov{x[n] | x[n-1]} (function 
%           handle @(x, t))
%   Ey_x    Mean of the likelihood E{y[n] | x[n]} (function handle @(x, n))
%   Vy_x    Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%           @(x, n))
%   L       Number of iterations (optional, default: 1)
%   Xi      Unit sigma-points (optional, default: cubature rule)
%   wm      Mean sigma-point weights (optional, default: cubature rule)
%   wc      Covariance sigma-point weights (optional, default: cubature
%           rule)
%
% RETURNS
%   mp, Pp  Mean and covariance of the Gaussian approximation of the
%           optimal importance density
%   my, Py  Mean and covariance of predictive density y[n] given x[n-1]
%
% AUTHOR
%   2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * When Xi is specified, at least wm should be too (and vice-versa)

    %% Defaults
    narginchk(7, 11);
    Nx = size(x, 1);
    Ny = size(y, 1);
    if nargin < 8 || isempty(L)
        L = 1;
    end
    if nargin < 9 || isempty(Xi)
        % Default: Cubature sigma-points
        Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), Nx);
    end
    if nargin < 10 || isemtpy(wm)
        % Default: Cubature sigma-points
        [wm, wc] = ut_weights(Nx, 1, 0, 0);
    end
    if nargin < 11 || isempty(wc)
        wc = wm;
    end

    %% Calculate proposal
    % Preallocate
    I = length(wm);
    Y = zeros(Ny, I);
    
    % Initialize
    mx = Ex(x, t);
    Px = Vx(x, t);
    mp = mx;
    Pp = Px;

    % Iterations
    for l = 1:L
        % Generate sigma-points
        X = mp*ones(1, I) + chol(Pp).'*Xi;
        
        % Calculate expectations w.r.t. linearziation density
        Ey = zeros(Ny, 1);              % E{y}
        Ey2 = zeros(Ny, Ny);            % E{y*y'}
        EVy_x = zeros(Ny, Ny);          % E{V{y|x}}
        Eyx = zeros(Ny, Nx);            % E{y*x'}
        for i = 1:I
            Y(:, i) = Ey_x(X(:, i), t);
            Ey = Ey + wm(i)*Y(:, i);
            Ey2 = Ey2 + wc(i)*(Y(:, i)*Y(:, i)');            
            EVy_x = EVy_x + wc(i)*Vy_x(X(:, i), t);
            Eyx = Eyx + wc(i)*(Y(:, i)*X(:, i)');
        end
                
        % Calculate (co)variances w.r.t. linearization density
        Vy = Ey2 - (Ey*Ey') + EVy_x;    % V{y}
        Vy = (Vy + Vy')/2;
        Cyx = Eyx - (Ey*mp');           % C{y,x}
        
        % Calculate linearization w.r.t. linearization density
        % y = Phi*x + Gamma + nu, nu ~ N(0, Sigma)
        Phi = Cyx/Pp;
        Gamma = Ey - Phi*mp;
        Sigma = Vy - Phi*Pp*Phi';

        % Moments of y of the joint approximation
        my = Phi*mx + Gamma;
        Py = Phi*Px*Phi' + Sigma;
        Pxy = Px*Phi';
        
        % Posterior of x given y
        K = Pxy/Py;
        mp = mx + K*(y - my);
        Pp = Px - K*Py*K';
    end
end
