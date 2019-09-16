function [xp, lqx, qstate] = sample_gaussian_flow(model, y, x, theta, f, Q, g, Gx, dGxdx, R, L)
% # Gaussian particle flow OID approximation sampling
% ## Usage
% * `[xp, lqx] = sample_gaussian_flow(model, y, x, theta, f, Q, g, Gx, dGxdx, R)`
% * `[xp, lqx, qstate] = sample_gaussian_flow(model, y, x, theta, f, Q, g, Gx, dGxdx, R, L)`
%
% ## Description
% Gaussian particle flow importance sampling according to [1]. Approximates
% the optimal importance density (OID) using a (deterministic or 
% stochastic) Gaussian flow by first sampling from the prior (bootstrap)
% and then propagating the particles and their weights according to the
% Gaussian flow.
%
% Notes:
% * For simplicity, a fixed step size is used, that is, integration is 
%   split into `L` fixed-length intervals.
% * Stochastic flow is not implemented yet.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector y[n].
% * `x`: dx-times-J matrix of particles for the state x[n-1].
% * `theta`: dtheta-times-1 vector of other parameters.
% * `f`: Mean of the dynamic model E{x[n] | x[n-1]} (function handle 
%   `@(x, theta)`).
% * `Q`: Covariance of the dynamic model Cov{x[n] | x[n-1]} (function
%   handle `@(x, theta)`).
% * `g`: Mean of the likelihood E{y[n] | x[n]} (function handle 
%   `@(x, theta)`).
% * `Gx`: Jacobian of the mean of the likelihood (function handle 
%   `@(x, theta)`).
% * `dGxdx`: 1-times-dx cell array of dy-times-dx matrices of second 
%   derivatives where the nth cell is
%
%                  _                                       _ 
%                 | d^2 g_1/dx_n dx_1 d^2 g_1/dx_n dx_2 ... |
%                 | d^2 g_2/dx_n dx_1 d^2 g_2/dx_n dx_2 ... |
%     dGxdx{n} =  | ...                                     |
%                 |_                                       _|
%
%   (function handle `@(x, theta)`).
% * `R`: Covariance of the likelihood Cov{y[n] | x[n]} (function handle
%   `@(x, theta)`).
% * `L`: Number of integration steps to use (default: `5`).
% 
% ## Output
% * `xp`: New samples.
% * `lqx`: 1-times-J vector of importance density evaluations at 
%   `xp(:, j)`.
% * `qstate`: Importance density state.
%
% ## References
% 1. P. Bunch and S. J. Godsill, "Approximations of the optimal importance 
%    density using Gaussian particle flow importance sampling," Journal of 
%    the American Statistical Association, vol. 111, no. 514, pp. 748–762, 
%    2016.
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
% * Add stochastic flow (with deterministic flow as default)

    %% Defaults
    narginchk(10, 11);
    if nargin < 11 || isempty(L)
        L = 5;
    end
    lambda = 1/L;   % Integration step size
    gamma = 0;      % Deterministic flow
    
    %% Gaussian flow
    % Initialize the log weights
    [dx, J] = size(x);
    xp = zeros(dx, J);
    lv = zeros(1, J);    
    depsilon = zeros(dx, 1);    % Stochastic increments; zero for deterministic flows
    
    for j = 1:J
        % Sample from prior
        xp(:, j) = model.px.rand(x(:, j), theta);
        mp = f(x(:, j), theta);
        dmpdxn = zeros(dx, dx);
        Pp = Q(x(:, j), theta);
        lvp = model.px.logpdf(xp(:, j), x(:, j), theta);
        lv(:, j) = lvp;
    
        % Integrate using L intervals
        for l = 1:L
            % Linearize observation function using (17)
            Gxj = Gx(xp(:, j), theta);            
            dGxdxj = dGxdx(xp(:, j), theta);
            Rj = R(xp(:, j), theta);
            yj = y - g(xp(:, j), theta) + Gxj*xp(:, j);
            
            % Calculate approximate Gaussian moments using (18) (Kalman
            % form)
            S = Rj/lambda + Gxj*Pp*Gxj';
            K = Pp*Gxj'/S;
            ml = mp + K*(yj - Gxj*mp);   % "mhat lambda"
            Pl = Pp - K*S*K';            % "Phat lambda"
            Pl = (Pl + Pl')/2;
            PlinvPp = Pl/Pp;
            sqrtPlinvPp = sqrtm(PlinvPp);
            
            % Jacobian (n = j in the paper, i.e., the column index)
            dmldxn = zeros(dx, dx);
            tmp2 = zeros(dx, dx);
            tmp3 = zeros(dx, dx);
            for n = 1:dx
                % Matrix of second derivatives dGx/dx_n dx_k
                dGxdxn = dGxdxj{n};
                
                % Columns of the first term
                dmldxn(:, n) = lambda*Pl*( ...
                    dGxdxn'/Rj*(yj - Gxj*ml) + Gxj'/Rj*dGxdxn*(xp(:, j) - ml) ...
                );
            
                % Columns of the second term
                % TODO: Not implemented yet since we assume a deterministic
                % flow for now.
                
                % Columns of the third term
                dPlinvPpdxn = lambda*Pl*(dGxdxn'/Rj*Gxj + Gxj'/Rj*dGxdxn)*(eye(dx) - PlinvPp);
                dsqrtPlinvPpdxn = sylvester(sqrtPlinvPp, sqrtPlinvPp, dPlinvPpdxn);
                tmp3(:, n) = dsqrtPlinvPpdxn*(xp(:, j) - mp);                
            end
            Jxp = ( ...
                dmldxn + exp(-1/2*gamma*lambda)*sqrtPlinvPp*(eye(dx) - dmpdxn) ... % First term
                + sqrt((1-exp(-gamma*lambda))/lambda)*tmp2 ...                     % Second term (omitted since we assume a deterministic flow for now)
                + exp(-1/2*gamma*lambda)*tmp3 ...                                  % Third term
            );
            
            % Advance state using (25)
            xp(:, j) = ml + exp(-1/2*gamma*lambda)*sqrtPlinvPp*(xp(:, j) - mp) ...
                +((1-exp(-gamma*lambda))/lambda)*sqrtm(Pl)*depsilon;
            
            % Advance weights using (26)
            lvl = l/L*model.py.logpdf(y, xp(:, j), theta) + model.px.logpdf(xp(:, j), x(:, j), theta);
            lv(:, j) = lv(:, j) + lvl - lvp + log(abs(det(Jxp)));
            
            % Store for next iteration
            mp = ml;
            dmpdxn = dmldxn;
            Pp = Pl;
            lvp = lvl;
        end
    end
    
    %% Proposal
    % There is no proposal in the same sense as for other importance
    % sampling schemes here. However, the particles and weights are
    % advanced using an ODE. Hence, we misuse the 'importance density'
    % output 'q' to contain the non-normalized incremental weights produced
    % by the weight ODE. Note that this requires the
    % 'calculate_incremental_weights_flow' to be used with 'pf'.
    %
    % TODO: This is abusive of the structure but works for now; we could
    % actually 'fake' the importance density such that it could be
    % evaluated but that is nonsense.
    % => Actually, we'll modify the interface of the sample() function to
    % also return the "lqx" along with the samples, and resample will
    % return "lqalpha".
    lqx = lv;
    
    % TODO: We also want to put something into qstate
    qstate = [];
end
