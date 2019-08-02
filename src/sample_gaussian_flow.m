function [xp, q] = sample_gaussian_flow(model, y, x, theta, f, Q, g, Gx, dGxdx, R, L)
% # Gaussian flow OID approximation sampling
% ## Usage
% * 
%
% ## Description
% 
%
% A few simplifications from the original article:
% * For simplicity, we use a fixed step size for integration, parameter L
% * The stochastic flow is not implemented yet (gamma = 0)
%
% ## Input
% 
% * `dGxdx`: 1-times-dx cell array of dy-times-dx matrices of second 
%   derivatives where the nth cell is
%                  _                                       _ 
%                 | d^2 g_1/dx_n dx_1 d^2 g_1/dx_n dx_2 ... |
%                 | d^2 g_2/dx_n dx_1 d^2 g_2/dx_n dx_2 ... |
%     dGxdx{n} =  | ...                                     |
%                 |_                                       _|
% * `R`: 
% * `L`: 
% 
% ## Outut
% * 
%
% ## References
% 1. 
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
% * Add documentation
% * Add possibilities for stochastic flow (and make deterministic flow
%   default)

    %% Defaults
    narginchk(10, 11);
    if nargin < 11 || isempty(L)
        L = 5;
    end
    
    %% Gaussian flow
    % Initialize the log weights
    [dx, J] = size(x);
    xp = zeros(dx, J);
    lv = zeros(1, J); %log(1/J)*ones(1, J);
    
    % Step size assumed fixed for now
    lambda = 1/L;
    
    % Fixed for deterministic flow
    gamma = 0;
    depsilon = zeros(dx, 1);
    
    for j = 1:J
        % Sample from prior
        xp(:, j) = model.px.rand(x(:, j), theta);
        mp = f(x(:, j), theta);
        dmpdxn = zeros(dx, dx);
        Pp = Q(x(:, j), theta);
        lvp = model.px.logpdf(xp(:, j), x(:, j), theta);
        lv(:, j) = lvp;
    
        % TODO: We split the interval (0,1] into L equidistant points for
        % now
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
            ml = mp + K*(yj - Gxj*mp);  % "mhat lambda"
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
                    dGxdxn/Rj*(yj - Gxj*ml) + Gxj'/Rj*dGxdxn*(xp(:, j) - ml) ...
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
    q = lv;
end
