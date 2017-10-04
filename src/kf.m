function [m, P, mp, Pp] = kf(y, F, H, Q, R, m0, P0, K)
% Kalman filter
%
% SYNOPSIS
%   [m, P, mp, Pp] = kf(y, F, H, Q, R, m0, P0, K)
%
% DESCRIPTION
%   Kalman filter for the linear, Gaussian state space system of the form
%
%       x[n] = F x[n-1] + q[n]
%       y[n] = H x[n] + r[n]
%
%   where x[0] ~ N(m0, P0), q[n] ~ N(0, Q), and r[n] ~ N(0, R).
%
%   If K is supplied, the stationary variant is used.
%
% PARAMETERS
%   y   A batch of measurements where each column is a time step.
%
%   F, H
%       State transition and observation matrices
%
%   Q, R
%       Noise covariances
%
%   m0, P0
%       Parameters of the initial state distribution
%   
%   K   Stationary Kalman gain (optional)
%
% RETURNS
%   m, P
%       Posterior mean and covariance
%
%   mp, Pp
%       Predicted mean and covariance
%
% AUTHOR
%   2016-03-23 -- Roland Hostettler <roland.hostettler@aalto.fi>

    Ny = size(y, 2);
    Nx = size(m0, 1);

    m = zeros([Nx, Ny]);
    P = zeros([Nx, Nx, Ny]);
    mp = zeros([Nx, Ny]);
    Pp = zeros([Nx, Nx, Ny]);
    
    mu = m0;
    Sigma = P0;
    
    for n = 1:Ny
        %% Predict
        mu_p = F*mu;
        Sigma_p = F*Sigma*F' + Q;
        
        %% Update
        S = H*Sigma_p*H' + R;
        if nargin < 8
            K = (Sigma_p*H')/S;
        end
        mu = mu_p + K*(y(:, n) - H*mu_p);
        Sigma = Sigma_p - K*S*K';
        
        %% Store
        mp(:, n) = mu_p;
        Pp(:, :, n) = Sigma_p;
        m(:, n) = mu;
        P(:, :, n) = Sigma;
    end
end
