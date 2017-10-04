function [ms, Ps] = rtss(F, m, P, mp, Pp)
% Rauch-Tung-Striebel smoother
%
% SYNOPSIS
%   [ms, Ps] = rtss(F, m, P, mp, Pp)
%
% DESCRIPTION
%   Rauch-Tung Striebel smoother for linear, Gaussian state space systems.
%   Requires to have run a Kalman filter first.
%
% PARAMETERS
%   F
%       State transiton matrix
%
%   m, P
%       Mean and covariance of the Kalman filter step
%
%   mp, Pp
%       Prediction mean and covariance of the Kalman filter step
%
% RETURNS
%
% SEE ALSO
%   kf
%
% AUTHOR
%   2016-03-23 -- Roland Hostettler <roland.hostettler@aalto.fi>

    [Nx, Ny] = size(m);
    ms = zeros([Nx, Ny]);
    Ps = zeros([Nx, Nx, Ny]);
    
    % Initialize
    ms(:, Ny) = m(:, Ny);
    Ps(:, :, Ny) = P(:, :, Ny);
    
    % Calculate
    for n = Ny-1:-1:1
        G = (P(:, :, n)*F')/Pp(:, :, n+1);
        ms(:, n) = m(:, n) + G*(ms(:, n+1) - mp(:, n+1));
        Ps(:, :, n) = P(:, :, n) + G*(Ps(:, :, n+1) - Pp(:, :, n+1))*G';
    end
end
