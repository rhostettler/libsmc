% Test for grid filter
%
% Note: Requires the EKF/UKF toolbox installed on your system and on the
% Matlab path.
% 
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

% Housekeeping
clear variables;
addpath ../src;

%% Parameters
% Simulation parameters
N = 10;        % No. of samples

% Model parameters
m0 = 0;
P0 = 1;
F = 1;
Q = 0.5;
G = 0.5;
R = 0.1;

% Model
model = model_lgssm(F, Q, G, R, m0, P0);

%% Simulation
[x, y] = simulate_model(model, [], N);

%% Estimation
% KF
[m_kf, P_kf] = kf_loop(m0, P0, G, R, y, F, Q);

% Grid filter
xg = -10:1e-2:10;
[m_grid, w] = gridf(model, y, [], xg);

%% Plots
for n = 1:N
    figure(1); clf();
    plot(xg, mvnpdf(xg.', m_kf(n), P_kf(:, :, n))); hold on;
    plot(xg, w(n, :));
    legend('Kalman Filter', 'Grid Filter');
    title(sprintf('Posterior at n = %d', n));

    figure(2); clf();
    plot(xg, w(n, :)-mvnpdf(xg.', m_kf(n), P_kf(:, :, n)).');
    title(sprintf('Error in posterior at n = %d', n));
    
    drawnow();
    pause(0.1);
end
