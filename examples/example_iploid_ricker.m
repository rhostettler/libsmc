% Ricker population example
%
% Example of particle filtering with the iterated-conditional-
% expectations-based importance density. The model considered here is the
% Ricker population model
%
%   x[n] = log(44.7) + x[n-1] - exp(x[n-1]) + q[n]
%   y[n] ~ P(10*exp(x[n]))
%   x[0] ~ N(log(7), 0.1)
%
% with q[n] ~ N(0, 0.3^2).
%
% 2019 -- Roland Hostettler

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
rng(5011);

% Disable warnings for efficiency
if 0
    spmd
        warning('off', 'libsmc:warning');
    end
else
    warning('off', 'libsmc:warning');
end

%% Parameters
% Filter parameters
J = 10;         % Number of particles
L = 1;           % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-0.5));
beta = 1;
kappa = 0;

% Simulation parameters
N = 1000;       % Number of time samples
K = 10;        % Number of MC simulations

% Model parameters
Q = 0.3^2;
m0 = log(7);
P0 = 0.1;

% Save the simulation?
store = false;

%% Model
f = @(x, n) log(44.7) + x - exp(x);
g = @(x, n) 10*exp(x);

px0 = struct( ...
    'rand', @(M) m0*ones(1, M) + chol(P0).'*randn(1, M) ...
);
LQ = sqrt(Q);
px = struct( ...
    'fast', false, ...
    'rand', @(x, theta) f(x, theta) + LQ.'*randn(1, size(x, 2)), ...
    'logpdf', @(xp, x, theta) logmvnpdf(xp.', f(x, theta).', Q.').' ...
);
py = struct( ...
    'fast', false, ...
    'rand', @(x, theta) poissrnd(10*exp(x), 1), ...
    'logpdf', @(y, x, theta) log(poisspdf(y, 10*exp(x))) ...
);
model = struct('px0', px0, 'px', px, 'py', py);
theta = [];

%% Algorithm parameters
% Approximation of the optimal proposal using linearization
Gx = @(x, theta) 10*exp(x);
R = @(x, n) 10*exp(x);
slr_lin = @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R);
par_lin = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_lin, L), ...
    'calculate_incremental_wights', @calculate_incremental_weights_generic ...
);

% SLR using unscented transform
dx = size(m0, 1);
[wm, wc, c] = ut_weights(dx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(dx, 1), eye(dx), c);
slr_sp = @(m, P, theta) slr_sp(m, P, theta, g, R, Xi, wm, wc);
par_sp = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_sp, L), ...
    'calculate_incremental_wights', @calculate_incremental_weights_generic ...
);

% Closed-form solution to the moment integrals
Ey = @(m, P, theta) 10*exp(m + P/2);
Cy = @(m, P, theta) 100*exp(2*m + P)*(exp(P) - 1) + 10*exp(m + P/2);
Cyx = @(m, P, theta) 10*P*exp(m + P/2);
slr_cf = @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx);
par_cf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_cf, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);
ys = zeros(1, N, K);

xhat_bpf = zeros(1, N, K);
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
xhat_cf = xhat_bpf;

ess_bpf = zeros(1, N, K);
ess_lin = ess_bpf;
ess_sp = ess_bpf;
ess_cf = ess_bpf;

r_bpf = zeros(1, N, K);
r_lin = r_bpf;
r_sp = r_bpf;
r_cf = r_bpf;

t_bpf = zeros(1, K);
t_lin = t_bpf;
t_sp = t_bpf;
t_cf = t_bpf;

fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);
% parfor k = 1:K
for k = 1:K
    %% Simulation
    [xs(:, :, k), ys(:, :, k)] = simulate_model(model, theta, N);

    %% Estimation
    % Bootstrap PF
    if 1 %L == 1
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).rstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
        r_bpf(:, :, k) = cat(2, tmp.r);
    else
        xhat_bpf(:, :, k) = NaN*ones(1, N);
    end
    
    % Taylor series approximation of SLR
    tic;
    [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J, par_lin);
    t_lin(k) = toc;
    tmp = cat(2, sys_lin(2:N+1).rstate);
    ess_lin(:, :, k) = cat(2, tmp.ess);
    r_lin(:, :, k) = cat(2, tmp.r);

    % SLR using sigma-points, L iterations
    tic;
    [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J, par_sp);
    t_sp(k) = toc;
    tmp = cat(2, sys_sp(2:N+1).rstate);
    ess_sp(:, :, k) = cat(2, tmp.ess);
    r_sp(:, :, k) = cat(2, tmp.r);
    
    % SLR using closed-form expressions, L iterations
    tic;
    [xhat_cf(:, :, k), sys_cf] = pf(model, ys(:, :, k), theta, J, par_cf);
    t_cf(k) = toc;
    tmp = cat(2, sys_cf(2:N+1).rstate);
    ess_cf(:, :, k) = cat(2, tmp.ess);
    r_cf(:, :, k) = cat(2, tmp.r);

    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance figures
iNaN_bpf = squeeze(isnan(xhat_bpf(:, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(:, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(:, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(:, N, :)));

e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));
e_rmse_cf = trmse(xhat_cf(:, :, ~iNaN_cf) - xs(:, :, ~iNaN_cf));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_lin), std(e_rmse_lin), mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_sp), std(e_rmse_sp), mean(t_sp), std(t_sp), ...
    mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
    1-sum(iNaN_sp)/K ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_cf), std(e_rmse_cf), mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N), std(sum(r_cf(:, :, ~iNaN_cf))/N), ...
    1-sum(iNaN_cf)/K ...
);

%% Plots
% Plots of how the mean and covariance evolve due to the iterations; only
% used for debugging and specific insight
if 0
mps = zeros(L+1, N);
Pps = zeros(L+1, N);

for j = 1:J
    for n = 2:N+1
        mps(:, n) = sys_cf(n).q(j).mp.';
        Pps(:, n) = squeeze(sys_cf(n).q(j).Pp);
    end
if 0
    figure(1); clf();
    plot(mps(1:4, :).');
    legend('l = 0', 'l = 1', 'l = 2', 'l = 3', 'l = 4', 'l = 5');
    title('Mean');
end

    figure(2); clf();
    plot(Pps(1:4, :).');
    legend('l = 0', 'l = 1', 'l = 2', 'l = 3', 'l = 4', 'l = 5');
    title('Variance');
    drawnow();
    pause(1);
end
end

% ESS
figure(3); clf();
plot(mean(ess_bpf(:, :, ~iNaN_bpf), 3)); hold on; grid on;
plot(mean(ess_lin(:, :, ~iNaN_lin), 3));
plot(mean(ess_sp(:, :, ~iNaN_sp), 3));
plot(mean(ess_cf(:, :, ~iNaN_cf), 3));
legend('BPF', 'ICE-PF (Taylor)', 'ICE-PF (SP)', 'ICE-PF (CF)');
title('Effective sample size');

%% Store results
if store
    outfile = sprintf('Savefiles/example_ricker_J=%d_L=%d.mat', J, L);
    save(outfile);
end
