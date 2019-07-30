% UNGM example
% 
% Example using iterated conditional expectations optimal importance 
% density (OID) approximation. The model considered is the univariate 
% nonlinear growth model (UNGM):
% 
%   x[n] = 0.5*x[n-1] + 25*x[n-1]/(1+x[n-1]^2) + 8*cos(1.2*n) + q[n]
%   y[n] = x[n]^2/20 + r[n]
%   x[0} ~ N(0, 5)
%
% with q[n] ~ N(0, 10) and r[n] either zero (approximate Bayesian
% computation example; ABC) or r[n] ~ N(0, 1e-4). In the ABC case, either a
% Gaussian or uniform pseudo-likelihood are implemented. In the regular
% case, the measurement noise variance of 1e-4 is 4 magnitudes smaller than
% the standard UNGM example considered in the literature, which makes the
% likelihood (and posterior) extremely peaky and very difficult for
% particle filtering. Check some of the density plots at the end.
%
% Several algorithms can be compared, but the ones really tested are:
% * (Dense grid filter)
% * Bootstrap particle filter
% * SIR particle filter with one-step Gaussian OID approximation
% * SIR particle filter with iterated conditional expectations and
%   posterior linearization OID approximation
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
rng(5011);

%% Parameters
% Grid for the grid filter
xg = -25:0.5e-2:25;

% Common particle filter parameters
J = 10000;       % Number of particles; TODO: 10-2000 are ok, need to run 5k and 10k for bpf/one-step
L = 5;         % Maximum number of iterations (5)

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-1/2));
beta = 1;
kappa = 0;

% Simulation parameters
N = 100;            % Number of time samples (100)
K = 100;            % Number of MC simulations (100)

% Model parameters: Typical UNGM parameters except for the measurement
% noise covariance (R), which is normally set to 1 (1e-4 makes it much more
% difficult)
Q = 10;             % Process noise variance (10)
R = 1e-4;           % Measurement noise variance (1e-4)
m0 = 0;             % Initial state mean (0)
P0 = 5;             % Initial state variance (5)

% Algorithms to run
gridf = false;      % Dense grid filter, very computationally expensive
bpf = true;         % Bootstrap particle filter
cf1 = true;         % One-step OID approximation (EKF/UKF-like)
cf = false;          % Closed form iterated conditional expectations w/ posterior linearization

% Other switches
abc = false;        % If set to true, measurements are noise-free
uniform = false;    % Use uniform pseudo-likelihood? (Gaussian is used by default)
store = true;       % Save the simulation results (true/false)
plots = false;      % Show plots

%% Model
% Dynamic and measurement function
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
g = @(x, theta) x.^2/20;
Gx = @(x, theta) 2*x/20;

% Closed-form moments
Ey = @(m, P, theta) (m^2 + P)/20;
Cy = @(m, P, theta) R + (4*m^2*P + 2*P^2)/20^2;
Cyx = @(m, P, theta) 2*m*P/20;

% Model parameters (time in this case)
theta = 1:N;

% Model struct
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0, true);
if uniform
    epsilon = sqrt(12*R)/2; % To match variance with Gaussian used previously
    model.py = struct( ...
        'fast', true, ...
        'rand', @(x, theta) g(x, theta) - epsilon + epsilon*rand(1, size(x, 2)), ...
        'logpdf', @(y, x, theta) log(unifpdf(y-g(x, theta), -epsilon, epsilon)) ...
    );
end

%% Sampling algorithms
% Approximation of the optimal proposal using approximate Gaussian particle
% flow
par_gf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian_flow(model, y, x, theta, f, @(x, theta) Q, g, Gx, dGxdx, @(x, theta) R, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_flow ...
);

% SLR using Taylor series approximation
slr_lin = @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R);
par_lin = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_lin, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% SLR using sigma-points
Nx = size(m0, 1);
[wm, wc, c] = ut_weights(Nx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), c);
slr_sp = @(mp, Pp, theta) slr_sp(mp, Pp, theta, g, @(x, theta) R, Xi, wm, wc);
par_sp = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_sp, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic... , ...
);

% One-step closed form Gaussian OID approximation
slr_cf = @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx);
par_cf1 = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_cf, 1), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% Proposed method
par_cf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_cf, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

%% MC Simulations
% Preallocate
if 1
xs = zeros(1, N, K);
ys = zeros(1, N, K);

xhat_grid = zeros(1, N, K);
xhat_bpf = xhat_grid;
xhat_gf = xhat_grid;
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
xhat_cf1 = xhat_bpf;
xhat_cf = xhat_bpf;

ess_bpf = zeros(1, N, K);
ess_gf = ess_bpf;
ess_lin = ess_bpf;
ess_sp = ess_bpf;
ess_cf1 = ess_bpf;
ess_cf = ess_bpf;

r_bpf = zeros(1, N+1, K);
r_gf = r_bpf;
r_lin = r_bpf;
r_sp = r_bpf;
r_cf1 = r_bpf;
r_cf = r_bpf;

t_grid = zeros(1, K);
t_bpf = t_grid;
t_gf = t_grid;
t_lin = t_bpf;
t_sp = t_bpf;
t_cf1 = t_bpf;
t_cf = t_bpf;

H_grid = zeros(1, N, K);
H_bpf = H_grid;
H_cf1 = H_bpf;
H_cf = H_bpf;

l_cf = zeros(1, K);

if gridf
    NGrid = length(xg);
    w = zeros(N, NGrid, K);
end
end

% Simulate
fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);

% par
for k = 1:K
    %% Model simulation
    if ~abc
        [xs(:, :, k), ys(:, :, k)] = simulate_model(model, 1:N, N);
    else
        y = zeros(1, N);
        x = model.px0.rand(1);
        for n = 1:N
            x = model.px.rand(x, theta(n));
            y(:, n) = g(x, n);        
            xs(:, n, k) = x;
        end
        ys(:, :, k) = y;
    end

    %% Estimation
    % Grid filter
    if gridf
        tic;
        [xhat_grid(:, :, k), w(:, :, k)] = gridf(model, ys(:, :, k), 1:N, xg);
        t_grid(k) = toc;
    end
    
    % Bootstrap PF
    if bpf
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).rstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
    %         [H_bpf(:, :, k), H_grid(:, :, k)] = estimate_cross_entropy(xg, w(:, :, k), sys_bpf(2:N+1));
    end
    
    % Gaussian flow
if 0
    tic;
    [xhat_gf(:, :, k), sys_gf] = pf(model, ys(:, :, k), theta, J, par_gf);
    t_gf(k) = toc;
    tmp = cat(2, sys_gf(2:N+1).rstate);
    ess_gf(:, :, k) = cat(2, tmp.ess);
end
    
    % Taylor series
if 0
    tic;
    [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J, par_lin);
    t_lin(k) = toc;
end

    % Sigma-point approximation of SLR
if 0
    tic;
    [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J, par_sp);
    t_sp(k) = toc;
    tmp = cat(2, sys_sp(2:N+1).rstate);
    ess_sp(:, :, k) = cat(2, tmp.ess);
end

    % One-step OID approximation
    if cf1
        tic;
        [xhat_cf1(:, :, k), sys_cf1] = pf(model, ys(:, :, k), theta, J, par_cf1);
        t_cf1(k) = toc;
        tmp = cat(2, sys_cf1(2:N+1).rstate);
        ess_cf1(:, :, k) = cat(2, tmp.ess);
    %     H_cf1(:, :, k) = estimate_cross_entropy(xg, w(:, :, k), sys_cf1(2:N+1));
    end
   
    % Closed form, iterated conditional expectations
    if cf
        tic;
        [xhat_cf(:, :, k), sys_cf] = pf(model, ys(:, :, k), theta, J, par_cf);
        t_cf(k) = toc;
        tmp = cat(2, sys_cf(2:N+1).rstate);
        ess_cf(:, :, k) = cat(2, tmp.ess);
        tmp = cat(1, sys_cf(2:N+1).q);
        l_cf(k) = mean(cat(1, tmp.l));
    %     H_cf(:, :, k) = estimate_cross_entropy(xg, w(:, :, k), sys_cf(2:N+1));
    end

    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance summary
iNaN_grid = squeeze(isnan(xhat_grid(1, N, :)));
iNaN_bpf = squeeze(isnan(xhat_bpf(1, N, :)));
iNaN_gf = squeeze(isnan(xhat_gf(1, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(1, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(1, N, :)));
iNaN_cf1 = squeeze(isnan(xhat_cf1(1, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(1, N, :)));

e_rmse_grid = trmse(xhat_grid(:, :, ~iNaN_grid) - xs(:, :, ~iNaN_grid));
e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_gf = trmse(xhat_gf(:, :, ~iNaN_gf) - xs(:, :, ~iNaN_gf));
e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));
e_rmse_cf1 = trmse(xhat_cf1(:, :, ~iNaN_cf1) - xs(:, :, ~iNaN_cf1));
e_rmse_cf = trmse(xhat_cf(:, :, ~iNaN_cf) - xs(:, :, ~iNaN_cf));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\tESS\n');
fprintf( ...
    'Grid\t%.2e (%.2e)\t%.2f (%.2f)\tn/a\t\tn/a\t\tn/a\n', ...
    mean(e_rmse_grid), std(e_rmse_grid), mean(t_grid), std(t_grid) ...
);
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K, ...
    mean(mean(ess_bpf)), std(mean(ess_bpf)) ...
);
if 0
fprintf( ...
    'Flow\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_gf), std(e_rmse_gf), mean(t_gf), std(t_gf), ...
    mean(sum(r_gf(:, :, ~iNaN_gf))/N), std(sum(r_gf(:, :, ~iNaN_gf))/N), ...
    1-sum(iNaN_gf)/K ...
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
end
fprintf( ...
    'CF1\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_cf1), std(e_rmse_cf1), mean(t_cf1), std(t_cf1), ...
    mean(sum(r_cf1(:, :, ~iNaN_cf1))/N), std(sum(r_sp(:, :, ~iNaN_cf1))/N), ...
    1-sum(iNaN_cf1)/K, ...
    mean(mean(ess_cf1)), std(mean(ess_cf1)) ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_cf), std(e_rmse_cf), mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N), std(sum(r_sp(:, :, ~iNaN_cf))/N), ...
    1-sum(iNaN_cf)/K, ...
    mean(mean(ess_cf)), std(mean(ess_cf)) ...
);
fprintf('Average number of iterations: %.1f (%.1f)\n', mean(l_cf), std(l_cf));

%% Plots
if plots
% For plots of a specific MC run, use the last one since we only have the 
% particle system for that one available (too much to store otherwise, 
% we'll run out of memory).
k = K;

% Show the true state and the MMSE
figure(1); clf();
subplot(211);
plot(ys(:, :, k)); grid on;
title('Data');
subplot(212);
plot(xs(:, :, k)); hold on;
plot(xhat_bpf(:, :, k));
plot(xhat_lin(:, :, k));
plot(xhat_sp(:, :, k));
plot(xhat_cf1(:, :, k));
plot(xhat_cf(:, :, k));
legend('State', 'Bootstrap', 'Linearized', 'Sigma-Points', 'CF1', 'ICE-CF');
title('MMSE');

% Plot the posterior in a waterfall plot, together with the state trace and
% particles (for the kth MC run)
figure(2); clf();
waterfall(xg, 1:N, w(:, :, k)); hold on; grid on;
plot(xs(:, :, k), 1:N, '--r', 'LineWidth', 2);

for n = 1:N
    x_bpf = sys_bpf(n+1).x;
    plot3(x_bpf, n*ones(1, J), zeros(1, J), '.');
    
    x_cf1 = sys_cf1(n+1).x;
    plot3(x_cf1, n*ones(1, J), zeros(1, J), 'o');
    
    x_cf = sys_cf(n+1).x;
    plot3(x_cf, n*ones(1, J), zeros(1, J), '*');
end

% Iterate through all posteriors (slightly easier to see the differences
% than in the waterfall plot).
for n = 1:N
    figure(3); clf();
    plot(xg, w(n, :, k)); hold on; grid on;
    plot([xs(:, n, k), xs(:, n, k)], [0, 1]*max(w(n, :, k)));

    x_bpf = sys_bpf(n+1).x;
    plot(x_bpf, 0*ones(1, J), '.');
    
    x_cf1 = sys_cf1(n+1).x;
    plot(x_cf1, 0*ones(1, J), 'o');

    x_cf = sys_cf(n+1).x;
    plot(x_cf, 0*ones(1, J), 'x');

%     set(gca, 'YScale', 'log');
    legend('Grid', 'True state', 'BPF', 'CF1', 'ICE-CF');
    title(sprintf('Posterior and particles at n = %d', n));
    pause();
end

if 0
% Mean ESS
figure(4); clf();
plot(mean(ess_bpf, 3)); hold on; grid on;
plot(mean(ess_cf1, 3));
plot(mean(ess_cf, 3));
legend('BPF', 'CF1', 'ICE-PF (CF)');
title('Effective sample size');
end
end

%% Store results
% Do this before the plots so that we don't store a bunch of temporary
% variables that we don't need
if store
    % Get the particle system at a particular time n, used for illustration
    if gridf
        px_n = w([1, 3], :, k);
        x_bpf_n = [
            sys_bpf(1+1).x;
            sys_bpf(3+1).x;
        ];
        x_cf1_n = [
            sys_cf1(1+1).x;
            sys_cf1(3+1).x;
        ];
        x_cf_n = [
            sys_cf(1+1).x;
            sys_cf(3+1).x;
        ];
    end
    
    % Remove these from the savefile; they make the file to explode
    clear sys_bpf sys_cf1 sys_cf
    
    % Store
    outfile = sprintf('Savefiles/example_ungm_J=%d_L=%d_K=%d_N=%d.mat', J, L, K, N);
    save(outfile);
end
