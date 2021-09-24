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
% * Gaussian flow OID approximation
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
J = 100;        % Number of particles; (10, 25, 50, 100, 250, 500, 1000)
L = 5;          % Maximum number of iterations and GF integration steps (5)

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-1/2));
beta = 1;
kappa = 0;

% Simulation parameters
N = 100;            % Number of time samples (100)
K = 10;            % Number of MC simulations (100)

% Model parameters: Typical UNGM parameters except for the measurement
% noise covariance (R), which is normally set to 1 (1e-4 makes it much more
% difficult)
Q = 10;             % Process noise variance (10)
R = 1e-2;           % Measurement noise variance (1e-4)
m0 = 0;             % Initial state mean (0)
P0 = 5;             % Initial state variance (5)

% Algorithms to run
use_gridf = false;      % Dense grid filter, very computationally expensive (false)
use_bpf = true;         % Bootstrap particle filter
use_gf = false;         % Gaussian flow (don't run this for 5e3, 10e3)
use_pfpf = false;
use_cf1 = true;         % One-step OID approximation (EKF/UKF-like)
use_cf = true;          % Closed form iterated conditional expectations w/ posterior linearization (don't run this for 5e3, 10e3)
use_lin = true;         % Sigma-point dito
use_sp = true;          % Taylor series dito

% Other switches
abc = false;        % If set to true, measurements are noise-free (false)
uniform = false;    % Use uniform pseudo-likelihood? (false)
store = false;      % Save the simulation results (true/false)
plots = false;      % Show plots (false)

%% Model
% Dynamic and measurement function
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
Fx = @(x, n) 0.5 + 25./(1+x.^2) - 50*x.^2./(1+x.^2).^2;
g = @(x, theta) x.^2/20;
Gx = @(x, theta) 2*x/20;
dGxdx = @(x, theta)  {1/10};

% Closed-form moments
Ey = @(m, P, theta) (m^2 + P)/20;
Cy = @(m, P, theta) R + (4*m^2*P + 2*P^2)/20^2;
Cyx = @(m, P, theta) 2*m*P/20;

% Model parameters (time in this case)
theta = 1:N;

% Model struct
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0, Fx, Gx, true);
% model.px0.m = m0;
% model.px0.P = P0;
% model.px.m = f;
% model.px.dm = Fx;
% model.px.P = Q;
% model.py.m = g;
% model.py.dm = Gx;
% model.py.P = R;
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

% Particle flow particle filter
par_pfpf = struct( ...
    'L', 29, ...
    'ukf', [alpha, beta, kappa] ...
);

% SLR using Taylor series approximation
par_lin_sampler = struct( ...
    'L', L, ...
    'slr', @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R) ...
);
par_lin = struct( ...
    'sample', @(model, y, x, lw, theta) sample_gaussian(model, y, x, lw, theta, par_lin_sampler), ...
    'calculate_weights', @calculate_weights ...
);

% SLR using sigma-points
Nx = size(m0, 1);
[wm, wc, c] = ut_weights(Nx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), c);
par_sp_sampler = struct( ...
    'L', L, ...
    'slr', @(mp, Pp, theta) slr_sp(mp, Pp, theta, g, @(x, theta) R, Xi, wm, wc) ...
);
par_sp = struct( ...
    'sample', @(model, y, x, lw, theta) sample_gaussian(model, y, x, lw, theta, par_sp_sampler), ...
    'calculate_weights', @calculate_weights ...
);

% One-step closed form Gaussian OID approximation
par_cf1_sampler = struct( ...
    'L', 1, ...
    'slr', @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx) ...
);
par_cf1 = struct( ...
    'sample', @(model, y, x, lw, theta) sample_gaussian(model, y, x, lw, theta, par_cf1_sampler), ...
    'calculate_weights', @calculate_weights ...
);

% Proposed method
par_cf_sampler = struct( ...
    'L', L, ...
    'slr', @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx) ...
);
par_cf = struct( ...
    'sample', @(model, y, x, lw, theta) sample_gaussian(model, y, x, lw, theta, par_cf_sampler), ...
    'calculate_weights', @calculate_weights ...
);

%% Monte Carlo simulations
% Preallocate
xs = zeros(1, N, K);
ys = zeros(1, N, K);

% MMSE
xhat_grid = zeros(1, N, K);
xhat_bpf = xhat_grid;
xhat_gf = xhat_grid;
xhat_pfpf = xhat_grid;
xhat_cf1 = xhat_bpf;
xhat_cf = xhat_bpf;
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;

% ESS
ess_bpf = zeros(1, N, K);
ess_gf = ess_bpf;
ess_pfpf = ess_bpf;
ess_cf1 = ess_bpf;
ess_cf = ess_bpf;
ess_lin = ess_bpf;
ess_sp = ess_bpf;

% Resampling
r_bpf = zeros(1, N+1, K);
r_gf = r_bpf;
r_pfpf = r_bpf;
r_cf1 = r_bpf;
r_cf = r_bpf;
r_lin = r_bpf;
r_sp = r_bpf;

% Computational time
t_grid = zeros(1, K);
t_bpf = t_grid;
t_gf = t_grid;
t_pfpf = t_grid;
t_cf1 = t_bpf;
t_cf = t_bpf;
t_lin = t_bpf;
t_sp = t_bpf;

% No. of iterations
l_cf = zeros(1, K);
l_lin = l_cf;
l_sp = l_cf;

if use_gridf
    NGrid = length(xg);
    w = zeros(N, NGrid, K);
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
    if use_gridf
        tic;
        [xhat_grid(:, :, k), w(:, :, k)] = gridf(model, ys(:, :, k), 1:N, xg);
        t_grid(k) = toc;
    end
    
    % Bootstrap PF
    if use_bpf
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).qstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
    end
    
    % Gaussian flow
    if use_gf
        tic;
        [xhat_gf(:, :, k), sys_gf] = pf(model, ys(:, :, k), theta, J, par_gf);
        t_gf(k) = toc;
        tmp = cat(2, sys_gf(2:N+1).rstate);
        ess_gf(:, :, k) = cat(2, tmp.ess);
    end
    
    % PFPF
    if use_pfpf
        tic;
        [xhat_pfpf(:, :, k), sys_pfpf] = pfpf(model, ys(:, :, k), theta, J, par_pfpf);
        t_pfpf(k) = toc;
        tmp = cat(2, sys_pfpf(2:N+1).rstate);
        ess_pfpf(:, :, k) = cat(2, tmp.ess);
    end
    
    % One-step OID approximation
    if use_cf1
        tic;
        [xhat_cf1(:, :, k), sys_cf1] = pf(model, ys(:, :, k), theta, J, par_cf1);
        t_cf1(k) = toc;
        qstates = cat(2, sys_cf1(2:N+1).qstate);
        rstates = cat(2, qstates.rstate);
        ess_cf1(:, :, k) = cat(2, rstates.ess);
    end
   
    % Closed form, iterated conditional expectations
    if use_cf
        tic;
        [xhat_cf(:, :, k), sys_cf] = pf(model, ys(:, :, k), theta, J, par_cf);
        t_cf(k) = toc;
        qstates = cat(2, sys_cf(2:N+1).qstate);
        rstates = cat(2, qstates.rstate);
        ess_cf(:, :, k) = cat(2, rstates.ess);
        qjs = cat(2, qstates.qj);
        l_cf(k) = mean(cat(1, qjs.l));
    end
    
    if use_lin
        % Taylor series
        tic;
        [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J, par_lin);
        t_lin(k) = toc;
        qstates = cat(2, sys_lin(2:N+1).qstate);
        rstates = cat(2, qstates.rstate);
        ess_lin(:, :, k) = cat(2, rstates.ess);
        qjs = cat(2, qstates.qj);
        l_lin(k) = mean(cat(1, qjs.l));
    end

    % Sigma-point approximation of SLR
    if use_sp
        tic;
        [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J, par_sp);
        t_sp(k) = toc;
        qstates = cat(2, sys_sp(2:N+1).qstate);
        rstates = cat(2, qstates.rstate);
        ess_sp(:, :, k) = cat(2, rstates.ess);
        qjs = cat(2, qstates.qj);
        l_sp(k) = mean(cat(1, qjs.l));
    end

    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance summary
iNaN_grid = squeeze(isnan(xhat_grid(1, N, :)));
iNaN_bpf = squeeze(isnan(xhat_bpf(1, N, :)));
iNaN_gf = squeeze(isnan(xhat_gf(1, N, :)));
iNaN_pfpf = squeeze(isnan(xhat_pfpf(1, N, :)));
iNaN_cf1 = squeeze(isnan(xhat_cf1(1, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(1, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(1, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(1, N, :)));

e_rmse_grid = trmse(xhat_grid(:, :, ~iNaN_grid) - xs(:, :, ~iNaN_grid));
e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_gf = trmse(xhat_gf(:, :, ~iNaN_gf) - xs(:, :, ~iNaN_gf));
e_rmse_pfpf = trmse(xhat_pfpf(:, :, ~iNaN_pfpf) - xs(:, :, ~iNaN_pfpf));
e_rmse_cf1 = trmse(xhat_cf1(:, :, ~iNaN_cf1) - xs(:, :, ~iNaN_cf1));
e_rmse_cf = trmse(xhat_cf(:, :, ~iNaN_cf) - xs(:, :, ~iNaN_cf));
e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\tESS\t\tIterations\n');
fprintf( ...
    'Grid\t%.2e (%.2e)\t%.2f (%.2f)\tn/a\t\tn/a\t\tn/a\t\tn/a\n', ...
    mean(e_rmse_grid), std(e_rmse_grid), mean(t_grid), std(t_grid) ...
);
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\tn/a\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K, ...
    mean(mean(ess_bpf)), std(mean(ess_bpf)) ...
);
fprintf( ...
    'GF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\tn/a\n', ...
    mean(e_rmse_gf), std(e_rmse_gf), mean(t_gf), std(t_gf), ...
    mean(sum(r_gf(:, :, ~iNaN_gf))/N), std(sum(r_gf(:, :, ~iNaN_gf))/N), ...
    1-sum(iNaN_gf)/K, ...
    mean(mean(ess_gf)), std(mean(ess_gf)) ...
);
fprintf( ...
    'PFPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\tn/a\n', ...
    mean(e_rmse_pfpf), std(e_rmse_pfpf), mean(t_pfpf), std(t_pfpf), ...
    mean(sum(r_pfpf(:, :, ~iNaN_pfpf))/N), std(sum(r_pfpf(:, :, ~iNaN_pfpf))/N), ...
    1-sum(iNaN_pfpf)/K, ...
    mean(mean(ess_pfpf)), std(mean(ess_pfpf)) ...
);
fprintf( ...
    'CF1\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\tn/a\n', ...
    mean(e_rmse_cf1), std(e_rmse_cf1), mean(t_cf1), std(t_cf1), ...
    mean(sum(r_cf1(:, :, ~iNaN_cf1))/N), std(sum(r_cf1(:, :, ~iNaN_cf1))/N), ...
    1-sum(iNaN_cf1)/K, ...
    mean(mean(ess_cf1)), std(mean(ess_cf1)) ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t%.1f (%.1f)\n', ...
    mean(e_rmse_cf), std(e_rmse_cf), mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N), std(sum(r_cf(:, :, ~iNaN_cf))/N), ...
    1-sum(iNaN_cf)/K, ...
    mean(mean(ess_cf)), std(mean(ess_cf)), ...
    mean(l_cf), std(l_cf) ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t%.1f (%.1f)\n', ...
    mean(e_rmse_lin), std(e_rmse_lin), mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K, ...
    mean(mean(ess_lin)), std(mean(ess_lin)), ...
    mean(l_lin), std(l_lin) ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t%.1f (%.1f)\n', ...
    mean(e_rmse_sp), std(e_rmse_sp), mean(t_sp), std(t_sp), ...
    mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
    1-sum(iNaN_sp)/K, ...
    mean(mean(ess_sp)), std(mean(ess_sp)), ...
    mean(l_sp), std(l_sp) ...
);

%% Plots
% For plots of a specific MC run, use the last one since we only have the 
% particle system for that one available (too much to store otherwise, 
% we'd run out of memory).
k = K;

% Show the true state and the MMSE
figure(1); clf();
subplot(211);
plot(ys(:, :, k)); grid on;
title('Data');
subplot(212);
plot(xs(:, :, k)); hold on;
plot(xhat_bpf(:, :, k));
plot(xhat_gf(:, :, k));
plot(xhat_pfpf(:, :, k));
plot(xhat_cf1(:, :, k));
plot(xhat_cf(:, :, k));
plot(xhat_lin(:, :, k));
plot(xhat_sp(:, :, k));
legend('State', 'Bootstrap', 'GF', 'PFPF', 'CF1', 'ICE-CF', 'Linearized', 'Sigma-Points');
title('MMSE');

% Mean ESS
figure(2); clf();
plot(mean(ess_bpf, 3)); hold on; grid on;
plot(mean(ess_gf, 3));
plot(mean(ess_pfpf, 3));
plot(mean(ess_cf1, 3));
plot(mean(ess_cf, 3));
plot(mean(ess_lin, 3));
plot(mean(ess_sp, 3));
legend('Bootstrap', 'GF', 'PFPF', 'CF1', 'ICE-PF (CF)', 'ICE-PF (LIN)', 'ICE-PF (SP)');
xlabel('n'); ylabel('ESS');
title('Effective sample size');

% Posteriors and importance density
for n = 1:N
    % Posterior
    figure(3); clf();
    if use_gridf
        plot(xg, w(n, :, k)); hold on; grid on;
        plot([xs(:, n, k), xs(:, n, k)], [0, 1]*max(w(n, :, k)));
    end

    if use_bpf
        x_bpf = sys_bpf(n+1).x;
        plot(x_bpf, 0*ones(1, J), '.'); hold on; grid on;
    end
    
    if use_gf
        x_gf = sys_gf(n+1).x;
        plot(x_gf, 0.025*ones(1, J), 'x');
    end

    if use_pfpf
        x_pfpf = sys_pfpf(n+1).x;
        plot(x_pfpf, 0.05*ones(1, J), 'o');
    end

    if use_cf1
        x_cf1 = sys_cf1(n+1).x;
        plot(x_cf1, 0.075*ones(1, J), '*');
    end

    if use_cf
        x_cf = sys_cf(n+1).x;
        plot(x_cf, 0.1*ones(1, J), '^');
    end
    legend('Grid', 'True state', 'BPF', 'GF', 'PFPF', 'CF1', 'ICE-CF');
    title(sprintf('Posterior and particles at n = %d', n));
    % ylim([0, 0.1]);

    % OID for a particular particle
    if use_cf
        % Get ancestor particle and proposal
        j = 1;
        xnj = sys_cf(n).x(:, sys_cf(n+1).alpha(j));
        qj = sys_cf(n+1).qstate.qj(j);

        lpx = model.px.logpdf(xg, xnj, theta(:, n));
        px = exp(lpx);

        lpy = model.py.logpdf(ys(:,  n, k), xg, theta(:, n));
        py = exp(lpy);

        lp_oid = lpy + lpx;
        p_oid = exp(lp_oid)/(sum(exp(lp_oid))*mean(diff(xg)));

        p_cf1 = normpdf(xg, qj.mean(2), sqrt(qj.cov(:, :, 2)));
        p_cf = normpdf(xg, qj.mean(qj.l+1), sqrt(qj.cov(:, :, qj.l+1)));

        figure(4); clf();
        plot(xg, p_oid); hold on;
        plot(xg, px, '--');
%         plot(xg, py, '--');
        plot(xg, p_cf1, '--');
        plot(xg, p_cf, '--');
        plot(xnj, 0, '*');
        legend('OID', 'px', 'CF1', 'CF');
        title(sprintf('OID approximation for particle %d from %d to %d', j, n-1, n));
    end

    pause();
end

%% [debug]
% Plot the posterior in a waterfall plot, together with the state trace and
% particles (for the kth MC run)
if plots && 0
    figure(3); clf();
    waterfall(xg, 1:N, w(:, :, k)); hold on; grid on;
    plot(xs(:, :, k), 1:N, '--r', 'LineWidth', 2);

    for n = 1:N
        if use_bpf
            x_bpf = sys_bpf(n+1).x;
            plot3(x_bpf, n*ones(1, J), zeros(1, J), '.');
        end

        if use_pfpf
            x_pfpf = sys_pfpf(n+1).x;
            plot3(x_pfpf, n*ones(1, J), zeros(1, J), 'x');
        end

        if use_cf1
            x_cf1 = sys_cf1(n+1).x;
            plot3(x_cf1, n*ones(1, J), zeros(1, J), 'o');
        end

        if use_cf
            x_cf = sys_cf(n+1).x;
            plot3(x_cf, n*ones(1, J), zeros(1, J), '*');
        end
    end
end
% [/debug]
    
%% [debug] Store results
% Do this before the plots so that we don't store a bunch of temporary
% variables that we don't need
if store
    if ~use_gridf
        % Remove these from the savefile; they make the file to explode
        % (but only if we don't simulate with the grid filter, which is
        % used for illustration purposes)
        clear sys_bpf sys_cf1 sys_gf sys_pfpf sys_cf
    end
        
    % Store
    outfile = sprintf('Save/iploid_ungm/example_ungm_J=%d_L=%d_K=%d_N=%d.mat', J, L, K, N);
    save(outfile);
end
% [/debug]