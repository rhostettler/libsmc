% UNGM Approximate Bayesian Computation (likelihood-free) example
% 
% Example using iterated conditional expectations for likelihood-free
% inference (i.e., with extremely narrow likelihood). The model considered
% is the univariate nonlinear growth model (UNGM):
% 
%   x[n] = 0.5*x[n-1] + 25*x[n-1]/(1+x[n-1]^2) + 8*cos(1.2*n) + q[n]
%   y[n] = x[n]^2/20
%   x[0} ~ N(0, 5)
%
% with q ~ N(0, 10). As a pseudo-likelihood, a Gaussian density is used
% with variance 1e-12. Note that the dynamic model corresponds to the UNGM
% dynamics, however, the measurement model is unimodal (the bimodality in
% the original model is difficult to evaluate).
% with q ~ N(0, 10).
% 
% (TODO: UPDATE THIS) As a pseudo-likelihood, a Gaussian density is used
% with variance 1e-12. This corresponds to the standard UNGM model but with
% a uniform likelihood rather than a Gaussian, which is also much more
% narrow than the original Gaussian. Hence, the model here is much trickier
% than the standard UNGM.
%
% 2019-present -- Roland Hostettler

% TODO:
% * Update header
% * Add license
% * Clean up code
% * Replace the Gaussian likelihood with the uniform and update the text
% above accordingly.
% * Add PPPF or similar for comparison

% Housekeeping
clear variables;
addpath ../src;
addpath ../../gp-pmcmc/lib % TODO: gp_plot should be an external dependency
rng(5011);
if 0
spmd
    warning('off', 'all');
end
end
% warning('off', 'all');

%% Parameters
% Grid for the grid filter
xg = -25:1e-2:25;

% Common particle filter parameters
J = 250;       % Number of particles
L = 3;         % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-1/2));
beta = 1;
kappa = 0;

% Simulation parameters
N = 10;   % Number of time samples
K = 1;     % Number of MC simulations

% Model parameters
Q = 10;
R = 1e-4;
m0 = 0;
P0 = 5;

% If set to true, measurements are noise-free and a uniform
% pseudo-likelihood with variance R is used rather than the true Gaussian
abc = true;

% Save the simulation results (true/false)
store = false;

%% Model
% Dynamic and measurement function
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
g = @(x, theta) x.^2/20;
Gx = @(x, theta) 2*x/20;

% libsmc model structure
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0, true);
if abc
    epsilon = sqrt(12*R)/2; % To match variance with Gaussian used previously
    model.py = struct( ...
        'fast', true, ...
        'rand', @(x, theta) g(x, theta) - epsilon + epsilon*rand(1, size(x, 2)), ...
        'logpdf', @(y, x, theta) log(unifpdf(y-g(x, theta), -epsilon, epsilon)) ...
    );
end

%% Sampling algorithms
% Resampling parameter: Set ESS threshold to J+1 to enforce resampling at
% every time step (makes for a more fair comparison of the ESS)
% par_resample = struct('Jt', J+1);

% 
par_bpf = struct( ...
    ... %'resample', @(lw) resample_ess(lw, par_resample) ...
);

% Approximation of the optimal proposal using linearization
if 0
par_lin = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_taylor(y, x, theta, model, f, @(x, theta) Q, g, Gx, @(x, theta) R, L), ...
    'resample', @(lw) resample_ess(lw, par_resample), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);
end

% SLR using sigma-points
Nx = size(m0, 1);
[wm, wc, c] = ut_weights(Nx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), c);
par_sp = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian_sp(model, y, x, theta, f, @(x, theta) Q, g, @(x, theta) R, L, Xi, wm, wc), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic... , ...
... %     'resample', @(lw) resample_ess(lw, par_resample) ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);
ys = zeros(1, N, K);
xhat_grid = zeros(1, N, K);
xhat_bpf = xhat_grid;
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
ess_bpf = zeros(1, N, K);
ess_lin = ess_bpf;
ess_sp = ess_bpf;
r_bpf = zeros(1, N+1, K);
r_lin = zeros(1, N+1, K);
r_sp = zeros(1, N+1, K);
t_grid = zeros(1, K);
t_bpf = zeros(1, K);
t_lin = t_bpf;
t_sp = t_bpf;

% Simulate
fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);

% In this example, the additional parameter is time
theta = 1:N;

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
    tic;
    [xhat_grid(:, :, k), w] = gridf(model, ys(:, :, k), 1:N, xg);
    t_grid(k) = toc;
    
    % Bootstrap PF
    if 1 %L == 1
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J, par_bpf);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).rstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
%         r_bpf(:, :, k) = cat(2, sys_bpf.r);
    end
    
if 0
    % Taylor series approximation of SLR
    tic;
    [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J, par_lin);
    t_lin(k) = toc;
%     r_lin(:, :, k) = cat(2, sys_lin.r);
end

    % Sigma-point approximation of SLR
if 1
    tic;
    [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J, par_sp);
    t_sp(k) = toc;
    tmp = cat(2, sys_sp(2:N+1).rstate);
    ess_sp(:, :, k) = cat(2, tmp.ess);
%     r_sp(:, :, k) = cat(2, sys_sp.r);
end
    
    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance summary
iNaN_grid = squeeze(isnan(xhat_grid(1, N, :)));
iNaN_bpf = squeeze(isnan(xhat_bpf(1, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(1, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(1, N, :)));

e_rmse_grid = trmse(xhat_grid(:, :, ~iNaN_grid) - xs(:, :, ~iNaN_grid));
e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\n');
fprintf( ...
    'Grid\t%.2e (%.2e)\t%.2f (%.2f)\tn/a\t\tn/a\n', ...
    mean(e_rmse_grid), std(e_rmse_grid), mean(t_grid), std(t_grid) ...
);
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

%% Plots
% TODO: Clean these up; not all of them are used anymore.
if K == 1
    figure(1); clf();
    plot(xs); hold on;
    plot(xhat_bpf);
    plot(xhat_lin);
    plot(xhat_sp);
    legend('State', 'Bootstrap', 'Linearized', 'Sigma-Points');
    title('MMSE');

    figure(2); clf();
    plot(xs - xhat_bpf); hold on;
    plot(xs - xhat_lin);
    plot(xs - xhat_sp);
    legend('Bootstrap', 'Linearized', 'Sigma-Points');
    title('Error');

    figure(3); clf();
%     plot(xs); hold on;
    plot(ys);
    title('Data');
end

% more plots
if K == 1
    figure(11); clf();
    waterfall(xg, 1:N, w); hold on;
    plot(xs, 1:N, '--r', 'LineWidth', 2);

    for n = 1:N
        x_sp = sys_sp(n).x;
        [x_sp, i_sp] = sort(x_sp);
        w_sp = sys_sp(n).w(i_sp);
        plot3(x_sp, n*ones(1, J), w_sp);
    end

    % 
    for n = 1:N
        figure(12); clf();
        plot(xg, w(n, :)); hold on;
        plot([xs(n), xs(n)], [0, 1]*max(w(n, :)));

        x_sp = sys_sp(n+1).x;
        [x_sp, i_sp] = sort(x_sp);
        w_sp = sys_sp(n+1).w(i_sp);

    %     plot(x_sp, w_sp, 'o');
        plot(x_sp, zeros(1, J), 'o');

        x_bpf = sys_bpf(n+1).x;
        [x_bpf, i_bpf] = sort(x_bpf);
        w_bpf = sys_bpf(n+1).w(i_bpf);

    %     plot(x_bpf, w_bpf, 'o');
        plot(x_bpf, zeros(1, J), '.');
%         set(gca, 'YScale', 'log');
        legend('Grid', 'True state', 'SP', 'BPF');
       ylim([-0.01, max(w(n, :))]);

        pause();
    end
end

% ESS
figure(3); clf();
plot(theta, mean(ess_bpf, 3)); hold on; grid on;
plot(theta, mean(ess_sp, 3));
% gp_plot(theta, mean(ess_bpf, 3), var(ess_bpf, [], 3)); hold on; grid on;
% gp_plot(theta, mean(ess_sp, 3), var(ess_bpf, [], 3));
legend('BPF', 'ICE-PF (SP)');
title('Effective sample size');

%% Store results
if store
    outfile = sprintf('Savefiles/example_abc_J=%d_L=%d.mat', J, L);
    save(outfile);
end
