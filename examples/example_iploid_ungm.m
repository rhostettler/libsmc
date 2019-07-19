% Approximate Bayesian Computation (likelihood-free) example
% 
% Example using iterated conditional expectations for likelihood-free
% inference (i.e., with extremely narrow likelihood). The model considered
% is:
% 
%   x[n] = 0.5*x[n-1] + 25*x[n-1]/(1+x[n-1]^2) + 8*cos(1.2*n) + q[n]
%   y[n] = tanh(x[n]/20)
%   x[0} ~ N(0, 5)
%
% with q ~ N(0, 10). As a pseudo-likelihood, a Gaussian density is used
% with variance 1e-12. Note that the dynamic model corresponds to the UNGM
% dynamics, however, the measurement model is unimodal (the bimodality in
% the original model is difficult to evaluate).
%
% 2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
% * Update header
% * Add license
% * Clean up code

% Housekeeping
clear variables;
addpath ../src ../external
rng(5011);
if 0
spmd
    warning('off', 'all');
end
end
% warning('off', 'all');

%% Parameters
% Filter parameters
J = 250;       % Number of particles
L = 3;         % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-1/2));
beta = 1;
kappa = 0;

% Simulation parameters
N = 10;    % Number of time samples
K = 1;     % Number of MC simulations

% Model parameters
Q = 10;
R = 1;
m0 = 0;
P0 = 5;

% Save the simulation results (true/false)
store = false;

xg = linspace(-30, 30, 1000);

%% Model
% Dynamic and measurement function
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
g = @(x, theta) x.^2/20;
Gx = @(x, theta) 2*x/20;

% libsmc model structure
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0);

%% Sampling algorithms
% par_resample = struct('Jt', J+1);

% 
par_bpf = struct( ...
    ... 'resample', @(lw) resample_ess(lw, par_resample) ...
);

if 0
% Approximation of the optimal proposal using linearization
par_lin = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_taylor(y, x, theta, model, f, @(x, theta) Q, g, Gx, @(x, theta) R, L), ...
    'resample', @(lw) resample_ess(lw, par_resample) ...
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
for k = 1:K
    %% Model simulation
    [xs(:, :, k), ys(:, :, k)] = simulate_model(model, 1:N, N);

    %% Estimation
    % Grid filter
    tic;
    [xhat_grid(:, :, k), w] = gridfilter(model, ys(:, :, k), xg, 1:N);
    t_grid(k) = toc;
    
    % Bootstrap PF
    if 1 %L == 1
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), 1:N, J, par_bpf);
        t_bpf(k) = toc;
%         r_bpf(:, :, k) = cat(2, sys_bpf.r);
    end
    
if 0
    % Taylor series approximation of SLR
    tic;
    [xhat_lin(:, :, k)] = pf(y, 1:N, model, J, par_lin);
    t_lin(k) = toc;
%     r_lin(:, :, k) = cat(2, sys_lin.r);
end

    % Sigma-point approximation of SLR
if 1
    tic;
    [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), 1:N, J, par_sp);
    t_sp(k) = toc;
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

e_rmse_grid = trmse(xhat_grid - xs);
e_rmse_bpf = trmse(xhat_bpf - xs);
e_rmse_lin = trmse(xhat_lin - xs);
e_rmse_sp = trmse(xhat_sp - xs);

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\n');
fprintf( ...
    'Grid\t%.2e (%.2e)\t%.2f (%.2f)\tn/a\t\tn/a\n', ...
    mean(e_rmse_grid, 3), std(e_rmse_grid, [], 3), ...
    mean(t_grid), std(t_grid) ...
);
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_bpf, 3), std(e_rmse_bpf, [], 3), ...
    mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_lin, 3), std(e_rmse_lin, [], 3), ...
    mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(e_rmse_sp, 3), std(e_rmse_sp, [], 3), ...
    mean(t_sp), std(t_sp), ...
    mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
    1-sum(iNaN_sp)/K ...
);

%% Plots
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

%%
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

% view(0, -90);

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
%     set(gca, 'YScale', 'log');
    legend('Grid', 'True state', 'SP', 'BPF');
    ylim([-0.1, max(w(n, :))]);
    
    pause();
end

end

%% Store results
if store
    outfile = sprintf('Savefiles/example_abc_J=%d_L=%d.mat', J, L);
    save(outfile);
end



