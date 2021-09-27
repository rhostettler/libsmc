% Example comparing bootstrap PF with sequential MCMC for a high-
% dimensional LGSSM
%
% 2021-present -- Roland Hostettler

clear variables;
addpath ../src

%% Parameters
dx = 100;           % State dimension
tau = 1;            % Process noise parameter
lambda = 1;         % Process noise parameter
sigma2y = 0.25^2;   % Measurement noise variance
N = 10;             % No. of timesteps
K = 20;             % No. of Monte Carlo simulations

%% Filter parameters
% Bootstrap PF
J_bpf = 10;         % No. of particles for the bootstrap PF

% Fully adapted APF
par_apf = struct( ...
    'sample', @sample_fapf, ...
    'calculate_weights', [] ...
);

% SMCMC
J_mcmc = J_bpf;
par_mcmc = struct( ...
    'sample', @sample_smcmc_composite, ...
    'calculate_weights', [] ...
);
par_mcmc_bootstrap = struct( ...
    'sample', @sample_smcmc_bootstrap, ...
    'calculate_weights', [] ...
);

%% Model
m0 = zeros(dx, 1);
P0 = eye(dx);

F = 0.5*eye(dx);
Q = diag(-lambda*ones(dx-1, 1), -1) + diag((tau+2*lambda)*ones(dx, 1)) + diag(-lambda*ones(dx-1, 1), 1);
Q(1, 1) = tau + lambda;
Q(dx, dx) = tau + lambda;

G = eye(dx);
R = sigma2y*eye(dx);
dy = dx;

model = model_lgss(F, Q, G, R, m0, P0);
model.px.loggradient = @(xp, x, theta) -Q\(xp - F*x);
model.px.loghessian = @(xp, x, theta) -Q\eye(dx);
model.py.loggradient = @(y, x, theta) G'/R*(y-G*x);
model.py.loghessian = @(y, x, theta) -G'/R*G;

%% Generate data
x = zeros(dx, N, K);
y = zeros(dy, N, K);
rng(1245);
for k = 1:K
    [x(:, :, k), y(:, :, k)] = simulate_model(model, [], N);
end
    
%% Estimate
rng(1892);
e_bpf = zeros(dx, N, K);
e_apf = e_bpf;
e_mcmc = e_bpf;
e_mcmc_bootstrap = e_bpf;
e_kf = e_bpf;

t_bpf = zeros(1, K);
t_apf = t_bpf;
t_mcmc = t_bpf;
t_mcmc_bootstrap = t_bpf;
t_kf = t_bpf;

for k = 1:K    
    % BPF
    tstart = tic;
    xhat_bpf = pf(model, y(:, :, k), [], J_bpf);
    t_bpf(k) = toc(tstart);
    e_bpf(:, :, k) = xhat_bpf - x(:, :, k);

    % FAPF
    tstart = tic;
    xhat_apf = pf(model, y(:, :, k), [], J_bpf, par_apf);
    t_apf(k) = toc(tstart);
    e_apf(:, :, k) = xhat_apf - x(:, :, k);
    
    % SMCMC (Bootstrap)
    tstart = tic;
    xhat_mcmc_bootstrap = pf(model, y(:, :, k), [], J_mcmc, par_mcmc_bootstrap);
    t_mcmc_bootstrap(k) = toc(tstart);
    e_mcmc_bootstrap(:, :, k) = xhat_mcmc_bootstrap - x(:, :, k);

    % SMCMC
    tstart = tic;
    xhat_mcmc = pf(model, y(:, :, k), [], J_mcmc, par_mcmc);
    t_mcmc(k) = toc(tstart);
    e_mcmc(:, :, k) = xhat_mcmc - x(:, :, k);

    % KF
    tstart = tic;
    xhat_kf = kf_loop(m0, P0, G, R, y(:, :, k), F, Q);
    t_kf(k) = toc(tstart);
    e_kf(:, :, k) = xhat_kf - x(:, :, k);
end

%% Performance
e_rmse_bpf = trmse(e_bpf);
e_rmse_apf = trmse(e_apf);
e_rmse_mcmc = trmse(e_mcmc);
e_rmse_mcmc_bootstrap = trmse(e_mcmc_bootstrap);
e_rmse_kf = trmse(e_kf);

fprintf('\tRMSE\t\t\tTime\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf) ...
);
fprintf( ...
    'APF\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_apf), std(e_rmse_apf), mean(t_apf), std(t_apf) ...
);
fprintf( ...
    'Bootstrap\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_mcmc_bootstrap), std(e_rmse_mcmc_bootstrap), mean(t_mcmc_bootstrap), std(t_mcmc_bootstrap) ...
);
fprintf( ...
    'Composite\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_mcmc), std(e_rmse_mcmc), mean(t_mcmc), std(t_mcmc) ...
);
fprintf( ...
    'KF\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_kf), std(e_rmse_kf), mean(t_kf), std(t_kf) ...
);
