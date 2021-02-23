% Example comparing bootstrap PF with sequential MCMC for a 1D LGSSM
%
% 2021-present -- Roland Hostettler

clear variables;
addpath ../src

%% Parameters
Ts = 0.1;           % Sampling time
N = 20;             % No. of datapoints
K = 100;            % No. of Monte Carlo simulations

%% Filter parameters
% Bootstrap PF
J_bpf = 100;         % No. of particles for the bootstrap PF

% SMCMC
J_mcmc = J_bpf;     % No. of samples
par_mcmc = struct( ...
    'sample', @sample_composite ...
);

%% Model
m0 = 0;
P0 = 1;
F = 1;
Q = 0.1*Ts;
G = 1;
R = 0.1^2;
dx = 1;
dy = 1;

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
e_mcmc = zeros(dx, N, K);

t_bpf = zeros(1, K);
t_mcmc = zeros(1, K);

for k = 1:K
    % BPF
    tstart = tic;
    xhat_bpf = pf(model, y(:, :, k), [], J_bpf);
    t_bpf(k) = toc(tstart);
    e_bpf(:, :, k) = xhat_bpf - x(:, :, k);
    
    % SMCMC
    tstart = tic;
    [xhat_mcmc, sys] = smcmc(model, y(:, :, k), [], J_mcmc, par_mcmc);
    t_mcmc(k) = toc(tstart);
    e_mcmc(:, :, k) = xhat_mcmc - x(:, :, k);
end

%% Performance
e_rmse_bpf = trmse(e_bpf);
e_rmse_mcmc = trmse(e_mcmc);

fprintf('\tRMSE\t\t\tTime\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf) ...
);
fprintf( ...
    'SMCMC\t%.2e (%.2e)\t%.2f (%.2f)\n', ...
    mean(e_rmse_mcmc), std(e_rmse_mcmc), mean(t_mcmc), std(t_mcmc) ...
);
