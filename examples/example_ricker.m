% Ricker Example (Iterated Conditional Expectations)
%
% 2018 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath(genpath('../src'));
rng(5011);

%% Parameters
% Filter parameters
J = 1000;         % Number of particles
L = 5;          % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-0.5));
beta = 1;
kappa = 0;

% Simulation parameters
N = 1000;       % Number of time samples
K = 100;        % Number of MC simulations

% Model parameters
Q = 0.3^2;
m0 = log(7);
P0 = 0.1;

% Save the simulation?
store = true;

%% Model
f = @(x, n) log(44.7) + x - exp(x);
g = @(x, n) 10*exp(x);
R = @(x, n) 10*exp(x);

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

%% Algorithm parameters
% SLR using unscented transform
Nx = size(m0, 1);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), Nx);
[wm, wc] = ut_weights(Nx, alpha, beta, kappa);

par_sp = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_sp(y, x, theta, model, f, @(x, theta) Q, g, R, L, Xi, wm, wc) ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);
xhat_bpf = zeros(1, N, K);
xhat_sp = xhat_bpf;
t_bpf = zeros(1, K);
t_sp = t_bpf;

fh = pbar(K);
for k = 1:K
    %% Simulation
    y = zeros(1, N);

    % Simulate
    x = m0 + sqrt(P0)*randn(1);
    for n = 1:N
        x = px.rand(x, n);
        y(:, n) = py.rand(x, n);
        xs(:, n, k) = x;
    end

    %% Estimation
    % Bootstrap PF
    tic;
    xhat_bpf(:, :, k) = pf(y, 1:N, model, J);
    t_bpf(k) = toc;

    % SLR using sigma-points, L iterations
    tic;
    xhat_sp(:, :, k) = pf(y, 1:N, model, J, par_sp);
    t_sp(k) = toc;
    
    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Results
iNaN_bpf = squeeze(isnan(xhat_bpf(:, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(:, N, :)));

fprintf('\tRMSE\t\tTime\t\tDivergence\n');
fprintf( ...
    'BPF\t%.2f (%.2f)\t%.2f (%.2f)\t%d\n', ...
    mean(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), 3), ...
    std(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), [], 3), ...
    mean(t_bpf), std(t_bpf), sum(iNaN_bpf) ...
);
fprintf( ...
    'ICE\t%.2f (%.2f)\t%.2f (%.2f)\t%d\n', ...
    mean(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), 3), ...
    std(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), [], 3), ...
    mean(t_sp), std(t_sp), sum(iNaN_sp) ...
);

%% Plots
if 0
figure(1); clf();
plot(xs); hold on;
plot(xhat_bpf);
plot(xhat_sp);
legend('State', 'Bootstrap', 'Sigma-Points');
end

%% Store results
if store
    outfile = sprintf('Savefiles/example_ricker_J=%d_L=%d.mat', J, L);
    save(outfile);
end
