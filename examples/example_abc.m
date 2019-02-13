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

% Housekeeping
clear variables;
addpath(genpath('../src'));
rng(5011);
spmd
    warning('off', 'all');
end
warning('off', 'all');

%% Parameters
% Filter parameters
J = 100;       % Number of particles
L = 1;         % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-1/2));
beta = 1;
kappa = 0;

% Simulation parameters
N = 1000;    % Number of time samples
K = 100;     % Number of MC simulations

% Model parameters
Q = 10;
R = 1e-6;
m0 = 0;
P0 = 5;

% Save the simulation results (true/false)
store = false;

%% Model
% Dynamic and measurement function
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
% g = @(x, theta) x.^2/20;
% Gx = @(x, theta) 2*x/20;

g = @(x, theta) x.^2/20 + x;
Gx = @(x, theta) 2*x/20 + 1;

% g = @(x, theta) tanh(x/2);
% Gx = @(x, theta) 1/10*(1-tanh(x/10).^2);

% libsmc model structure
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0);

%% Sampling algorithms
% Approximation of the optimal proposal using linearization
par_lin = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_taylor(y, x, theta, model, f, @(x, theta) Q, g, Gx, @(x, theta) R, L) ...
);

% SLR using sigma-points
Nx = size(m0, 1);
[wm, wc, c] = ut_weights(Nx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), c);
par_sp = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_sp(y, x, theta, model, f, @(x, theta) Q, g, @(x, theta) R, L, Xi, wm, wc) ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);
ys = zeros(1, N, K);
xhat_bpf = zeros(1, N, K);
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
r_bpf = zeros(1, N+1, K);
r_lin = zeros(1, N+1, K);
r_sp = zeros(1, N+1, K);
t_bpf = zeros(1, K);
t_lin = t_bpf;
t_sp = t_bpf;

% Simulate
fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);
parfor k = 1:K
    %% Model simulation
    y = zeros(1, N);
    x = m0 + sqrt(P0)*randn(1);
    for n = 1:N
        q = sqrt(Q)*randn(1);
        x = f(x, n) + q;
        y(:, n) = g(x, n);
        
        xs(:, n, k) = x;
    end
    ys(:, :, k) = y;

    %% Estimation
    % Bootstrap PF
    if L == 1
        tic;
        [xhat_bpf(:, :, k)] = pf(y, 1:N, model, J);
        t_bpf(k) = toc;
%         r_bpf(:, :, k) = cat(2, sys_bpf.r);
    end
    
if 1
    % Taylor series approximation of SLR
    tic;
    [xhat_lin(:, :, k)] = pf(y, 1:N, model, J, par_lin);
    t_lin(k) = toc;
%     r_lin(:, :, k) = cat(2, sys_lin.r);
end

    % Sigma-point approximation of SLR
if 1
    tic;
    [xhat_sp(:, :, k)] = pf(y, 1:N, model, J, par_sp);
    t_sp(k) = toc;
%     r_sp(:, :, k) = cat(2, sys_sp.r);
end
    
    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance summary
iNaN_bpf = squeeze(isnan(xhat_bpf(:, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(:, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(:, N, :)));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), 3), ...
    std(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), [], 3), ...
    mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_lin) - xhat_lin(:, :, ~iNaN_lin)), 3), ...
    std(rms(xs(:, :, ~iNaN_lin) - xhat_lin(:, :, ~iNaN_lin)), [], 3), ...
    mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), 3), ...
    std(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), [], 3), ...
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

%% Store results
if store
    outfile = sprintf('Savefiles/example_abc_J=%d_L=%d.mat', J, L);
    save(outfile);
end
