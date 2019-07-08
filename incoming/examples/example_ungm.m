% UNGM example
%
% 
% Model:
%   x[n] = 0.5*x[n-1] + 25*x[n-1]/(1+x[n-1]^2) + 8*cos(1.2*n) + q[n]
%   y[n] = 0.05*x[n]^2 + r[n]
%
% Q = 10, R = 1, x[0] ~ N(0, 5) 
% 
% TODO:
%   * Use different sigma-points

% Housekeeping
clear variables;
addpath(genpath('../src'));
rng(5011);

%% Parameters
% Filter parameters
J = 100;       % Number of particles
L = 3;       % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-0.5));
beta = 1;
kappa = 0;

% Simulation parameters
N = 1000;    % Number of time samples
K = 100;      % Number of MC simulations TODO: Not implemented yet!

% Model parameters
Q = 10;
R = 1;
m0 = 0;
P0 = 5;

%% Model
f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
g = @(x, n) 0.05*x.^2;
Gx = @(x, n) 0.1*x;
% f = @(x, n) atan(x);
% g = @(x, n) abs(x);
% Gx = @(x, n) sign(x);
% f = @(x, n) x;
% g = @(x, n) x;
% Gx = @(x, n) 1;

% libsmc-type model
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0);
model.px.fast = false;
model.py.fast = false;

%% Algorithm parameters
% Approximation of the optimal proposal using linearization
par_lin = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_taylor(y, x, theta, model, f, Q, g, Gx, R) ...
);

% SLR using sigma-points
Nx = size(m0, 1);
Xi = ut_sigmas(zeros(Nx, 1), eye(Nx), Nx);
[wm, wc] = ut_weights(Nx, alpha, beta, kappa);
par_sp = struct( ...
    'update', @(y, x, theta, model) sis_update_gaussian_sp(y, x, theta, model, f, @(x, theta) Q, g, @(x, theta) R, L, Xi, wm, wc) ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);

xhat_bpf = zeros(1, N, K);
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
t_bpf = zeros(1, K);
t_lin = t_bpf;
t_sp = t_bpf;


for k = 1:K
    %% Simulation
    y = zeros(1, N);

    % Simulate
    x = m0 + sqrt(P0)*randn(1);
    for n = 1:N
        q = sqrt(Q)*randn(1);
        x = f(x, n) + q;
        r = sqrt(R)*randn(1);

        y(:, n) = g(x, n) + r;
        xs(:, n, k) = x;
    end

    %% Estimation
    % Bootstrap PF
    tic;
    xhat_bpf(:, :, k) = pf(y, 1:N, model, J);
    t_bpf(k) = toc;

    % Approximation of the optimal proposal using Taylor series linearization,
    % no iterations
    tic;
    xhat_lin(:, :, k) = pf(y, 1:N, model, J, par_lin);
    t_lin(k) = toc;

    % SLR using sigma-points, L iterations
    tic;
    xhat_sp(:, :, k) = pf(y, 1:N, model, J, par_sp);
    t_sp(k) = toc;
end

%% RMS
mean(squeeze([rms(xs-xhat_bpf), rms(xs-xhat_lin), rms(xs-xhat_sp)]), 2).'
mean([t_bpf; t_lin; t_sp], 2).'

%% Plots
if 1
figure(1); clf();
plot(xs); hold on;
plot(xhat_bpf);
plot(xhat_lin);
plot(xhat_sp);
legend('State', 'Bootstrap', 'Linearized', 'Sigma-Points');
end
