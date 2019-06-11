% 
%
% 2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath(genpath('../src'));
rng(511);

%% Parameters
N = 100;        % No. of time samples
J = 250;        % No. of particles
K = 50;         % No. of MCMC samples
Ts = 0.025;     % Sampling time, only needed for tracking example

%% Model
% UNGM
Nx = 1;
Ny = 1;
Q = 10;
R = 1;
m0 = 0;
P0 = 5;
f = @(x, n) 0.5*x(:, :, end) + 25*x./(1+x(:, :, end).^2) + 8*cos(1.2*n);
g = @(x, n) 0.05*x(:, :, end).^2;

LP0 = sqrt(5);
px0 = struct( ...
    'rand', @(J) m0*ones(1, J) + LP0*randn(Nx, J) ...
);
LQ = sqrt(Q).';
px = struct( ...
    'logpdf', @(xp, x, theta) logmvnpdf(xp(:, 1).', f(x(:, :, end), theta(size(x, 3)+1)).', Q.').', ...
    'rand', @(x, theta) f(x(:, :, end), theta(size(x, 3)+1)) + LQ*randn(Nx, size(x, 2)), ...
    'fast', false ...
);
LR = sqrt(R).';
py = struct( ...
    'logpdf', @(y, x, theta) logmvnpdf(y(:, end).', g(x(:, :, end), theta(size(x, 3))).', R.').', ...
    'rand', @(x, theta) g(x(:, :, end), theta(size(x, 3))) + LR*randn(Nx, size(x, 2)), ...
    'fast', false ...
);
model = struct('px0', px0, 'px', px, 'py', py);

% model.px.kappa = model.px.pdf(0, 0, []); %%%%%% TODO: Is this a mistake? Isn't there a time-dependency here, i.e., should kappa be time-dependent?
model = @(theta) model; % TODO: This is again the gibbs_pmcmc() curiosity that needs sorting out

%% Algorithm parameters
% Parameters for categorical sampling
par = struct();
par.sample_states = @(y, xtilde, theta, lambda) cpfas2(model(theta), y, xtilde, lambda, J);

%% Simulate System
mod = model([]);
x = m0 + chol(P0).'*randn(Nx, 1);
for n = 1:N
    qn = LQ*randn(size(Q, 1), 1);
    x = f(x, n) + qn;
    rn = chol(R).'*randn(size(R, 1), 1);
    y(:, n) = g(x, n) + rn;
    xs(:, n) = x;
end

%% Estimation
tic;
[x_cs, ~, sys] = gibbs_pmcmc(model, y, [], 1:N, K, par);
t_cs = toc;
x_cs = x_cs(:, 2:end, :);
xhat_cs = mean(x_cs, 3);

%% Illustrate
figure(1); clf();
plot(squeeze(x_cs), 'Color', [0.9, 0.9, 0.9]); hold on;
plot(xhat_cs, 'r', 'LineWidth', 2);
plot(xs, 'k', 'LineWidth', 2);
title('Posterior Mean (red), True Trajectory (black), Sampled Trajectories (grey)');
