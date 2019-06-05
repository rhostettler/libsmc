% Example of particle Gibbs using conditional particle filter with 
% rejection-sampling-based ancestor sampling
%
% 
%
% 2017-2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath(genpath('../src'));
rng(511);

%% Parameters
N = 100;    % No. of time samples
J = 500;    % No. of particles
K = 100;    % No of MC samples

%% Model
m0 = zeros(2, 1);
P0 = eye(2);
F = [
    1, 1;
    0, 1;
];
Q = 0.25*eye(2);
G = [0.25, 0];
R = 1;
model = model_lgssm(F, Q, G, R, m0, P0);

model.px.fast = 0; % if we don't use this, it's slower :(
model.px.kappa = model.px.pdf(zeros(2, 1), zeros(2, 1), 0);  % TODO: This should be solved more elegantly
model = @(theta) model;

%%

% Proposal (locally optimal proposal)
% TODO: Needs to be added
if 0
S = G*Q*G' + R;
L = Q*G'/S;
mu = @(x, y) F*x + L*(y - G*F*x);
Sigma = Q - L*S*L';
q.fast = 1;
q.rand = @(y, x, theta) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, theta) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
end

% CPF parameters
par_cpf = struct();
par_cpf.sample_ancestor_index = @sample_ancestor_index_rs;

% Particle Gibbs parameters
par = struct();
par.sample_states = @(y, t, xt, theta) cpfas(model(theta), y, xt, [], J, par_cpf);

%% 
xs = zeros(size(m0, 1), N);
y = zeros(1, N);

%% Simulate System
x = m0 + chol(P0).'*randn(2, 1);
for n = 1:N
    qn = chol(Q).'*randn(size(Q, 1), 1);
    x = F*x + qn;
    rn = chol(R).'*randn(size(R, 1), 1);
    y(:, n) = G*x + rn;
    xs(:, n) = x;
end

%% Estimate
% KF
%tic;
%xhat_kf = kf(y, F, G, Q, R, m0, P0);
% [m, P, mp, Pp] = kf(y, F, G, Q, R, m0, P0);
% xhat_kf = rtss(F, m, P, mp, Pp);
%t_kf = toc;

% CPF-AS smoother
tic;
[x_sm, sys_sm] = gibbs_pmcmc(y, [], model, [], K, par);
x_sm = x_sm(:, 2:end, :);
xhat_sm = mean(x_sm, 3);
t_sm = toc;

%% Illustrate
figure(1); clf();
set(gcf, 'name', 'Trajectories');
figure(2); clf();
set(gcf, 'name', 'Distribution at XXX');
for i = 1:2
    figure(1);
    subplot(2, 1, i);
    plot(squeeze(x_sm(i, :, :)), 'Color', [0.9, 0.9, 0.9]); hold on;
    plot(xhat_sm(i, :), 'r', 'LineWidth', 2);
    plot(xs(i, :), 'k', 'LineWidth', 2);
    title('Posterior Mean (red), True Trajectory (black), Sampled Trajectories (grey)');
    
    figure(2);
    subplot(1, 2, i);
    hist(squeeze(x_sm(i, 41, :)), 20);
end

%% Stats
trms = @(e) sqrt(mean(sum(e.^2, 1), 2));
fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', K, J);
fprintf('\tRMSE\t\tTime\n');
fprintf('\t----\t\t----\n');
% fprintf('KF\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
%     mean(trms(xhat_kf-xs)), std(trms(xhat_kf-xs)), mean(t_kf), std(t_kf));
fprintf('CPF-AS\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
    mean(trms(xhat_sm-xs)), std(trms(xhat_sm-xs)), mean(t_sm), std(t_sm));
