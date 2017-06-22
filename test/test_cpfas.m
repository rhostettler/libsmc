% 
%
% test_cpfas.m -- 2017-04-07
% Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath lib;

%% Parameters
% No. of time samples
N = 100;

% No. of particles
M = 200;

K = 1;

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
model = lgss_model(F, Q, G, R, m0, P0);
model.px.fast = 0;
%model.px.rho = model.px.pdf(zeros(2, 1), zeros(2, 1), 0);

%% Proposal
% Optimal proposal
S = G*Q*G' + R;
L = Q*G'/S;
mu = @(x, y) F*x + L*(y - G*F*x);
Sigma = Q - L*S*L';
q.fast = 1;
q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
q.bootstrap = 0;   %%%%%% TODO: This should be set automatically if omitted

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
t = 1:N;

% KF
tic;
%xhat_kf = kf(y, F, G, Q, R, m0, P0);
[m, P, mp, Pp] = kf(y, F, G, Q, R, m0, P0);
xhat_kf = rtss(F, m, P, mp, Pp);
t_kf = toc;

% CPF-AS
if 0
tic;
[~, ~, sys_cpfas] = cpfas(y, t, model, q, M);
t_cpfas = toc;

% CPF-AS
tic;
[~, sys_pgas] = pgas(y, t, model, q, M);
t_cpfas = toc;
end

% CPF-AS smoother
par = struct();
par.filter = @(y, t, model, q, M, par) pgas(y, t, model, q, M, par);
par.Kburnin = 0;
par.Kmixing = 1;
tic;
[xhat_sm, sys_sm] = cpfas_ps(y, t, model, q, M, 10, par);
t_sm = toc;

%% Illustrate
figure(1); clf();
set(gcf, 'name', 'Trajectories');
for i = 1:2
    subplot(2, 1, i);
%    plot(t, squeeze(sys_cpfas.xf(i, :, :)), 'b'); hold on;
    plot(t, squeeze(sys_sm.xs(i, :, :)), 'r');
    plot(t, xs(i, :), 'k', 'LineWidth', 2);
    title('CPF Trajectories (blue), Smoother Trajectories (red), True Trajectory (black)');
end

%% Stats
trms = @(e) sqrt(mean(sum(e.^2, 1), 2));
fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', K, M);
fprintf('\tRMSE\t\tTime\n');
fprintf('\t----\t\t----\n');
fprintf('KF\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
    mean(trms(xhat_kf-xs)), std(trms(xhat_kf-xs)), mean(t_kf), std(t_kf));
fprintf('CPF-AS\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
    mean(trms(xhat_sm-xs)), std(trms(xhat_sm-xs)), mean(t_sm), std(t_sm));
