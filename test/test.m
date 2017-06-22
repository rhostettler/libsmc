% Test & comparison of the different SMC algorithms
%
% test.m -- 2017-03-27
% Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath lib;

%% Parameters
% No. of time samples
N = 100;

% No. of Monte Carlo runs
K = 4;

% No. of particles
M = 200;

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

% Model struct
model = lgss_model(F, Q, G, R, m0, P0);

%% Proposal
S = G*Q*G' + R;
L = Q*G'/S;
mu = @(x, y) F*x + L*(y - G*F*x);
Sigma = Q - L*S*L';
q.fast = 1;
q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';

%% 
xs = zeros(size(m0, 1), N, K);
xhat_kf = xs;
xhat_rts = xs;
xhat_bpf = xs;
xhat_sisr = xs;
xhat_cpfas = xs;
xhat_ksd = xs;
xhat_ffbsi = xs;
y = zeros(1, N, K);
t_kf = zeros(1, K);
t_rts = t_kf;
t_bpf = t_kf;
t_sisr = t_kf;
t_ksd = t_kf;
t_ffbsi = t_kf;
t_cpfas = t_kf;

t = 1:N;

for k = 1:K
    %% Simulate System
    x = m0 + chol(P0).'*randn(2, 1);
    for n = 1:N
        qn = chol(Q).'*randn(size(Q, 1), 1);
        x = F*x + qn;
        rn = chol(R).'*randn(size(R, 1), 1);
        y(:, n, k) = G*x + rn;
        xs(:, n, k) = x;
    end

    %% Estimate
    % KF
    tic;
    xhat_kf(:, :, k) = kf(y(:, :, k), F, G, Q, R, m0, P0);
    t_kf(k) = toc;
    
    % RTS
    tic;
    [m, P, mp, Pp] = kf(y(:, :, k), F, G, Q, R, m0, P0);
    xhat_rts(:, :, k) = rtss(F, m, P, mp, Pp);
    t_rts(k) = toc;
    
    % Optimal proposal PF
    tic;
    [xhat_sisr(:, :, k)] = sisr_pf(y(:, :, k), 1:N, model, q, M);
    t_sisr(k) = toc;
    
    % Bootstrap PF
    tic;
    [xhat_bpf(:, :, k)] = bootstrap_pf(y(:, :, k), 1:N, model, M);
    t_bpf(k) = toc;
        
    % Kronander-Sch?n-Dahlin smoother
    tic;
    [xhat_ksd(:, :, k)] = ksd_ps(y(:, :, k), 1:N, model, 2*M, M);
    t_ksd(k) = toc;
    
    % FFBSi smoother
    tic;
    xhat_ffbsi(:, :, k) = ffbsi_ps(y(:, :, k), 1:N, model, 2*M, M);
    t_ffbsi(k) = toc;
    
    % CPF-AS MCMC smoother
    par.Nmixing = 1;
    tic;
    xhat_cpfas(:,:, k) = cpfas_ps(y(:, :, k), t, model, [], 2*M, 10, par);
    t_cpfas(k) = toc;
end

%% Stats
trms = @(e) mean(sqrt(sum(e.^2, 1)), 2);
fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', K, M);
fprintf('\tRMSE\t\tTime\n');
fprintf('\t----\t\t----\n');
fprintf('KF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_kf-xs)), std(trms(xhat_kf-xs)), mean(t_kf), std(t_kf));
fprintf('SISR-PF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_sisr-xs)), std(trms(xhat_sisr-xs)), mean(t_sisr), std(t_sisr));
fprintf('B-PF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_bpf-xs)), std(trms(xhat_bpf-xs)), mean(t_bpf), std(t_bpf));
fprintf('RTSS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_rts-xs)), std(trms(xhat_rts-xs)), mean(t_rts), std(t_rts));
fprintf('KSD-PS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_ksd-xs)), std(trms(xhat_ksd-xs)), mean(t_ksd), std(t_ksd));
fprintf('FFBSi\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_ffbsi-xs)), std(trms(xhat_ffbsi-xs)), mean(t_ffbsi), std(t_ffbsi));
fprintf('CPF-AS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_cpfas-xs)), std(trms(xhat_cpfas-xs)), mean(t_cpfas), std(t_cpfas));
