% Test of the basic pf algorithms
%
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath ../src;

%% Parameters
N = 100;        % No. of time samples
J = 200;        % No. of particles
L = 1;          % No. of Monte Carlo runs

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
model = model_lgssm(F, Q, G, R, m0, P0);

%% Proposal
Sp = G*Q*G' + R;
Lp = Q*G'/Sp;
mu = @(x, y) F*x + Lp*(y - G*F*x);
Sigma = Q - Lp*Sp*Lp';
q.fast = 1;
q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';

%% 
xs = zeros(size(m0, 1), N, L);
xhat_kf = xs;
xhat_rts = xs;
xhat_bpf1 = xs;
xhat_bpf = xs;
xhat_sisr = xs;
xhat_cpfas = xs;
xhat_ksd = xs;
xhat_ffbsi = xs;
y = zeros(1, N, L);
t_kf = zeros(1, L);
t_rts = t_kf;
t_bpf = t_kf;
t_bpf1 = t_kf;
t_sisr = t_kf;
t_ksd = t_kf;
t_ffbsi = t_kf;
t_cpfas = t_kf;

% t = 1:N;

for l = 1:L
    %% Simulate System
    x = m0 + chol(P0).'*randn(2, 1);
    for n = 1:N
        qn = chol(Q).'*randn(size(Q, 1), 1);
        x = F*x + qn;
        rn = chol(R).'*randn(size(R, 1), 1);
        y(:, n, l) = G*x + rn;
        xs(:, n, l) = x;
    end

    %% Estimate
if 0
    % KF
    tic;
    xhat_kf(:, :, l) = kf(y(:, :, l), F, G, Q, R, m0, P0);
    t_kf(l) = toc;
    
    % RTS
    tic;
    [m, P, mp, Pp] = kf(y(:, :, l), F, G, Q, R, m0, P0);
    xhat_rts(:, :, l) = rtss(F, m, P, mp, Pp);
    t_rts(l) = toc;
end

    % Bootstrap PF (indirect)
    tic;
    [xhat_bpf1(:, :, l), sys_bpf1] = pf(model, y(:, :, l), [], J);
    t_bpf1(l) = toc;

if 0    
    % Optimal proposal PF
    tic;
    [xhat_sisr(:, :, l)] = sisr_pf(y(:, :, l), 1:N, model, q, J);
    t_sisr(l) = toc;
    
    % Bootstrap PF
    tic;
    [xhat_bpf(:, :, l)] = bootstrap_pf(y(:, :, l), 1:N, model, J);
    t_bpf(l) = toc;
        
    % Kronander-Sch?n-Dahlin smoother
    tic;
    [xhat_ksd(:, :, l)] = ksd_ps(y(:, :, l), 1:N, model, 2*J, J);
    t_ksd(l) = toc;
    
    % FFBSi smoother
    tic;
    xhat_ffbsi(:, :, l) = ffbsi_ps(y(:, :, l), 1:N, model, 2*J, J);
    t_ffbsi(l) = toc;
    
    % CPF-AS MCMC smoother
    par.Nmixing = 1;
    tic;
    xhat_cpfas(:,:, l) = cpfas_ps(y(:, :, l), t, model, [], 2*J, 10, par);
    t_cpfas(l) = toc;
end
end

%% Stats

fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', L, J);
fprintf('\tRMSE\t\tTime\n');
fprintf('\t----\t\t----\n');

[mean_rmse_bpf1, var_rmse_bpf1] = trmse(xhat_bpf1 - xs);
fprintf('BPF (1)\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_bpf1, sqrt(var_rmse_bpf1), mean(t_bpf1), std(t_bpf1) ...
);
if 0
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
end
