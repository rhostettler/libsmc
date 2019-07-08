% Test of the basic PF/PS algorithms
%
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath ../src;

%% Parameters
N = 100;        % No. of time samples
J = 200;        % No. of particles
L = 100;          % No. of Monte Carlo runs

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
model_pmcmc = @(theta) model;               % TODO: This should not be necessary in the end, but gibbs_pmcmc needs to be updated first.

%% Proposal
if 0
Sp = G*Q*G' + R;
Lp = Q*G'/Sp;
mu = @(x, y) F*x + Lp*(y - G*F*x);
Sigma = Q - Lp*Sp*Lp';
q.fast = 1;
q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
end

%% Preallocate
xs = zeros(size(m0, 1), N, L);
y = zeros(1, N, L);

m_kf = xs;
xhat_bpf = xs;
% xhat_sisr = xs;

m_rts = xs;
% xhat_ksd = xs;
% xhat_ffbsi = xs;
xhat_cpfas = xs;

t_kf = zeros(1, L);
t_bpf = t_kf;
% t_sisr = t_kf;

t_rts = t_kf;
% t_ksd = t_kf;
% t_ffbsi = t_kf;
t_cpfas = t_kf;

%% MC simulations
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

    %% Filters
    % KF (requires EKF/UKF toolbox)
    tic;
    [m_kf(:, :, l), P_kf] = kf_loop(m0, P0, G, R, y(:, :, l), F, Q);
    t_kf(l) = toc;
    
    % Bootstrap PF (indirect)
    tic;
    [xhat_bpf(:, :, l), sys_bpf1] = pf(model, y(:, :, l), [], J);
    t_bpf(l) = toc;

if 0    
    % Optimal proposal PF
    tic;
    [xhat_sisr(:, :, l)] = sisr_pf(y(:, :, l), 1:N, model, q, J);
    t_sisr(l) = toc;
end
        
    %% Smoothers
    % RTS smoother (requires EKF/UKF toolbox)
    tic;
    m_rts(:, :, l) = rts_smooth(m_kf(:, :, l), P_kf, F, Q);
    t_rts(l) = toc;

if 0
    % Kronander-Sch?n-Dahlin smoother
    tic;
    [xhat_ksd(:, :, l)] = ksd_ps(y(:, :, l), 1:N, model, 2*J, J);
    t_ksd(l) = toc;
    
    % FFBSi smoother
    tic;
    xhat_ffbsi(:, :, l) = ffbsi_ps(y(:, :, l), 1:N, model, 2*J, J);
    t_ffbsi(l) = toc;
end
    
    % CPF-AS MCMC smoother
    tic;
    [x_cpfas, sys] = gibbs_pmcmc(model_pmcmc, y(:, :, l));
    xhat_cpfas(:, :, l) = mean(x_cpfas(:, 2:end, :), 3);
    t_cpfas(l) = toc;
end

%% Calculate stats
% Filters
[mean_rmse_none, var_rmse_none] = trmse(xs);
[mean_rmse_kf, var_rmse_kf] = trmse(m_kf - xs);
[mean_rmse_bpf1, var_rmse_bpf1] = trmse(xhat_bpf - xs);

% Smoothers
[mean_rmse_rts, var_rmse_rts] = trmse(m_rts - xs);
[mean_rmse_cpfas, var_rmse_cpfas] = trmse(xhat_cpfas - xs);

%% Print stats
% Header
fprintf('\nResults for L = %d MC simulations, J = %d particles.\n\n', L, J);
fprintf('\tRMSE\t\tTime\n');
fprintf('\t----\t\t----\n');

% Filters
fprintf('None\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_none, sqrt(var_rmse_none), 0, 0 ...
);
fprintf('KF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_kf, sqrt(var_rmse_kf), mean(t_kf), std(t_kf) ...
);
fprintf('BPF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_bpf1, sqrt(var_rmse_bpf1), mean(t_bpf), std(t_bpf) ...
);
if 0
fprintf('SISR-PF\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_sisr-xs)), std(trms(xhat_sisr-xs)), mean(t_sisr), std(t_sisr) ...
);
end

% Smoothers
fprintf('RTSS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_rts, sqrt(var_rmse_rts), mean(t_rts), std(t_rts) ...
);
fprintf('CPF-AS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_cpfas, sqrt(var_rmse_cpfas), mean(t_cpfas), std(t_cpfas) ...
);

if 0
fprintf('KSD-PS\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_ksd-xs)), std(trms(xhat_ksd-xs)), mean(t_ksd), std(t_ksd) ...
);
fprintf('FFBSi\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean(trms(xhat_ffbsi-xs)), std(trms(xhat_ffbsi-xs)), mean(t_ffbsi), std(t_ffbsi) ...
);
end
