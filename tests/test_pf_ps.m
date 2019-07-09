% Test of the basic PF/PS algorithms
%
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath ../src;
rng(2872);

%% Parameters
N = 100;        % No. of time samples
J = 200;        % No. of particles
L = 20;         % No. of Monte Carlo runs

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
model_pmcmc = @(theta) model;   % TODO: This should not be necessary in the end, but gibbs_pmcmc needs to be updated first.

%% Optimal proposal
Sn = G*Q*G' + R;
Ln = Q*G'/Sn;
mu = @(x, y) F*x + Ln*(y - G*F*x);
Sigma = Q - Ln*Sn*Ln';
LSigma = chol(Sigma).';
q.fast = true;
q.rand = @(y, x, theta) mu(x, y) + LSigma*randn(size(x));
q.logpdf = @(xp, y, x, theta) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
par_opt = struct( ...
    'sample', @(model, y, x, theta) sample_generic(model, y, x, theta, q), ...
    'calculate_incremental_weights', @(model, y, xp, x, theta) calculate_incremental_weights_generic(model, y, xp, x, theta, q) ...
);

par_ksd = struct('smooth', @smooth_ksd);

%% Preallocate
xs = zeros(size(m0, 1), N, L);
y = zeros(1, N, L);

m_kf = xs;
xhat_bpf = xs;
xhat_opt = xs;

m_rts = xs;
xhat_ksd = xs;
xhat_ffbsi = xs;
xhat_cpfas = xs;

t_kf = zeros(1, L);
t_bpf = t_kf;
t_opt = t_kf;

t_rts = t_kf;
t_ksd = t_kf;
t_ffbsi = t_kf;
t_cpfas = t_kf;

%% MC simulations
fh = pbar(L);
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
    xhat_bpf(:, :, l) = pf(model, y(:, :, l), [], J);
    t_bpf(l) = toc;
    
    % Optimal proposal PF
    tic;
    xhat_opt(:, :, l) = pf(model, y(:, :, l), [], J, par_opt);
    t_opt(l) = toc;
        
    %% Smoothers
    % RTS smoother (requires EKF/UKF toolbox)
    tic;
    m_rts(:, :, l) = rts_smooth(m_kf(:, :, l), P_kf, F, Q);
    t_rts(l) = toc;

    % Kronander-Schon-Dahlin smoother
    tic;
    [xhat_ksd(:, :, l)] = ps(model, y(:, :, l), [], 2*J, J, par_ksd);
    t_ksd(l) = toc;
    
    % FFBSi smoother
    tic;
    xhat_ffbsi(:, :, l) = ps(model, y(:, :, l), [], 2*J, J);
    t_ffbsi(l) = toc;
    
    % CPF-AS MCMC smoother
    tic;
    [x_cpfas, sys] = gibbs_pmcmc(model_pmcmc, y(:, :, l));
    xhat_cpfas(:, :, l) = mean(x_cpfas(:, 2:end, :), 3);
    t_cpfas(l) = toc;
    
    %% Progress
    pbar(l, fh);
end
pbar(0, fh);

%% Calculate stats
% Filters
[mean_rmse_none, var_rmse_none] = trmse(xs);
[mean_rmse_kf, var_rmse_kf] = trmse(m_kf - xs);
[mean_rmse_bpf, var_rmse_bpf] = trmse(xhat_bpf - xs);
[mean_rmse_opt, var_rmse_opt] = trmse(xhat_opt - xs);

% Smoothers
[mean_rmse_rts, var_rmse_rts] = trmse(m_rts - xs);
[mean_rmse_cpfas, var_rmse_cpfas] = trmse(xhat_cpfas - xs);
[mean_rmse_ksd, var_rmse_ksd] = trmse(xhat_ksd - xs);
[mean_rmse_ffbsi, var_rmse_ffbsi] = trmse(xhat_ffbsi - xs);

%% Print stats
% Header
fprintf('\nResults for L = %d MC simulations, J = %d particles.\n\n', L, J);
fprintf('\tRMSE\t\t\tTime\n');
fprintf('\t----\t\t\t----\n');

% Filters
fprintf('None\t%.4f (%.2f)\t%.2e (%.2e)\n', ...
    mean_rmse_none, sqrt(var_rmse_none), 0, 0 ...
);
fprintf('KF\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_kf, sqrt(var_rmse_kf), mean(t_kf), std(t_kf) ...
);
fprintf('BPF\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_bpf, sqrt(var_rmse_bpf), mean(t_bpf), std(t_bpf) ...
);
fprintf('OPT PF\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_opt, sqrt(var_rmse_opt), mean(t_opt), std(t_opt) ...
);

% Smoothers
fprintf('RTSS\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_rts, sqrt(var_rmse_rts), mean(t_rts), std(t_rts) ...
);
fprintf('CPF-AS\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_cpfas, sqrt(var_rmse_cpfas), mean(t_cpfas), std(t_cpfas) ...
);
fprintf('KSD-PS\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_ksd, sqrt(var_rmse_ksd), mean(t_ksd), std(t_ksd) ...
);
fprintf('FFBSi\t%.4f (%.2f)\t\t%.2e (%.2e)\n', ...
    mean_rmse_ffbsi, sqrt(var_rmse_ffbsi), mean(t_ffbsi), std(t_ffbsi) ...
);
