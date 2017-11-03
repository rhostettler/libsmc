% Example showing how to use PSIS for resampling and comparing it to the
% ESS
%
% 2017-03-23 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath(genpath('../src'));

%% Parameters
N = 100;    % No. of time samples
K = 100;    % No. of Monte Carlo runs
M = 1000;   % No. of particles
Mt = M/3;   % ESS resampling threshold
kt = 0.5;   % PSIS resampling threshold

%% Model
% A linear Gaussian model
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

%% Filter Parameters
% ESS resampling Parameters
par_rs_ess = struct();
par_rs_ess.Mt = Mt;
par_ess = struct();
par_ess.resample = @(lw) resample_ess(lw, par_rs_ess);

% PSIS resampling Parameters
par_rs_psis = struct();
par_rs_psis.kt = kt;
par_psis = struct();
par_psis.resample = @(lw) resample_psis(lw, par_rs_psis);

% Importance density
if 0
    % Optimal proposal
    S = G*Q*G' + R;
    L = Q*G'/S;
    mu = @(x, y) F*x + L*(y - G*F*x);
    Sigma = Q - L*S*L';
    q = struct();
    q.fast = true;
    q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
    q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
else
    % Bootstrap proposal
    q = struct();
    q.fast = model.px.fast;
    q.rand = @(y, x, t) model.px.rand(x, t);
    q.logpdf = @(xp, y, x, t) model.px.logpdf(xp, x, t);
    
    par_ess.calculate_incremental_weights = @calculate_incremental_weights_bootstrap;
    par_psis.calculate_incremental_weights = @calculate_incremental_weights_bootstrap;
end

%% Simulate
xx = zeros(size(m0, 1), N, K);
xhat_ess = xx;
xhat_psis = xx;
y = zeros(1, N, K);
t_ess = zeros(1, K);
t_psis = t_ess;
r_ess = zeros(1, N, K);
r_psis = r_ess;

for k = 1:K
    %% Simulate Data
    x = m0+chol(P0).'*randn(2, 1);
    for n = 1:N
        qn = chol(Q).'*randn(size(Q, 1), 1);
        x = F*x + qn;
        rn = chol(R).'*randn(size(R, 1), 1);
        y(:, n, k) = G*x + rn;
        xx(:, n, k) = x;
    end

    %% Estimate    
    % Filter w/ ESS
    tic;
    [xhat_ess(:, :, k), sys_ess] = sisr_pf(y(:, :, k), 1:N, model, q, M, par_ess);
    r_ess(:, :, k) = cat(2, sys_ess(2:N+1).r);
    t_ess(k) = toc;
    
    % Fitler w/ PSIS
    tic;
    [xhat_psis(:, :, k), sys_psis] = sisr_pf(y(:, :, k), 1:N, model, q, M, par_psis);
    r_psis(:, :, k) = cat(2, sys_psis(2:N+1).r);
    t_psis(k) = toc;
end

%% Stats
trms = @(e) mean(sqrt(sum(e.^2, 1)), 2);
fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', K, M);
fprintf('\t\tESS\t\t\tPSIS\n');
fprintf('\t\t---\t\t\t----\n');
fprintf('RMSE\t\t%.4f (%.2f)\t\t%.4f (%.2f)\n', ...
    mean(trms(xhat_ess-xx)), std(trms(xhat_ess-xx)), mean(trms(xhat_psis-xx)), std(trms(xhat_psis-xx)));
fprintf('Resampling\t%.2f (%.2f)\t\t%.2f (%.2f)\n', ...
    mean(squeeze(sum(r_ess, 2))), std(squeeze(sum(r_ess, 2))), mean(squeeze(sum(r_psis, 2))), std(squeeze(sum(r_psis, 2))));
fprintf('Time\t\t%.2e (%.2e)\t%.2e (%.2e)\n', ...
    mean(t_ess), std(t_ess), mean(t_psis), std(t_psis));
