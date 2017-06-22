% PSIS for Resampling Test
%
% test_sisr_pf.m -- 2017-03-23
% Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath lib lib/psis;

%% Parameters
% No. of time samples
N = 100;

% No. of Monte Carlo runs
K = 100;

% No. of particles
M = 1000;

% ESS resampling threshold
Mt = M/2;

export = 0;

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

%% Model for Filter
% Model
model.px.fast = 1;
model.px.logpdf = @(xp, x, t) logmvnpdf(xp.', (F*x).', Q).';
model.py.fast = 1;
model.py.logpdf = @(y, x, t) logmvnpdf(y.', (G*x).', R).';
model.px0.rand = @(M) m0*ones(1, M) + chol(P0).'*randn(size(P0, 1), M);

% Bootstrap proposal
if 0
q.fast = 1;
q.rand = @(y, x, t) F*x + chol(Q).'*randn(size(Q, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (F*x).', Q).';
end

% Optimal proposal
if 1
S = G*Q*G' + R;
L = Q*G'/S;
mu = @(x, y) F*x + L*(y - G*F*x);
Sigma = Q - L*S*L';
q.fast = 1;
q.rand = @(y, x, t) mu(x, y) + chol(Sigma).'*randn(size(Sigma, 1), size(x, 2));
q.logpdf = @(xp, y, x, t) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
end

%
par_ess.Mt = Mt;
par_psis.resample = @(x, lw) psisresample(x, lw, 0.5);

%% 
xs = zeros(size(m0, 1), N, K);
xhat_kf = xs;
xhat_ess = xs;
xhat_psis = xs;
y = zeros(1, N, K);
t_kf = zeros(1, K);
t_ess = zeros(1, K);
t_psis = t_ess;
r_ess = zeros(1, N, K);
r_psis = r_ess;

for k = 1:K
    %% Simulate System
    x = m0+chol(P0).'*randn(2, 1);
    for n = 1:N
        qn = chol(Q).'*randn(size(Q, 1), 1);
        x = F*x + qn;
        rn = chol(R).'*randn(size(R, 1), 1);
        y(:, n, k) = G*x + rn;
        xs(:, n, k) = x;
    end

    %% Estimate
    tic;
    xhat_kf(:, :, k) = kf(y(:, :, k), F, G, Q, R, m0, P0);
    t_kf(k) = toc;
    
    % Traditional (ESS)
    tic;
    [xhat_ess(:, :, k), ~, debug] = sisr_pf(y(:, :, k), 1:N, model, q, M, par_ess);
    r_ess(:, :, k) = debug.r;
    t_ess(k) = toc;
    
    % New (PSIS)
    tic;
    [xhat_psis(:, :, k), ~, debug] = sisr_pf(y(:, :, k), 1:N, model, q, M, par_psis);
    r_psis(:, :, k) = debug.r;
    t_psis(k) = toc;  
end

%% Calculate Errors

%% Stats
trms = @(e) mean(sqrt(sum(e.^2, 1)), 2);
fprintf('\nResults for K = %d MC simulations, M = %d particles.\n\n', K, M);
fprintf('\t\tKF\t\t\tESS\t\t\tPSIS\n');
fprintf('\t\t--\t\t\t---\t\t\t----\n');
fprintf('RMSE\t\t%.4f (%.2f)\t\t%.4f (%.2f)\t\t%.4f (%.2f)\n', ...
    mean(trms(xhat_kf-xs)), std(trms(xhat_kf-xs)), mean(trms(xhat_ess-xs)), std(trms(xhat_ess-xs)), mean(trms(xhat_psis-xs)), std(trms(xhat_psis-xs)));
fprintf('Resampling\tN/A\t\t\t%.2f (%.2f)\t\t%.2f (%.2f)\n', ...
    mean(squeeze(sum(r_ess, 2))), std(squeeze(sum(r_ess, 2))), mean(squeeze(sum(r_psis, 2))), std(squeeze(sum(r_psis, 2))));
fprintf('Time\t\t%.2e (%.2e)\t%.2e (%.2e)\t%.2e (%.2e)\n', ...
    mean(t_kf), std(t_kf), mean(t_ess), std(t_ess), mean(t_psis), std(t_psis));

%% 
if export
% TODO: have a file with results for each method instead
if exist('results.mat', 'file')
    load('results.mat')
    iM = find(results(:, 1) == M);
else
    results = [];
    iM = [];
    header = { ...
        'M', ...
        'rms_ess', 'std_rms_ess', ...
        'rms_psis', 'std_rms_psis', ...
        'samp_ess', 'std_samp_ess', ...
        'samp_psis', 'std_samp_psis', ...
        't_ess', 'std_t_ess', ...
        't_psis', 'std_t_psis' ...
    };
end

res = [
    M,...
    mean(trms(xhat_ess-xs)), std(trms(xhat_ess-xs)), ...
    mean(trms(xhat_psis-xs)), std(trms(xhat_psis-xs)), ...
    mean(squeeze(sum(r_ess, 2))), std(squeeze(sum(r_ess, 2))), ...
    mean(squeeze(sum(r_psis, 2))), std(squeeze(sum(r_psis, 2))), ...
    mean(t_ess), std(t_ess), mean(t_psis), std(t_psis) ...
];
if ~isempty(iM)
    results(iM, :) = res;
else
    results = [
        results;
        res
    ];
end
save('results.mat', 'results', 'header');
end