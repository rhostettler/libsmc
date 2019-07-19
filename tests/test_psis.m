% Test for PSIS-based resampling
%
% 2017-present -- Roland Hostettler

%{
% This file is part of the libsmc Matlab toolbox.
%
% libsmc is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libsmc is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libsmc. If not, see <http://www.gnu.org/licenses/>.
%}

% Housekeeping
clear variables;
addpath ../src ../external/psis;
rng(29742);

%% Parameters
N = 1000;    % No. of time samples
K = 100;    % No. of Monte Carlo simulations
J = 1000;   % No. of particles
Jt = J/10;   % ESS resampling threshold
kt = 0.5;   % PSIS resampling threshold

theta = [];
oid = false;

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
% Importance density (default: bootstrap)
if oid
    % Optimal proposal
    S = G*Q*G' + R;
    L = Q*G'/S;
    mu = @(x, y) F*x + L*(y - G*F*x);
    Sigma = Q - L*S*L';
    LSigma = chol(Sigma).';
    q = struct();
    q.fast = true;
    q.rand = @(y, x, theta) mu(x, y) + LSigma*randn(size(Sigma, 1), size(x, 2));
    q.logpdf = @(xp, y, x, theta) logmvnpdf(xp.', (mu(x, y)).', Sigma).';
    sample_oid = @(model, y, x, theta) sample_generic(model, y, x, theta, q);
end

% ESS resampling Parameters
par_rs_ess = struct();
par_rs_ess.Jt = Jt;
par_ess = struct();
par_ess.resample = @(lw) resample_ess(lw, par_rs_ess);
if oid
    par_ess.sample = sample_oid;
    par_ess.calculate_incremental_weights = @calculate_incremental_weights_generic;
end

% PSIS resampling Parameters
par_rs_psis = struct();
par_rs_psis.kt = kt;
par_psis = struct();
par_psis.resample = @(lw) resample_psis(lw, par_rs_psis);
if oid
    par_psis.sample = sample_oid;
    par_psis.calculate_incremental_weights = @calculate_incremental_weights_generic;
end

%% Simulate
dy = 1;
dx = size(m0, 1);

ys = zeros(dy, N, K);

xs = zeros(dx, N, K);
xhat_ess = xs;
xhat_psis = xs;

t_ess = zeros(1, K);
t_psis = t_ess;

r_ess = zeros(1, N, K);
r_psis = r_ess;

fprintf('Simulating with N = %d time steps, J = %d particles, K = %d MC simulations...\n', N, J, K);
fh = pbar(K);
for k = 1:K
    %% Simulate Data
    [xs(:, :, k), ys(:, :, k)] = simulate_model(model, theta, N);

    %% Estimate    
    % Filter w/ ESS
    tic;
    [xhat_ess(:, :, k), sys_ess] = pf(model, ys(:, :, k), theta, J, par_ess);
    t_ess(k) = toc;
    tmp = cat(2, sys_ess(2:N+1).rstate);
    r_ess(:, :, k) = cat(2, tmp.r);
    
    % Fitler w/ PSIS
    tic;
    [xhat_psis(:, :, k), sys_psis] = pf(model, ys(:, :, k), theta, J, par_psis);
    t_psis(k) = toc;
    tmp = cat(2, sys_psis(2:N+1).rstate);
    r_psis(:, :, k) = cat(2, tmp.r);
    
    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Stats
e_rmse_ess = trmse(xhat_ess - xs);
e_rmse_psis = trmse(xhat_psis - xs);

fprintf('\t\tESS\t\t\tPSIS\n');
fprintf('\t\t---\t\t\t----\n');
fprintf('RMSE\t\t%.4f (%.2f)\t\t%.4f (%.2f)\n', ...
    mean(e_rmse_ess), std(e_rmse_ess), mean(e_rmse_psis), std(e_rmse_psis) ...
);
fprintf('Resampling / %%\t%.2f (%.2f)\t\t%.2f (%.2f)\n', ...
    mean(squeeze(sum(r_ess, 2)/N))*100, std(squeeze(sum(r_ess, 2)/N))*100, ...
    mean(squeeze(sum(r_psis, 2)/N))*100, std(squeeze(sum(r_psis, 2)/N))*100 ...
);
fprintf('Time / s\t%.2e (%.2e)\t%.2e (%.2e)\n', ...
    mean(t_ess), std(t_ess), mean(t_psis), std(t_psis) ...
);
