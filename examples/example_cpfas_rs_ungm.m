% UNGM example of rejection-sampling-based CPF-AS
%
% 2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath ../src;
addpath ../../gp-pmcmc/lib      % TODO: This is used for the custom gp_plot only; consider moving this here.
rng(511);

%% Parameters
N = 100;            % No. of time samples (100)
J = 100;            % No. of particles (100)
Kburnin = 50;
K = 100;
Kmcmc = Kburnin+K;  % No. of MCMC samples
L = J;              % No. of rejection sampling trials (J)
P = 100;            % No. of MC simulations (100)

%% Model
dx = 1;
dy = 1;
Q = 10;
R = 1;
m0 = 0;
P0 = 5;
lambda = 1:N;
f = @(x, lambda) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*lambda);
g = @(x, lambda) 0.05*x.^2;
mod = model_nonlinear_gaussian(f, Q, g, R, m0, P0);
model = @(theta) mod;           % TODO: This is the gibbs_pmcmc() curiosity that needs sorting out

%% Algorithm parameters
% Parameters for categorical sampling
par = struct();
par.sample_states = @(y, xtilde, theta, lambda) cpfas(model(theta), y, xtilde, lambda, J);

% Parameters for rejection sampling
par_cpf = struct();
par_cpf.sample_ancestor_index = @(model, y, xt, x, lw, theta) sample_ancestor_index_rs(model, y, xt, x, lw, theta, L);
par_rs = struct();
par_rs.sample_states = @(y, xtilde, theta, lambda) cpfas(model(theta), y, xtilde, lambda, J, par_cpf);

%% Simulate
% Preallocate
xs = zeros(dx, N, P);
y = zeros(dy, N, P);

t_cs = zeros(1, P);
t_rs = zeros(1, P);
xhat_cs = zeros(dx, N, P);
xhat_rs = zeros(dx, N, P);
l_rs = zeros(P, L);
a_rs = zeros(P, 1);

dgamma_grid = 0:1e-3:1;
Ncdf = length(dgamma_grid);
dgamma_rs = zeros(P, Ncdf);

fh = pbar(P);
for p = 1:P
    %% Simulate system
    [xs(:, :, p), y(:, :, p)] = simulate_model(mod, lambda, N);
    if 0
    x = mod.px0.rand(1);
    for n = 1:N
        x = mod.px.rand(x, n);
        y(:, n, p) = mod.py.rand(x, n);
        xs(:, n, p) = x;
    end
    end

    %% Estimate
    % Standard
    tic;
    [x_cs, ~, sys] = gibbs_pmcmc(model, y(:, :, p), [], lambda, Kmcmc, par);
    t_cs(p) = toc;
    x_cs = x_cs(:, 2:end, :);
    xhat_cs(:, :, p) = mean(x_cs(:, :, Kburnin+1:end), 3);

    % Rejection sampling
    tic;
    [x_rs, ~, sys_rs] = gibbs_pmcmc(model, y(:, :, p), [], lambda, Kmcmc, par_rs);
    t_rs(p) = toc;
    x_rs = x_rs(:, 2:end, :);
    xhat_rs(:, :, p) = mean(x_rs(:, :, Kburnin+1:end), 3);
    
    %% Calculate statistics
    % Get acceptance rate and l statistics for rejection sampling
    l_tmp = zeros(Kmcmc, N+1);
    a_tmp = zeros(Kmcmc, N+1);
    dgamma_tmp = zeros(Kmcmc, J, N+1);
    for k = 1:Kmcmc
        tmp = sys_rs{k};
        for n = 2:N+1
            state = tmp(n).state;
            l_tmp(k, n) = state.l;
            a_tmp(k, n) = state.accepted;
            dgamma_tmp(k, :, n) = state.dgamma;
        end
    end
    
    dgamma_rs(p, :) = hist(dgamma_tmp(:), dgamma_grid);
    l_rs(p, :) = hist(l_tmp(l_tmp > 0), 1:L);
    a_rs(p) = sum(sum(a_tmp));
    
    %% Progress
    pbar(p, fh);
end
pbar(0, fh);

%% Stats
trms = @(e) sqrt(mean(sum(e.^2, 1), 2));
fprintf('\nResults for P = %d MC simulations, K = %d MCMC samples (burn-in: %d), J = %d particles.\n\n', P, K, Kburnin, J);
fprintf('\t\tRMSE\t\tTime\n');
fprintf('\t\t----\t\t----\n');
fprintf('Standard\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
    mean(trms(xhat_cs-xs)), std(trms(xhat_cs-xs)), mean(t_cs), std(t_cs));
fprintf('Rejection\t%.4f (%.2f)\t%.3g (%.2f)\n', ...
    mean(trms(xhat_rs-xs)), std(trms(xhat_rs-xs)), mean(t_rs), std(t_rs));

fprintf('\nRate of indices accepted by RS: %.2f (%.2f)\n', mean(a_rs/(Kmcmc*N)), std(a_rs/(Kmcmc*N)));

%% Illustrate
if 0
if P == 1
    figure(1); clf();
    figure(2); clf();
    for i = 1:dx
        figure(1);
        subplot(dx, 1, i);
        plot(squeeze(x_cs(i, :, :)), 'Color', [0.9, 0.9, 0.9]); hold on;
        plot(xhat_cs(i, :), 'r', 'LineWidth', 2);
        plot(xs(i, :), 'k', 'LineWidth', 2);
        title('Posterior Mean (red), True Trajectory (black), Sampled Trajectories (grey)');
        
        figure(2);
        subplot(dx, 1, i);
        plot(squeeze(x_rs(i, :, :)), 'Color', [0.9, 0.9, 0.9]); hold on;
        plot(xhat_rs(i, :), 'r', 'LineWidth', 2);
        plot(xs(i, :), 'k', 'LineWidth', 2);
        title('Posterior Mean (red), True Trajectory (black), Sampled Trajectories (grey)');
    end
end
end

% Distribution over ls
rates = a_rs/(Kmcmc*N);
lhistn_rs = l_rs./(sum(l_rs, 2)*ones(1, L)).*(rates*ones(1, L))*100;
ml_rs = mean(lhistn_rs);
stdl_rs = std(lhistn_rs);

figure(3); clf();
gp_plot(1:L, ml_rs, stdl_rs.^2, 2);
xlim([1, L]);
title('Number of accepted indices');

% Distribution of the acceptance probability error
pdgamma = dgamma_rs./(sum(dgamma_rs, 2)*ones(1, Ncdf));
mdgamma_rs = mean(pdgamma);
stddgamma_rs = std(pdgamma);
figure(4); clf();
% semilogx(dgamma_grid, pdgamma);
gp_plot(dgamma_grid, mdgamma_rs, stddgamma_rs.^2, 2);
set(gca, 'XScale', 'log');
title('Distribution of acceptance probability error');

