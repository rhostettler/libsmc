% Stochastic volatility example of rejection-sampling-based CPF-AS/PGAS
%
% Depends on:
%   * libgp (https://github.com/rhostettler/libgp)
%   * gp-pmcmc (to be released)
% 
% 2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% Housekeeping
clear variables;
addpath(genpath('../src'));
addpath ../../libgp/src ../../gp-pmcmc/lib

%% Parameters
% MCMC parameters
K = 100;                % No. of MCMC samples (both MCMC and Particle MCMC) (250)
Kburnin = 50;            % No. of burn-in samples
Kmcmc = K+Kburnin;      % Total no. of MCMC samples
J = 250;                % No. of particles in particle filter
L = 20;                 % Maximum number of rejection sampling proposals

% Model parameters
nu1 = 0.5;              % Order of 'slow' cov. (0.5 for Ornstein-Uhlenbeck)
ell1 = 400;             % Initial length scale of 'slow' cov.
sigma21 = 1;            % Initial magnitude of 'slow' cov.

nu2 = 0.5;              % Order of 'fast' cov. (0.5 for Ornstein-Uhlenbeck)
ell2 = 1;               % Initial length scale of 'fast' cov.
sigma22 = 1;            % Initial magnitude of 'fast' cov.

Ts = 1;                 % Sampling time

%% Load IBM data
% The data has six columns, these are:
%
% [date, return, rv, rv(65), rv(15), rv(5)]
%
% where rv(XX) is the realized (measured) volatility using XX-min sampling.
% The NYSE trading hours are from 9:30 am to 4:00 pm, that is, 6.5 h.
% Hence, 65 min sampling corresponds 6 samples, 15 min sampling to 26 
% samples, and 5 min sampling to 78 samples.
data = load('../../gp-pmcmc/data/ibm_data.txt');

% Second column: daily returns
y = data(:, 2).';

% 5th column: realized (measured) volatility, 5 min sampling
rv = data(:, 6).';
N = length(y);
t = 1:N;

%% Model
% Priors for the hyperparameters
pell = struct();
pell.parameters = [1, 1];
pell.logpdf = @(logell) loginvgampdf(exp(logell), 1, 1);
psigma2 = struct();
psigma2.parameters = [1, 1];
psigma2.logpdf = @(sigma2) loginvgampdf(sigma2, 1, 1);
ptheta = {pell; pell; psigma2; psigma2};
pprior = @(theta) (...
   pell.logpdf(theta(1)) + pell.logpdf(theta(2)) ...
   + psigma2.logpdf(exp(theta(3))) + psigma2.logpdf(exp(theta(4))) ...
);

% Likelihood
py = struct();
py.fast = 1;
py.logpdf = @(y, f, theta)  -1/2*log(2*pi) - f/2 - 1/2*y.^2.*exp(-f);   % lognormpdf(y, 0, a + exp(s));

% Model constructor (TODO: pmcmc_gibbs oddity)
model = @(theta) gpsmc_model(@() model_sv(exp(theta(1)), exp(theta(2)), theta(3), theta(4), nu1, nu2), [], [], [], py, ptheta, Ts);

%% Parameters
% Initial parameter guesses
theta0 = [log(ell1), log(ell2), sigma21, sigma22].';
   
% Parameter samplers:
% * Direct Gibbs sampling of the variances
% * Metropolis-within-Gibbs sampling of the lengthscales (Gaussian random
%   walk)
par_ls = struct();
par_ls.C = 1e-3*eye(2);
samplers = {
    @sample_variance, 3;
    @sample_variance, 4;
%        @sample_lengthscale, [1, 2];
    @(y, t, s, theta, model, iell, ~) sample_lengthscale(y, t, s, theta, model, iell, [], par_ls), [1, 2];
};

%% Estimation: Standard
fprintf('Running standard PGAS...\n');

% Parameters
fh = pbar(Kmcmc);
par_cs = struct();
par_cs.sample_parameters = @(y, t, x, theta, model, state) sample_parameters(y, t, x, theta, model, samplers, state);
par_cs.show_progress = @(p, ~, ~) pbar(round(p*Kmcmc), fh);

% Estimation
tstart = tic;
[xs_cs, thetas_cs, sys_cs] = gibbs_pmcmc(model, y, theta0, t, Kmcmc, par_cs);
t_cs = toc(tstart);
pbar(0, fh);

%% Estimation: Rejection sampling
fprintf('Running PGAS w/ rejection sampling...\n');

% Parameters
fh = pbar(Kmcmc);
par_cpf = struct();
par_cpf.sample_ancestor_index = @(model, y, xt, x, lw, theta) sample_ancestor_index_rs(model, y, xt, x, lw, theta, L);
par_rs.sample_states = @(y, xtilde, theta, lambda) cpfas(model(theta), y, xtilde, lambda, J, par_cpf);
par_rs.sample_parameters = @(y, t, x, theta, model, state) sample_parameters(y, t, x, theta, model, samplers, state);
par_rs.show_progress = @(p, ~, ~) pbar(round(p*Kmcmc), fh);

% Estimation
tstart = tic;
[xs_rs, thetas_rs, sys_rs] = gibbs_pmcmc(model, y, theta0, t, Kmcmc, par_rs);
t_rs = toc(tstart);
pbar(0, fh);

%% Post-processing
% Estimate posterior E(f | y) and var(f | y)
mod = model(theta0);
C = mod.C;
f_cs = smc_expectation(@(x) C*x, xs_cs(:, :, Kburnin+1:end));
sigma2_cs = smc_expectation(@(x) (C*x - f_cs).^2, xs_cs(:, :, Kburnin+1:end));

f_rs = smc_expectation(@(x) C*x, xs_rs(:, :, Kburnin+1:end));
sigma2_rs = smc_expectation(@(x) (C*x - f_rs).^2, xs_rs(:, :, Kburnin+1:end));

% Find rejection sampling statistics
a_rs = zeros(Kmcmc, N+1);   % Stores the 'accepted' flag
l_rs = zeros(Kmcmc, N+1);   % Stores the acceptance iteration 'l'
dgamma_rs = zeros(Kmcmc, J, N+1);
for k = 1:Kmcmc
    tmp = sys_rs{k};
    for n = 2:N+1
        state = tmp(n).state;
        a_rs(k, n) = state.accepted;
        l_rs(k, n) = state.l;
        dgamma_rs(k, :, n) = state.dgamma;
    end
end

%% Performance
fprintf('Standard PGAS RMSE: %.2f\n', rms(f_cs(2:end)-log(rv)));
fprintf('PGAS w/ rejection sampling RMSE: %.2f\n', rms(f_rs(2:end)-log(rv)));
fprintf('Ancestor indices sampled using rejection sampling: %.1f %%\n', sum(sum(a_rs))/(Kmcmc*N)*100)

%% Plots
figure(1); clf();
plot(log(rv)); hold on;
plot(f_cs(2:end));
plot(f_rs(2:end));
title('Realized and estimated log-volatility');
legend('Realized', 'Standard', 'Rejection sampling');
% plot(squeeze(sum(xs_cs(C == 1, :, Kburnin+1:end), 1)), 'Color', [0.9, 0.9, 0.9]);
% plot(f_cs, 'r', 'LineWidth', 2);
% plot(log(rv));

figure(2); clf();
plot(t, y); hold on;
plot(t, exp(f_cs(2:end)));
plot(t, 2*exp(f_cs(2:end)));
xlim([1, N]); grid on;
xlabel('t'); ylabel('y_n');
title('Return');
legend('Measured return', '1-sigma of volatility', '2-sigma of volatility');

if 0
% Estimated volatility vs. realized volatility
figure(3); clf();
gp_plot(t, f_cs, sigma2_cs); hold on;

% plot(f_pmcmc, 'b'); hold on;
% plot(f_laplace, 'r');
plot(log(data(:, end-1)), '.');
% legend('PMCMC', 'Laplace', 'Realized');
plot(f_cs + 2*sqrt(sigma2_cs), '--b');
plot(f_cs - 2*sqrt(sigma2_cs), '--b');
% plot(f_laplace+2*sqrt(diag(Sigma_laplace)).', '--r');
% plot(f_laplace-2*sqrt(diag(Sigma_laplace)).', '--r');
xlim([1, N]); grid on;
xlabel('t'); ylabel('f_n');
title('Log-Volatility');
end

% Trace plots
figure(4); clf();
Ntheta = size(thetas_cs, 1);
for i = 1:Ntheta
    subplot(Ntheta, 1, i);
    plot(thetas_cs(i, :));
    title(sprintf('PMCMC Trace Plot, \\theta_{%d}', i));
end

% Histogram of 
rate = sum(sum(a_rs))/(Kmcmc*N);
xhist = 1:L;
lhist = hist(l_rs(l_rs > 0), xhist);
lhist = lhist/sum(lhist)*rate*100;

figure(5); clf();
bar(xhist, lhist);
xlim([0, L]);
title(sprintf('Ancestor indices RS sampled: %d/%d', sum(sum(sum(a_rs))), Kmcmc*N))

% Distribution of the acceptance probability error
dgamma_grid = 0:1e-3:1;
dgamma_hist = hist(dgamma_rs(:), dgamma_grid);
dgamma_hist = dgamma_hist/sum(dgamma_hist);
figure(6); clf();
plot(dgamma_grid, dgamma_hist);
set(gca, 'xscale', 'log');
