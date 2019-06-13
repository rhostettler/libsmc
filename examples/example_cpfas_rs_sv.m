% PMCMC for State-Space GPs: Stochastic Volatility Example
% 
% 2019 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath(genpath('../src'));
addpath ../../libgp/src ../../gp-pmcmc/lib

%% Parameters
% MCMC parameters
K = 100;                % No. of MCMC samples (both MCMC and Particle MCMC) (250)
Kburnin = 0;            % No. of burn-in samples
Kmcmc = K+Kburnin;      % Total no. of MCMC samples
J = 100;                % No. of particles in particle filter
L = 20;

% Model parameters
nu1 = 0.5;              % Order of 'slow' cov. (0.5 for Ornstein-Uhlenbeck)
ell1 = 150;             % Initial length scale of 'slow' cov.
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

% The first column is the date in the format YYYY.MMDD
if 0
t = data(:, 1);
yyyy = floor(tt);
mm = (floor(tt*100)/100-yyyy)*1e2;
dd = (floor(tt*1e4)/1e4-yyyy-mm/1e2)*1e4;
t = datenum(round(yyyy), round(mm), round(dd));
t = t-min(t)+1;
end

% Second column: daily returns
y = data(:, 2).';

% 5th column: realized (measured) volatility, 15 min sampling
rv = data(:, 5);
N = length(y);
t = 1:N;

%% Model
% Priors
pell = struct();
pell.parameters = [1, 1];
pell.logpdf = @(logell) loginvgampdf(exp(logell), 1, 1);
psigma2 = struct();
psigma2.parameters = [1, 1];
psigma2.logpdf = @(sigma2) loginvgampdf(sigma2, 1, 1);
if 1
    % sum-kernel
    ptheta = {pell; pell; psigma2; psigma2};
    pprior = @(theta) (...
       pell.logpdf(theta(1)) + pell.logpdf(theta(2)) ...
       + psigma2.logpdf(exp(theta(3))) + psigma2.logpdf(exp(theta(4))) ...
    );
else
    ptheta = {pell; psigma2};
    pprior = @(theta) pell.logpdf(theta(1)) + psigma2.logpdf(exp(theta(2)));
end

% Likelihood
py = struct();
py.fast = 1;
py.logpdf = @(y, f, t)  -1/2*log(2*pi) - f/2 - 1/2*y.^2.*exp(-f);   % lognormpdf(y, 0, a + exp(s));

%% PMCMC
% Initial parameter guesses
if 1
    theta0 = [log(ell1), log(ell2), sigma21, sigma22].';
else
    theta0 = [log(ell1), sigma21].';
end
    
% Model constructor
model = @(theta) gpsmc_model(@() model_sv(exp(theta(1)), exp(theta(2)), theta(3), theta(4), nu1, nu2), [], [], [], py, ptheta, Ts);
    
% Parameter samplers
if 1
    samplers = {
        @sample_variance, 3;
        @sample_variance, 4;
        @sample_lengthscale, [1, 2];
    };
else
    par_ls = struct();
    par_ls.C = 1e-3; %eye(2);

    samplers = { ...
        @sample_variance_rb, 2;
        @(y, t, s, theta, model, iell, ~) sample_lengthscale_rb(y, t, s, theta, model, iell, [], par_ls), 1;
    };
end

% CPF-AS parameters
par_cpf = struct();
% par_cpf.sample_ancestor_index = @(model, y, xt, x, lw, theta) sample_ancestor_index_rs(model, y, xt, x, lw, theta, L);

% PMCMC parameters
par_pmcmc = struct();
par_pmcmc.sample_states = @(y, xtilde, theta, lambda) cpfas(model(theta), y, xtilde, lambda, J, par_cpf);
par_pmcmc.sample_parameters = @(y, t, x, theta, model, state) sample_parameters(y, t, x, theta, model, samplers, state);
fh = pbar(Kmcmc);
par_pmcmc.show_progress = @(p, ~, ~) pbar(round(p*Kmcmc), fh);

% Run PMCMC, takes about 2.5 mins
ts = tic;
[xs_pmcmc, thetas_pmcmc, sys] = gibbs_pmcmc(model, y, theta0, t, K, par_pmcmc);
t_pmcmc = toc(ts);
pbar(0, fh);

%% Estimate posterior E(f | y) and var(f | y)
mod = model(theta0);
C = mod.C;
f_pmcmc = smc_expectation(@(x) C*x, xs_pmcmc(:, 2:end, :));
sigma2_pmcmc = smc_expectation(@(x) (C*x - f_pmcmc).^2, xs_pmcmc(:, 2:end, :));

%% 
ars = zeros(Kmcmc, N+1);
ls = zeros(Kmcmc, N+1);
for k = 1:Kmcmc
    tmp = sys{k};
    for n = 2:N+1
        state = tmp(n).state;
        ls(k, n) = state.l;
        ars(k, n) = state.accepted;
    end
end

%% Plots
figure(10); clf();
plot(f_pmcmc, 'r', 'LineWidth', 2);
plot(log(rv));
plot(squeeze(sum(xs_pmcmc(C == 1, :, :), 1)), 'Color', [0.9, 0.9, 0.9]); hold on;
plot(f_pmcmc, 'r', 'LineWidth', 2);
plot(log(rv));
legend('Posterior mean', 'Realized volatility');

figure(1); clf();
plot(t, y); hold on;
plot(t, exp(f_pmcmc));
plot(t, 2*exp(f_pmcmc));
xlim([1, N]); grid on;
xlabel('t'); ylabel('y_n');
title('Return');
legend('Measured return', '1-sigma of volatility', '2-sigma of volatility');

% Estimated volatility vs. realized volatility
figure(2); clf();
gp_plot(t, f_pmcmc, sigma2_pmcmc); hold on;

% plot(f_pmcmc, 'b'); hold on;
% plot(f_laplace, 'r');
plot(log(data(:, end-1)), '.');
% legend('PMCMC', 'Laplace', 'Realized');
plot(f_pmcmc + 2*sqrt(sigma2_pmcmc), '--b');
plot(f_pmcmc - 2*sqrt(sigma2_pmcmc), '--b');
% plot(f_laplace+2*sqrt(diag(Sigma_laplace)).', '--r');
% plot(f_laplace-2*sqrt(diag(Sigma_laplace)).', '--r');
xlim([1, N]); grid on;
xlabel('t'); ylabel('f_n');
title('Log-Volatility');

% Trace plots
figure(11); clf();
Ntheta = size(thetas_pmcmc, 1);
for i = 1:Ntheta
    subplot(Ntheta, 1, i);
    plot(thetas_pmcmc(i, :));
    title(sprintf('PMCMC Trace Plot, \\theta_{%d}', i));
    
%     figure(20+i); clf();
%     plot(thetas_laplace(i, :));
%     title(sprintf('Laplace Trace Plot, theta_{%d}', i));
end

%
figure(30); clf();
hist(ls(ls < L & ls > 0), 1:L);
xlim([0, L]);
title(sprintf('Ancestor indices RS sampled: %d/%d', sum(sum(sum(ars))), Kmcmc*N))

if 0
figure(11); clf();
subplot(121);
hist(exp(thetas_pmcmc(1, Kburnin+1:end)));
title('Length Scales');
subplot(122);
hist(exp(thetas_pmcmc(2, Kburnin+1:end)));

figure(12); clf();
subplot(121);
hist(thetas_pmcmc(3, Kburnin+1:end));
title('Variances');
subplot(122);
hist(thetas_pmcmc(4, Kburnin+1:end));


% psrf(thetas_pmcmc.')
% xx = reshape(xs_pmcmc, [2*(N+1) 1 Kmcmc]);
% xx = squeeze(xx);
% figure(); plot(psrf(xx.'), '.');

tt = 500;
xx = squeeze(xs_pmcmc(:, tt+1, :));
ff = C*xx;
[pf, lf] = histn(ff, 30);

figure(3); clf();
plot(lf, pf); hold on;

fff = linspace(f_laplace(:, tt)-4*sqrt(Sigma_laplace(tt, tt)), f_laplace(:, tt)+4*sqrt(Sigma_laplace(tt, tt)), 1000);
plot(fff, normpdf(fff, f_laplace(:, tt), sqrt(Sigma_laplace(tt, tt))));
end 
