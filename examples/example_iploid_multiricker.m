% Ricker population example
%
% Example of particle filtering with the iterated-conditional-
% expectations-based importance density. The model considered here is the
% Ricker population model
%
%   x[n] = log(44.7) + x[n-1] - exp(x[n-1]) + q[n]
%   y[n] ~ P(10*exp(x[n]))
%   x[0] ~ N(log(7), 0.1)
%
% with q[n] ~ N(0, 0.3^2).
%
% 2019 -- Roland Hostettler

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

% TODO:
% * Update header
% * Update model
% * Update inference
% * Clean up code
% * Extend to different carrying capacities and process noises
% * check iterations, how many are used for high dimensions?
% * Increase dimensions (needs to recalculate the Ey, Cy, Cyx)

% Housekeeping
clear variables;
addpath ../src;
rng(5011);
warning('off', 'all');

%% Parameters
% Grid for the grid filter
xg = -5:1e-3:5;

% Filter parameters
J = 5000;                   % Number of particles (N.B.: For the iterated variants, J/L particles are used!)
L = 5;                     % Max. number of iterations

% Simulation parameters
N = 100;                    % Number of time samples
K = 20;                     % Number of MC simulations

% Model parameters
dx = 3;                     % No of patches (we assume a line only here)
r = 1;                      % Growth rate
c = 1;                      % Migration parameter (scaling of distance)
m = 0.1;                    % Migration rate
rho = 0.05;                 % Species event probability
sigma2v = 0.3^2;            % Species process noise variance
Qu = 1*eye(dx);             % Intra-patch process noise magnitude U(-l, l)
C = 20;                     % Carrying capacity
m0 = log(C)*ones(dx, 1);    % Initial mean
P0 = 0.1*eye(dx);           % Initial covariance
beta = 0.5;                 % Likelihood skewness/tail (0 = poisson)

% Which filters to run
use_grid = false;
use_bpf = true;
use_gf = false;             % TODO: Implement Gaussian flow
use_lin = false;
use_sp = false;
use_cf1 = true;
use_cf = true;

% Other switches
plots = true;
store = false;% Save the simulation?

% 
epsilon = 1e-3;

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha_sp = sqrt(1/(1-0.5));
beta_sp = 1;
kappa_sp = 0;

% Other model parameters
theta = 1:N;

%% Model
% Model struct
px0 = struct( ...
    'rand', @(M) m0*ones(1, M) + chol(P0).'*randn(dx, M), ...
    'logpdf', @(x, theta) logmvnpdf(x.', m0.', P0).' ...
);

% Dynamic model
if dx > 1
    [A, B] = meshgrid(1:dx, 1:dx);
    D = exp(-c*sqrt((B-A).^2));
    D(D == 1) = 0;
    D = D./(sum(D, 2)*ones(1, dx));
    H = (1-m)*eye(dx) + m*D;
    z = @(x) H*exp(x);
else
    z = @(x) exp(x);
end
f = @(x, theta) log(z(x)) + r*(1 - z(x)/C);
Lv = ones(dx, 1);
Qv = sigma2v*(Lv*Lv');
Q = @(x, theta) Qu + rho*Qv;
px = struct( ...
    'fast', true, ...
    'rand', @(x, theta) ( ...
        f(x, theta) + chol(Qu).'*randn([dx, size(x, 2)]) ...
        + (rand(1) < rho)*Lv*sqrt(sigma2v)*randn([1, size(x, 2)]) ...
    ), ...
    'logpdf', @(xp, x, theta) log( ...
        exp(log(1-rho)+logmvnpdf(xp.', f(x, theta).', Qu).') ...
        + exp(log(rho)+logmvnpdf(xp.', f(x, theta).', Qu + Qv').') ...
    ) ...
);
% w/ uniform process noise
% Q = @(x, theta) 1/12*(2*l)^2*eye(dx) + rho*sigma2*ones(dx, dx);
%     'rand', @(x, theta) f(x, theta) + (-l+2*l*rand([dx, size(x, 2)])) + ones(dx, 1)*(rand(1) < rho)*sqrt(sigma2)*randn(1, size(x, 2))... %, ...
%     'logpdf', @(xp, x, theta) logmvnpdf(xp.', f(x, theta).', Q(x, theta).').' ...

% Measurement model
g = @(x, theta) exp(x);                      % Mean
Gx = @(x, theta) diag(exp(x));               % Jacobian of mean
R = @(x, theta) diag(exp(x)/(1-beta).^2);    % Covariance  

py = struct( ...
    'fast', true, ...
    'rand', @(x, theta) gpoissonrnd(exp(x).*(1-beta), beta, size(x)), ...
    'logpdf', @(y, x, theta) sum(loggpoissonpdf(y, exp(x).*(1-beta), beta), 1) ...
);
model = struct('px0', px0, 'px', px, 'py', py);

%% Algorithm parameters
% Approximation of the optimal proposal using linearization
slr_lin = @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R);
par_lin = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, Q, slr_lin, L, [], epsilon), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% SLR using unscented transform
dx = size(m0, 1);
[wm, wc, c] = ut_weights(dx, alpha_sp, beta_sp, kappa_sp);
Xi = ut_sigmas(zeros(dx, 1), eye(dx), c);
slr_sp = @(m, P, theta) slr_sp(m, P, theta, g, R, Xi, wm, wc);
par_sp = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, Q, slr_sp, L, [], epsilon), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% Closed-form solution to the moment integrals
Ey = @(m, P, theta) exp(m + diag(P)/2);
Cy = @(m, P, theta) ( ...
    1/(1-beta)^2*diag(exp(m + diag(P)/2)) ...
    +(exp(P)-1).*exp((m + diag(P)/2)*ones(1, dx) + ones(dx, 1)*(m + diag(P)/2)') ...
);
Cyx = @(m, P, theta) P.*exp((m+diag(P)/2)*ones(1, dx));
if 0
a = 1;
Ey = @(m, P, theta) a*exp(m + P/2)*ones(dx, 1);
Cy = @(m, P, theta) (a^2*exp(2*m + P).*(exp(P) - 1) + a*exp(m + P/2)/(1-beta).^2)*eye(dx);
Cyx = @(m, P, theta) a*P*exp(m + P/2);
end

slr_cf = @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx);
par_cf1 = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, Q, slr_cf), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);
par_cf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, Q, slr_cf, L, [], epsilon), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

%% MC Simulations
% Preallocate
xs = zeros(dx, N, K);
ys = zeros(dx, N, K);

xhat_bpf = zeros(dx, N, K);
xhat_grid = xhat_bpf;
xhat_lin = xhat_bpf;
xhat_sp = xhat_bpf;
xhat_cf1 = xhat_bpf;
xhat_cf = xhat_bpf;

ess_bpf = zeros(1, N, K);
ess_lin = ess_bpf;
ess_sp = ess_bpf;
ess_cf1 = ess_bpf;
ess_cf = ess_bpf;

l_lin = zeros(1, K);
l_sp = l_lin;
l_cf = l_lin;

r_bpf = zeros(1, N, K);
r_lin = r_bpf;
r_sp = r_bpf;
r_cf1 = r_bpf;
r_cf = r_bpf;

t_bpf = zeros(1, K);
t_grid = t_bpf;
t_lin = t_bpf;
t_sp = t_bpf;
t_cf1 = t_bpf;
t_cf = t_bpf;

if use_grid
    NGrid = length(xg);
    w = zeros(N, NGrid, K);
end

fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);
for k = 1:K
    %% Simulation
    [xs(:, :, k), ys(:, :, k)] = simulate_model(model, theta, N);
    
    %% Estimation
    if use_grid && dx == 1
        tic;
        [xhat_grid(:, :, k), w(:, :, k)] = gridf(model, ys(:, :, k), theta, xg);
        t_grid(k) = toc;
    end

    % Bootstrap PF
    if use_bpf
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).rstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
        r_bpf(:, :, k) = cat(2, tmp.r);
    end
    
    if use_gf
    end
    
    % Taylor series approximation of SLR
    if use_lin
        tic;
        [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J/L, par_lin);
        t_lin(k) = toc;
        tmp = cat(2, sys_lin(2:N+1).rstate);
        ess_lin(:, :, k) = cat(2, tmp.ess);
        r_lin(:, :, k) = cat(2, tmp.r);
        tmp = cat(1, sys_lin(2:N+1).q);
        l_lin(k) = mean(cat(1, tmp.l));
    end

    % SLR using sigma-points, L iterations
    if use_sp
        tic;
        [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J/L, par_sp);
        t_sp(k) = toc;
        tmp = cat(2, sys_sp(2:N+1).rstate);
        ess_sp(:, :, k) = cat(2, tmp.ess);
        r_sp(:, :, k) = cat(2, tmp.r);
        tmp = cat(1, sys_sp(2:N+1).q);
        l_sp(k) = mean(cat(1, tmp.l));
    end
    
    % One-step OID approximation
    if use_cf1
        tic;
        [xhat_cf1(:, :, k), sys_cf1] = pf(model, ys(:, :, k), theta, J, par_cf1);
        t_cf1(k) = toc;
        tmp = cat(2, sys_cf1(2:N+1).rstate);
        ess_cf1(:, :, k) = cat(2, tmp.ess);
        r_cf1(:, :, k) = cat(2, tmp.r);
    end
    
    % SLR using closed-form expressions, L iterations
    if use_cf
        tic;
        [xhat_cf(:, :, k), sys_cf] = pf(model, ys(:, :, k), theta, L, par_cf);
        t_cf(k) = toc;
        tmp = cat(2, sys_cf(2:N+1).rstate);
        ess_cf(:, :, k) = cat(2, tmp.ess);
        r_cf(:, :, k) = cat(2, tmp.r);
        tmp = cat(1, sys_cf(2:N+1).q);
        l_cf(k) = mean(cat(1, tmp.l));
    end

    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance figures
iNaN_bpf = squeeze(isnan(xhat_bpf(1, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(1, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(1, N, :)));
iNaN_cf1 = squeeze(isnan(xhat_cf1(1, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(1, N, :)));

e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));
e_rmse_cf1 = trmse(xhat_cf1(:, :, ~iNaN_cf1) - xs(:, :, ~iNaN_cf1));
e_rmse_cf = trmse(xhat_cf(:, :, ~iNaN_cf) - xs(:, :, ~iNaN_cf));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\tIterations\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_lin), std(e_rmse_lin), mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K, mean(l_lin), std(l_lin) ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_sp), std(e_rmse_sp), mean(t_sp), std(t_sp), ...
    mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
    1-sum(iNaN_sp)/K, mean(l_sp), std(l_sp) ...
);
fprintf( ...
    'CF1\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\n', ...
    mean(e_rmse_cf1), std(e_rmse_cf1), mean(t_cf1), std(t_cf1), ...
    mean(sum(r_cf1(:, :, ~iNaN_cf1))/N), std(sum(r_cf(:, :, ~iNaN_cf1))/N), ...
    1-sum(iNaN_cf1)/K ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\n', ...
    mean(e_rmse_cf), std(e_rmse_cf), mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N), std(sum(r_cf(:, :, ~iNaN_cf))/N), ...
    1-sum(iNaN_cf)/K, mean(l_cf), std(l_cf) ...
);

%% Plots
if plots
    if K == 1
    figure(1); clf();
    plot(xs.'); hold on; grid on;
    plot(xhat_bpf.', '--');
    % plot(xhat_lin.', '-.');
    plot(xhat_sp.', '-.');
    title('States');
    end

    % ESS
    figure(3); clf();
    plot(mean(ess_bpf(:, :, ~iNaN_bpf)/J, 3)); hold on; grid on;
    plot(mean(ess_lin(:, :, ~iNaN_lin)/(J/L), 3));
    plot(mean(ess_sp(:, :, ~iNaN_sp)/(J/L), 3));
    plot(mean(ess_cf1(:, :, ~iNaN_cf1)/J, 3));
    plot(mean(ess_cf(:, :, ~iNaN_cf)/(J/L), 3));
    legend('BPF', 'ICE-Taylor', 'ICE-SP', 'ICE-CF1', 'ICE-CF');
    title('Effective sample size');

    % Posterior
    j = 12;
    if use_grid
        lpy = zeros(1, NGrid);
        for n = 1:N
            for ngrid = 1:NGrid
                lpy(ngrid) = model.py.logpdf(ys(:, n, k), xg(:, ngrid), theta);
            end
            lpx = model.px.logpdf(xg, sys_cf(n).x(:, sys_cf(n+1).alpha(j))*ones(1, NGrid), theta);
            lpoid = lpy+lpx;
            poid = exp(lpoid)/(1e-3*sum(exp(lpoid)));        

            figure(4); clf();
            plot(xg, w(n, :)); hold on; grid on;
            plot(xg, poid);
    %         addlegendentry('Posterior');
            if use_bpf
                plot(sys_bpf(n+1).x, zeros(1, J), '.');
    %             addlegendentry('Bootstrap');
            end
            if use_sp
                plot(sys_sp(n+1).x, zeros(1, J/L), 'o');
    %             addlegendentry('ICE-SP');
            end
            if use_cf
                plot(sys_cf(n+1).x, zeros(1, J/L), 'o');
    %             addlegendentry('ICE-SP');
            end

            legend('Posterior', 'OID', 'Bootstrap', 'ICE-SP');

            qj = sys_cf(n+1).q(j);

            figure(5); clf();
            plot(xg, exp(lpy)); hold on; grid on;
            plot(xg, poid, '--');
            if use_bpf
                plot(xg, exp(model.px.logpdf(xg, sys_bpf(n).x(:, sys_bpf(n+1).alpha(j))*ones(1, NGrid), theta)));
            end
            legend('Likelihood', 'OID for j', 'Bootstrap proposal');
            for Qu = 1:qj.l+1
                plot(xg, normpdf(xg, qj.mp(Qu), sqrt(qj.Pp(:, :, Qu))));
            end
            title(sprintf('Likelihood, OID, and ID at %d', n));

            pause();
        end
    end
    
    % 
    for i = 1:dx
        figure(20+i); clf();
        plot(exp(xs(i, :, k))); hold on; grid on;
        plot(ys(i, :, k));
    end
end

%% Store results
if store
    
% % % % % % % %     TODO: Update these.
    clear sys_bpf sys_cf1 sys_cf
    
    % Store
    outfile = sprintf('Savefiles/example_ungm_J=%d_L=%d_K=%d_N=%d.mat', J, L, K, N);
    
    outfile = sprintf('Savefiles/example_multiricker_J=%d_L=%d.mat', J, L);
    save(outfile);
end
