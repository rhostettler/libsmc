% Multi-dimensional Ricker population example
%
% Example of particle filtering with the iterated-conditional-
% expectations-based (ICE-based) optimal importance density (OID) 
% approximation. The model considered here is a multi-dimensional Ricker 
% population model.
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

% Housekeeping
clear variables;
addpath ../src;
rng(5011);
warning('off', 'all');

%% Parameters
% Grid for the grid filter
xg = -5:1e-3:5;

% Filter parameters
J_bpf = 250;              % Number of particles
J_cf1 = 500;
J_gf = 100;
J_pfpf = 100;
J_cf = 500;
L = 10;                     % Max. number of iterations (10)
L_gf = 10;                  % GF integration steps
L_pfpf = 29;                % PFPF integration steps
epsilon = 1e-3;             % Convergence threshold (1e-3)

% Simulation parameters
N = 10;                    % Number of time samples (100)
K = 2;                    % Number of MC simulations (100)

% Model parameters
dx = 5;                    % No. of patches; equals state dimension (10)
r = 1;                      % Growth rate (1)
c = 1;                      % Migration parameter (scaling of distance) (1)
m = 0.1;                    % Migration rate (0.1)
rho = 0.05;                 % Species event probability (0.05)
sigma2v = 0.3^2;            % Species process noise variance (0.3^2)
Qu = 1*eye(dx);             % Intra-patch process noise variance (1*eye(dx))
C = 20;                     % Carrying capacity (20)
m0 = log(C)*ones(dx, 1);    % Initial mean; somewhere around the carrying capacity
P0 = 0.1*eye(dx);           % Initial covariance (0.1*eye(dx))
beta = 0.5;                 % Likelihood skewness/tail (0 = poisson; 0.5)

% Which filters to run
use_grid = false;           % Grid filter; run this only when dx = 1
use_bpf = true;             % Bootstrap PF
use_cf1 = true;             % One-step closed form OID approximation
use_gf = true;              % Gaussian flow OID approximation
use_pfpf = true;            % Particle flow PF with LEDH flow
use_cf = true;              % Closed-form ICE OID approximation
use_lin = false;            % Linearization ICE OID approximation
use_sp = false;             % Sigma-point ICE OID approximation

% Other switches
plots = false;              % Show debugging plots?
store = false;              % Save the simulation?

% Other algorithm parameters
gammaT = 0.95;               % Gating tail probability (0.95)

% Sigma-points (for sigma-point moment approximation): Assigns weight 1/2 
% to the central point, same weights for mean and covariance
alpha_sp = sqrt(1/(1-0.5));
beta_sp = 1;
kappa_sp = 0;

%% Initial state
px0 = struct( ...
    'rand', @(M) m0*ones(1, M) + chol(P0).'*randn(dx, M), ...
    'logpdf', @(x, theta) logmvnpdf(x.', m0.', P0).', ...
    'm', m0, 'P', P0 ...
);

%% Dynamic model
if dx > 1
    [A, B] = meshgrid(1:dx, 1:dx);
    D = exp(-c*sqrt((B-A).^2));
    D(D == 1) = 0;
    D = D./(sum(D, 2)*ones(1, dx));
    H = (1-m)*eye(dx) + m*D;
else
    H = 1;
end
z = @(x) H*exp(x);
dzdx = @(x) H.*(exp(x)*ones(1, dx));
f = @(x, theta) log(z(x)) + r*(1 - z(x)/C);
Fx = @(x, theta) 1./z(x).*dzdx(x) - r*dzdx(x)/C;
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
    ), ...
    'm', f, ...
    'dm', Fx, ...
    'P', Qu + rho*Qv ...
);
% w/ uniform process noise
% Q = @(x, theta) 1/12*(2*l)^2*eye(dx) + rho*sigma2*ones(dx, dx);
%     'rand', @(x, theta) f(x, theta) + (-l+2*l*rand([dx, size(x, 2)])) + ones(dx, 1)*(rand(1) < rho)*sqrt(sigma2)*randn(1, size(x, 2))... %, ...
%     'logpdf', @(xp, x, theta) logmvnpdf(xp.', f(x, theta).', Q(x, theta).').' ...

%% Measurement model
g = @(x, theta) submat(exp(x), theta == 1, 1);                              % Mean
Gx = @(x, theta) submat(diag(exp(x)), theta == 1);                          % Jacobian of mean
dGxdx = @(x, theta) multiricker_dGxdx(x, theta);                            % Matrices of second derivatives
R = @(x, theta) submat(diag(exp(x)/(1-beta).^2), theta == 1, theta == 1);   % Covariance

% Closed-form solution to the moment integrals
Ey = @(m, P, theta) submat(exp(m + diag(P)/2), theta == 1);
Cy = @(m, P, theta) submat( ...
    1/(1-beta)^2*diag(exp(m + diag(P)/2)) ...
    +(exp(P)-1).*exp((m + diag(P)/2)*ones(1, dx) + ones(dx, 1)*(m + diag(P)/2)'), ...
    theta == 1, theta == 1 ...
);
Cyx = @(m, P, theta) submat(P.*exp((m+diag(P)/2)*ones(1, dx)), theta == 1);

% Likelihood
py = struct( ...
    'fast', true, ...
    'rand', @(x, theta) (theta*ones(1, size(x, 2))).*gpoissonrnd(exp(x).*(1-beta), beta, size(x)), ...
    'logpdf', @(y, x, theta) multiricker_lpy(y, x, theta, beta), ...
    'm', g, ...
    'dm', Gx, ...
    'P', R, ...
    'ytr', @(y, theta) submat(y, theta == 1, 1) ...
);

% Model struct
model = struct('px0', px0, 'px', px, 'py', py);

%% Algorithm parameters
% One-step OID approximation, closed-form moments
slr_cf = @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx);
par_cf1 = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y(theta == 1, :), x, theta, f, Q, slr_cf, 1, gammaT), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% Gaussian flow OID approximation
par_gf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian_flow(model, y(theta == 1, :), x, theta, f, Q, g, Gx, dGxdx, R, L_gf), ...
    'calculate_incremental_weights', @calculate_incremental_weights_flow ...
);

% Particle flow particle filter
par_pfpf = struct( ...
    'L', L_pfpf, ...
    'ukf', [alpha_sp, beta_sp, kappa_sp] ...
);

% ICE OID approximation, closed-form moments
par_cf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y(theta == 1, :), x, theta, f, Q, slr_cf, L, gammaT, epsilon), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

% % ICE OID approximation, Taylor series moment approximation
% slr_lin = @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R);
% par_lin = struct( ...
%     'sample', @(model, y, x, theta) sample_gaussian(model, y(theta == 1, :), x, theta, f, Q, slr_lin, L, gammaT, epsilon), ...
%     'calculate_incremental_weights', @calculate_incremental_weights_generic ...
% );
% 
% % ICE OID approximation, sigma-point moment approximation
% dx = size(m0, 1);
% [wm, wc, c] = ut_weights(dx, alpha_sp, beta_sp, kappa_sp);
% Xi = ut_sigmas(zeros(dx, 1), eye(dx), c);
% slr_sp = @(m, P, theta) slr_sp(m, P, theta, g, R, Xi, wm, wc);
% par_sp = struct( ...
%     'sample', @(model, y, x, theta) sample_gaussian(model, y(theta == 1, :), x, theta, f, Q, slr_sp, L, gammaT, epsilon), ...
%     'calculate_incremental_weights', @calculate_incremental_weights_generic ...
% );

%% Preallocate
xs = zeros(dx, N, K);
ys = zeros(dx, N, K);

xhat_bpf = zeros(dx, N, K);
xhat_grid = xhat_bpf;
xhat_cf1 = xhat_bpf;
xhat_gf = xhat_bpf;
xhat_pfpf = xhat_bpf;
xhat_cf = xhat_bpf;
% xhat_lin = xhat_bpf;
% xhat_sp = xhat_bpf;

ess_bpf = zeros(1, N, K);
ess_cf1 = ess_bpf;
ess_gf = ess_bpf;
ess_pfpf = ess_bpf;
ess_cf = ess_bpf;
% ess_lin = ess_bpf;
% ess_sp = ess_bpf;

l_cf = zeros(N*J_cf, K);

r_bpf = zeros(1, N, K);
r_cf1 = r_bpf;
r_gf = r_bpf;
r_pfpf = r_bpf;
r_cf = r_bpf;
% r_lin = r_bpf;
% r_sp = r_bpf;

t_bpf = zeros(1, K);
t_cf1 = t_bpf;
t_gf = t_bpf;
t_pfpf = t_bpf;
t_cf = t_bpf;
t_grid = t_bpf;
% t_lin = t_bpf;
% t_sp = t_bpf;

use_grid = use_grid && dx == 1;
if use_grid
    NGrid = length(xg);
    w = zeros(N, NGrid, K);
end

%% MC simulations
fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J_bpf, L, N, K);
fh = pbar(K);
for k = 1:K
    %% Simulation
    theta = rand([dx, N]) < 0.5;    % Randomly generate measurement instants
    [xs(:, :, k), ys(:, :, k)] = simulate_model(model, theta, N);
    
    %% Estimation
    if use_grid
        tic;
        [xhat_grid(:, :, k), w(:, :, k)] = gridf(model, ys(:, :, k), theta, xg);
        t_grid(k) = toc;
    end

    % Bootstrap PF
    if use_bpf
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, ys(:, :, k), theta, J_bpf);
        t_bpf(k) = toc;
        tmp = cat(2, sys_bpf(2:N+1).rstate);
        ess_bpf(:, :, k) = cat(2, tmp.ess);
        r_bpf(:, :, k) = cat(2, tmp.r);
    end
    
    % One-step OID approximation
    if use_cf1
        if J_cf1 < 5000
            tic;
            [xhat_cf1(:, :, k), sys_cf1] = pf(model, ys(:, :, k), theta, J_cf1, par_cf1);
            t_cf1(k) = toc;
            tmp = cat(2, sys_cf1(2:N+1).rstate);
            ess_cf1(:, :, k) = cat(2, tmp.ess);
            r_cf1(:, :, k) = cat(2, tmp.r);
        else
            tic;
            xhat_cf1(:, :, k) = pf(model, ys(:, :, k), theta, J_cf1, par_cf1);
            t_cf1(k) = toc;
        end
    end

    % Gaussian flow
    if use_gf
        tic;
        [xhat_gf(:, :, k), sys_gf] = pf(model, ys(:, :, k), theta, J_gf, par_gf);
        t_gf(k) = toc;
        tmp = cat(2, sys_gf(2:N+1).rstate);
        ess_gf(:, :, k) = cat(2, tmp.ess);
        r_gf(:, :, k) = cat(2, tmp.r);
    end
    
    % Particle flow PF
    if use_pfpf
        tic;
        [xhat_pfpf(:, :, k), sys_pfpf] = pfpf(model, ys(:, :, k), theta, J_pfpf, par_pfpf);
        t_pfpf(k) = toc;
        tmp = cat(2, sys_pfpf(2:N+1).rstate);
        ess_pfpf(:, :, k) = cat(2, tmp.ess);
        r_pfpf(:, :, k) = cat(2, tmp.r);
    end
    
    % SLR using closed-form expressions, L iterations
    if use_cf
        if J_cf < 5e3
            tic;
            [xhat_cf(:, :, k), sys_cf] = pf(model, ys(:, :, k), theta, J_cf, par_cf);
            t_cf(k) = toc;
            tmp = cat(2, sys_cf(2:N+1).rstate);
            ess_cf(:, :, k) = cat(2, tmp.ess);
            r_cf(:, :, k) = cat(2, tmp.r);
            tmp = cat(1, sys_cf(2:N+1).qstate);
            l_cf(:, k) = cat(1, tmp.l);
        else
            tic;
            xhat_cf(:, :, k) = pf(model, ys(:, :, k), theta, J_cf, par_cf);
            t_cf(k) = toc;
        end
    end
    
%     % Taylor series approximation of SLR
%     if use_lin
%         tic;
%         [xhat_lin(:, :, k), sys_lin] = pf(model, ys(:, :, k), theta, J_bpf/L, par_lin);
%         t_lin(k) = toc;
%         tmp = cat(2, sys_lin(2:N+1).rstate);
%         ess_lin(:, :, k) = cat(2, tmp.ess);
%         r_lin(:, :, k) = cat(2, tmp.r);
%         tmp = cat(1, sys_lin(2:N+1).q);
%         l_lin(:, k) = cat(1, tmp.l);
%     end
% 
%     % SLR using sigma-points, L iterations
%     if use_sp
%         tic;
%         [xhat_sp(:, :, k), sys_sp] = pf(model, ys(:, :, k), theta, J_bpf/L, par_sp);
%         t_sp(k) = toc;
%         tmp = cat(2, sys_sp(2:N+1).rstate);
%         ess_sp(:, :, k) = cat(2, tmp.ess);
%         r_sp(:, :, k) = cat(2, tmp.r);
%         tmp = cat(1, sys_sp(2:N+1).q);
%         l_sp(:, k) = cat(1, tmp.l);
%     end
        
    %% Progress
    pbar(k, fh);
end
pbar(0, fh);

%% Performance figures
iNaN_bpf = squeeze(isnan(xhat_bpf(1, N, :)));
iNaN_cf1 = squeeze(isnan(xhat_cf1(1, N, :)));
iNaN_gf = squeeze(isnan(xhat_gf(1, N, :)));
iNaN_pfpf = squeeze(isnan(xhat_pfpf(1, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(1, N, :)));
% iNaN_lin = squeeze(isnan(xhat_lin(1, N, :)));
% iNaN_sp = squeeze(isnan(xhat_sp(1, N, :)));

e_rmse_grid = trmse(xhat_grid - xs);
e_rmse_bpf = trmse(xhat_bpf(:, :, ~iNaN_bpf) - xs(:, :, ~iNaN_bpf));
e_rmse_cf1 = trmse(xhat_cf1(:, :, ~iNaN_cf1) - xs(:, :, ~iNaN_cf1));
e_rmse_gf = trmse(xhat_gf(:, :, ~iNaN_gf) - xs(:, :, ~iNaN_gf));
e_rmse_pfpf = trmse(xhat_pfpf(:, :, ~iNaN_pfpf) - xs(:, :, ~iNaN_pfpf));
e_rmse_cf = trmse(xhat_cf(:, :, ~iNaN_cf) - xs(:, :, ~iNaN_cf));
% e_rmse_lin = trmse(xhat_lin(:, :, ~iNaN_lin) - xs(:, :, ~iNaN_lin));
% e_rmse_sp = trmse(xhat_sp(:, :, ~iNaN_sp) - xs(:, :, ~iNaN_sp));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\tIterations\tESS\t\tESS (%%)\n');
fprintf( ...
    'Grid\t%.2e (%.2e)\t%.2f (%.2f)\tn/a\t\tn/a\t\tn/a\t\tn/a\t\tn/a\n', ...
    mean(e_rmse_grid), std(e_rmse_grid), mean(t_grid), std(t_grid) ...
);
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\t\t%.2f (%.2f)\t%.2f (%.2f)\n', ...
    mean(e_rmse_bpf), std(e_rmse_bpf), mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N*100), std(sum(r_bpf(:, :, ~iNaN_bpf))/N*100), ...
    1-sum(iNaN_bpf)/K, ...
    mean(mean(ess_bpf)), std(mean(ess_bpf)), ...
    mean(mean(ess_bpf/J_bpf*100)), std(mean(ess_bpf/J_bpf*100)) ...
);
fprintf( ...
    'CF1\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\t\t%.2f (%.2f)\t%.2f (%.2f)\n', ...
    mean(e_rmse_cf1), std(e_rmse_cf1), mean(t_cf1), std(t_cf1), ...
    mean(sum(r_cf1(:, :, ~iNaN_cf1))/N*100), std(sum(r_cf(:, :, ~iNaN_cf1))/N*100), ...
    1-sum(iNaN_cf1)/K, ...
    mean(mean(ess_cf1)), std(mean(ess_cf1)), ...
    mean(mean(ess_cf1/J_cf1*100)), std(mean(ess_cf1/J_cf1*100)) ...
);
fprintf( ...
    'GF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\t\t%.2f (%.2f)\t%.2f (%.2f)\n', ...
    mean(e_rmse_gf), std(e_rmse_gf), mean(t_gf), std(t_gf), ...
    mean(sum(r_gf(:, :, ~iNaN_gf))/N*100), std(sum(r_gf(:, :, ~iNaN_gf))/N*100), ...
    1-sum(iNaN_gf)/K, ...
    mean(mean(ess_gf)), std(mean(ess_gf)), ...
    mean(mean(ess_gf/J_gf*100)), std(mean(ess_gf/J_gf*100)) ...
);
fprintf( ...
    'PFPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\tn/a\t\t%.2f (%.2f)\t%.2f (%.2f)\n', ...
    mean(e_rmse_pfpf), std(e_rmse_pfpf), mean(t_pfpf), std(t_pfpf), ...
    mean(sum(r_pfpf(:, :, ~iNaN_pfpf))/N*100), std(sum(r_pfpf(:, :, ~iNaN_pfpf))/N*100), ...
    1-sum(iNaN_pfpf)/K, ...
    mean(mean(ess_pfpf(:, :, ~iNaN_pfpf))), std(mean(ess_pfpf(:, :, ~iNaN_pfpf))), ...
    mean(mean(ess_pfpf(:, :, ~iNaN_pfpf)/J_pfpf*100)), std(mean(ess_pfpf(:, :, ~iNaN_pfpf)/J_pfpf*100)) ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f (%.2f)\n', ...
    mean(e_rmse_cf), std(e_rmse_cf), mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N*100), std(sum(r_cf(:, :, ~iNaN_cf))/N*100), ...
    1-sum(iNaN_cf)/K, mean(l_cf(:)), std(l_cf(:)), ...
    mean(mean(ess_cf)), std(mean(ess_cf)), ...
    mean(mean(ess_cf/J_cf*100)), std(mean(ess_cf/J_cf*100)) ...
);
% fprintf( ...
%     'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t\t%.2f (%.2f)\n', ...
%     mean(e_rmse_lin), std(e_rmse_lin), mean(t_lin), std(t_lin), ...
%     mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
%     1-sum(iNaN_lin)/K, mean(l_lin), std(l_lin) ...
% );
% fprintf( ...
%     'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t\t%.2f (%.2f)\t\t%.2f (%.2f)\n', ...
%     mean(e_rmse_sp), std(e_rmse_sp), mean(t_sp), std(t_sp), ...
%     mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
%     1-sum(iNaN_sp)/K, mean(l_sp), std(l_sp) ...
% );

%% Plots
% Relative ESS
figure(1); clf();
plot(mean(ess_bpf(:, :, ~iNaN_bpf)/J_bpf, 3)); hold on; grid on;
plot(mean(ess_cf1(:, :, ~iNaN_cf1)/J_cf1, 3));
plot(mean(ess_gf(:, :, ~iNaN_gf)/(J_gf), 3));
plot(mean(ess_cf(:, :, ~iNaN_cf)/(J_cf/L), 3));
% plot(mean(ess_lin(:, :, ~iNaN_lin)/(J/L), 3));
% plot(mean(ess_sp(:, :, ~iNaN_sp)/(J/L), 3));
set(gca, 'YScale', 'log');
xlabel('n'); ylabel('Relative ESS');
legend('BPF', 'OID CF1', 'GF', 'ICE-CF', 'ICE-Taylor', 'ICE-SP');
title('Effective sample size (relative)');

% Absolute ESS
figure(2); clf();
plot(mean(ess_bpf(:, :, ~iNaN_bpf), 3)); hold on; grid on;
plot(mean(ess_cf1(:, :, ~iNaN_cf1), 3));
plot(mean(ess_gf(:, :, ~iNaN_gf), 3));
plot(mean(ess_cf(:, :, ~iNaN_cf), 3));
% plot(mean(ess_lin(:, :, ~iNaN_lin), 3));
% plot(mean(ess_sp(:, :, ~iNaN_sp), 3));
ylim([10, 1000]);
set(gca, 'YScale', 'log');
xlabel('n'); ylabel('ESS');
legend('BPF', 'OID CF1', 'GF', 'ICE-CF', 'ICE-Taylor', 'ICE-SP');
title('Effective sample size (absolute)');

% Number of iterations for approximating the OID
figure(3); clf();
hist(l_cf(:), 1:L);
xlabel('l'); ylabel('No. of occurences');
title('Number of iterations');

%% [debug] Plots for debugging
if plots && 0
    % Plot the posterior for each time step
    if use_grid
        j = 12;
        lpy = zeros(1, NGrid);
        for n = 1:N
            for ngrid = 1:NGrid
                lpy(ngrid) = model.py.logpdf(ys(:, n, k), xg(:, ngrid), theta);
            end
            lpx = model.px.logpdf(xg, sys_cf(n).x(:, sys_cf(n+1).alpha(j))*ones(1, NGrid), theta);
            lpoid = lpy+lpx;
            poid = exp(lpoid)/(1e-3*sum(exp(lpoid)));        

            figure(6); clf();
            plot(xg, w(n, :)); hold on; grid on;
            plot(xg, poid);
            if use_bpf
                plot(sys_bpf(n+1).x, zeros(1, J_bpf), '.');
            end
            if use_sp
                plot(sys_sp(n+1).x, zeros(1, J_bpf/L), 'o');
            end
            if use_cf
                plot(sys_cf(n+1).x, zeros(1, J_bpf/L), 'o');
            end
            legend('Posterior', 'OID', 'Bootstrap', 'ICE-SP');

            qj = sys_cf(n+1).q(j);

            figure(7); clf();
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

    % States
    if K == 1
        figure(1); clf();
        plot(xs.'); hold on; grid on;
        plot(xhat_bpf.', '--');
        % plot(xhat_lin.', '-.');
%         plot(xhat_sp.', '-.');
        title('States');
    end

    % States and measurements
    % N.B.: This will generate 2*dx plots!
    for i = 1:dx
        figure(100+i); clf();
        plot(xs(i, :, k)); hold on;
        plot(xhat_bpf(i, :, k));
        plot(xhat_cf(i, :, k));
        legend('True state', 'BPF', 'ICE-CF');
        title(sprintf('State %d', i));

        figure(200+i); clf();
        plot(exp(xs(i, :, k))); hold on; grid on;
        plot(ys(i, :, k));
        legend('Population', 'Measurement');
        title(sprintf('Population %d', i));
    end
end
% [/debug]

%% [debug] Store results
if store
    clear sys_bpf sys_cf1 sys_gf sys_cf;
    outfile = sprintf('Save/example_multiricker2_J=%d_L=%d_K=%d_N=%d.mat', J_bpf, L, K, N);
%     save(outfile);
    if 0
        if use_bpf
            save(outfile, 'e_rmse_bpf', '-append');
        end
        if use_cf1
            save(outfile, 'e_rmse_cf1', '-append');
        end
        if use_gf
            save(outfile, 'e_rmse_gf', '-append');
        end
        if use_pfpf
            save(outfile, 'e_rmse_pfpf', '-append');
        end
        if use_cf
            save(outfile, 'e_rmse_cf', '-append');
        end
    end
end
% [/debug]