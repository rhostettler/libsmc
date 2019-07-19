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
% * Clean up the code

% Housekeeping
clear variables;
addpath ../src;
rng(5011);
if 0
spmd
    warning('off', 'libsmc:warning');
end
end
warning('off', 'libsmc:warning');

%% Parameters
% Filter parameters
J = 100;         % Number of particles
L = 5;           % Number of iterations

% Sigma-points: Assigns weight 1/2 to the central point, same weights for
% mean and covariance
alpha = sqrt(1/(1-0.5));
beta = 1;
kappa = 0;

% Simulation parameters
N = 1000;       % Number of time samples
K = 1;        % Number of MC simulations

% Model parameters
Q = 0.3^2;
m0 = log(7);
P0 = 0.1;

% Save the simulation?
store = false;

%% Model
f = @(x, n) log(44.7) + x - exp(x);
g = @(x, n) 10*exp(x);

px0 = struct( ...
    'rand', @(M) m0*ones(1, M) + chol(P0).'*randn(1, M) ...
);
LQ = sqrt(Q);
px = struct( ...
    'fast', false, ...
    'rand', @(x, theta) f(x, theta) + LQ.'*randn(1, size(x, 2)), ...
    'logpdf', @(xp, x, theta) logmvnpdf(xp.', f(x, theta).', Q.').' ...
);
py = struct( ...
    'fast', false, ...
    'rand', @(x, theta) poissrnd(10*exp(x), 1), ...
    'logpdf', @(y, x, theta) log(poisspdf(y, 10*exp(x))) ...
);
model = struct('px0', px0, 'px', px, 'py', py);

%% Algorithm parameters
% Approximation of the optimal proposal using linearization
Gx = @(x, theta) 10*exp(x);
R = @(x, n) 10*exp(x);
slr_lin = @(m, P, theta) slr_taylor(m, P, theta, g, Gx, R);
par_lin = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_lin, L), ...
    'calculate_incremental_wights', @calculate_incremental_weights_generic ...
);

% SLR using unscented transform
dx = size(m0, 1);
[wm, wc, c] = ut_weights(dx, alpha, beta, kappa);
Xi = ut_sigmas(zeros(dx, 1), eye(dx), c);
slr_sp = @(m, P, theta) slr_sp(m, P, theta, g, R, Xi, wm, wc);
par_sp = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_sp, L), ...
    'calculate_incremental_wights', @calculate_incremental_weights_generic ...
);

% Closed-form solution to the moment integrals
Ey = @(m, P, theta) 10*exp(m + P/2);
Cy = @(m, P, theta) 100*exp(2*m + P)*(exp(P) - 1) + 10*exp(m + P/2);
Cyx = @(m, P, theta) 10*P*exp(m + P/2);
slr_cf = @(m, P, theta) slr_cf(m, P, theta, Ey, Cy, Cyx);
par_cf = struct( ...
    'sample', @(model, y, x, theta) sample_gaussian(model, y, x, theta, f, @(x, theta) Q, slr_cf, L), ...
    'calculate_incremental_weights', @calculate_incremental_weights_generic ...
);

%% MC Simulations
% Preallocate
xs = zeros(1, N, K);
xhat_bpf = zeros(1, N, K);
xhat_lin = zeros(1, N, K);
xhat_sp = zeros(1, N, K);
xhat_cf = zeros(1, N, K);
r_bpf = zeros(1, N+1, K);
r_lin = zeros(1, N+1, K);
r_sp = zeros(1, N+1, K);
r_cf = zeros(1, N+1, K);
t_bpf = zeros(1, K);
t_lin = zeros(1, K);
t_sp = zeros(1, K);
t_cf = zeros(1, K);

fprintf('Simulating with J = %d, L = %d, N = %d, K = %d...\n', J, L, N, K);
fh = pbar(K);
% parfor k = 1:K
for k = 1:K
    %% Simulation
    y = zeros(1, N);
    x = m0 + sqrt(P0)*randn(1);
    for n = 1:N
        x = px.rand(x, n);
        y(:, n) = py.rand(x, n);
        xs(:, n, k) = x;
    end

    %% Estimation
    % Bootstrap PF
    if L == 1
        tic;
        [xhat_bpf(:, :, k), sys_bpf] = pf(model, y, [], J);
        t_bpf(k) = toc;
%         r_bpf(:, :, k) = cat(2, sys_bpf.r);
    else
        xhat_bpf(:, :, k) = NaN*ones(1, N);
    end
    
    % Taylor series approximation of SLR
    tic;
    [xhat_lin(:, :, k), sys_lin] = pf(model, y, [], J, par_lin);
    t_lin(k) = toc;
%     r_lin(:, :, k) = cat(2, sys_lin.r);

    % SLR using sigma-points, L iterations
    tic;
    [xhat_sp(:, :, k), sys_sp] = pf(model, y, [], J, par_sp);
    t_sp(k) = toc;
%     r_sp(:, :, k) = cat(2, sys_sp.r);
    
    % SLR using closed-form expressions, L iterations
    tic;
    [xhat_cf(:, :, k), sys_cf] = pf(model, y, [], J, par_cf);
    t_cf(k) = toc;
%     r_cf(:, :, k) = cat(2, sys_cf.r);

    %% Progress
    pbar(k, fh);
end
pbar(0, fh);


%%
if 0
mps = zeros(L+1, N);
Pps = zeros(L+1, N);

for j = 1:J
    for n = 2:N+1
        mps(:, n) = sys_cf(n).q(j).mp.';
        Pps(:, n) = squeeze(sys_cf(n).q(j).Pp);
    end
if 0
    figure(1); clf();
    plot(mps(1:4, :).');
    legend('l = 0', 'l = 1', 'l = 2', 'l = 3', 'l = 4', 'l = 5');
    title('Mean');
end

    figure(2); clf();
    plot(Pps(1:4, :).');
    legend('l = 0', 'l = 1', 'l = 2', 'l = 3', 'l = 4', 'l = 5');
    title('Variance');
    drawnow();
    pause(1);
end
end

%% Results
iNaN_bpf = squeeze(isnan(xhat_bpf(:, N, :)));
iNaN_lin = squeeze(isnan(xhat_lin(:, N, :)));
iNaN_sp = squeeze(isnan(xhat_sp(:, N, :)));
iNaN_cf = squeeze(isnan(xhat_cf(:, N, :)));

fprintf('\tRMSE\t\t\tTime\t\tResampling\tConvergence\n');
fprintf( ...
    'BPF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), 3), ...
    std(rms(xs(:, :, ~iNaN_bpf) - xhat_bpf(:, :, ~iNaN_bpf)), [], 3), ...
    mean(t_bpf), std(t_bpf), ...
    mean(sum(r_bpf(:, :, ~iNaN_bpf))/N), std(sum(r_bpf(:, :, ~iNaN_bpf))/N), ...
    1-sum(iNaN_bpf)/K ...
);
fprintf( ...
    'LIN\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_lin) - xhat_lin(:, :, ~iNaN_lin)), 3), ...
    std(rms(xs(:, :, ~iNaN_lin) - xhat_lin(:, :, ~iNaN_lin)), [], 3), ...
    mean(t_lin), std(t_lin), ...
    mean(sum(r_lin(:, :, ~iNaN_lin))/N), std(sum(r_sp(:, :, ~iNaN_lin))/N), ...
    1-sum(iNaN_lin)/K ...
);
fprintf( ...
    'SP\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), 3), ...
    std(rms(xs(:, :, ~iNaN_sp) - xhat_sp(:, :, ~iNaN_sp)), [], 3), ...
    mean(t_sp), std(t_sp), ...
    mean(sum(r_sp(:, :, ~iNaN_sp))/N), std(sum(r_sp(:, :, ~iNaN_sp))/N), ...
    1-sum(iNaN_sp)/K ...
);
fprintf( ...
    'CF\t%.2e (%.2e)\t%.2f (%.2f)\t%.2f (%.2f)\t%.2f\n', ...
    mean(rms(xs(:, :, ~iNaN_cf) - xhat_cf(:, :, ~iNaN_cf)), 3), ...
    std(rms(xs(:, :, ~iNaN_cf) - xhat_cf(:, :, ~iNaN_cf)), [], 3), ...
    mean(t_cf), std(t_cf), ...
    mean(sum(r_cf(:, :, ~iNaN_cf))/N), std(sum(r_cf(:, :, ~iNaN_cf))/N), ...
    1-sum(iNaN_cf)/K ...
);

%% Store results
if store
    outfile = sprintf('Savefiles/example_ricker_J=%d_L=%d.mat', J, L);
    save(outfile);
end
