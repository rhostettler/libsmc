% UNGM example
%
% 
% Model:
%   x[n] = 0.5*x[n-1] + 25*x[n-1]/(1+x[n-1]^2) + 8*cos(1.2*n) + q[n]
%   y[n] = 0.05*x[n]^2 + r[n]
%
% Q = 10, R = 1, x[0] ~ N(0, 5) 

% Housekeeping
clear variables;
addpath(genpath('../src'));

%% Parameters
% Filter parameters
M = 100;       % Number of particles

% Simulation parameters
N = 100;

% Model parameters
Q = 10;
R = 1;
m0 = 0;
P0 = 5;

%% Model
% f = @(x, n) 0.5*x + 25*x./(1+x.^2) + 8*cos(1.2*n);
% g = @(x, n) 0.05*x.^2;
% Gx = @(x) 0.1*x;
f = @(x, n) x;
g = @(x, n) x;
Gx = @(x, n) 1;

% libsmc-type model
model = model_nonlinear_gaussian(f, Q, g, R, m0, P0);

%% Simulation
% Preallocate
xs = zeros(1, N);
y = zeros(1, N);

% Simulate
x = m0 + sqrt(P0)*randn(1);
for n = 1:N
    q = sqrt(Q)*randn(1);
    x = f(x, n) + q;
    r = sqrt(R)*randn(1);
    
    y(:, n) = g(x, n) + r;
    xs(:, n) = x;
end

%% Estimation
% Bootstrap PF
[xhat_bpf, sys_bpf] = pf(y, 1:N, model, M);

% Approximation of the optimal proposal using linearization (not APF!)
S = @(x, n) Gx(x)*Q*Gx(x)' + R;
K = @(x, n) Q*Gx(x)'/S(x);
mq = @(y, x, n) f(x, n) + K(x, n)*(y - g(f(x, n), n));
Pq = @(x, n) Q - K(x, n)*S(x, n)*K(x, n)';
q = struct();
q.fast = false;
q.rand = @(y, x, n) mq(y, x, n) + chol(Pq(x, n)).'*randn(1, size(x, 2));
q.logpdf = @(xp, y, x, n) logmvnpdf(xp.', mq(y, x, n).', Pq(x, n).').';

par_lin = struct( ...
    'sample', @(y, x, n, model) sample_generic(y, x, n, model, q), ...
    'calculate_incremental_weights', @(y, xp, x, n, model) calculate_incremental_weights_generic(y, xp, x, n, model, q) ...
);
[xhat_lin, sys_lin] = pf(y, 1:N, model, M, par_lin);


%% proposed method

Ex = @(x, n) f(x, n);
Vx = @(x, n) Q;
Ey_x = @(x, n) g(x, n);
%VEy_x = @(x, n) Gx(x)*Vx(x, n)*Gx(x)';
Vy_x = @(x, n) R;
%Cyx = @(x, n) Gx(x)*Vx(x, n);

par_gaapf = struct( ...
... %     'calculate_moments', @(y, x, t) calculate_moments2(y, x, t, Ex, Vx, Ey_x, VEy_x, Vy_x, Cyx) ...
    'calculate_proposal', @(y, x, t) calculate_gaussian_proposal_sp(y, x, t, Ex, Vx, Ey_x, Vy_x) ...
);

[xhat_apf, sys_apf] = gaapf(y, 1:N, model, M, par_gaapf);

%% Plots
figure(1); clf();
plot(xs); hold on;
plot(xhat_bpf);
plot(xhat_lin);
plot(xhat_apf);
legend('State', 'Bootstrap', 'Linearized', 'APF');

[rms(xs-xhat_bpf), rms(xs-xhat_lin), rms(xs-xhat_apf)]