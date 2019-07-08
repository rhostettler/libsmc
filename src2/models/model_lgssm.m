function model = model_lgssm(F, Q, G, R, m0, P0)
% Initializes a linear Gaussian state space model
%
% SYNOPSIS
%   model = MODEL_LGSSM(F, Q, G, R, m0, P0)
%
% DESCRIPTION
%   Creates a model structure with the appropriate probabilistic
%   description for linear, Gaussian state space models. Given the model of
%   the form
%
%       x[n] = F*x[n-1] + q
%       y[n] = G*x[n] + r
%
%   where q[n] ~ N(0, Q), r[n] ~ N(0, R), and p(x[0]) = N(m0, P0) (F, G,
%   Q, and R may all depend on t(n)), the function calculates the
%   corresponding transition density and likelihood, which are given by
%
%       p(x[n] | x[n-1]) = N(x[n]; F*x[n-1], Q), and
%       p(y[n] | x[n]) = N(y[n]; G*x[n], R),
%
%   respectively.
%
% PARAMETERS
%   F, Q, G, R, m0, P0
%           Model parameters. If any of the parameters is time-varying,
%           supply a function handle of the form @(t).
%
% RETURNS
%   model   Model struct containing px0, px, and py as described above.
%
% AUTHORS
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * As it is now, the matrices are also stored; I might consider removing
%     this in the future.
    
    %% Defaults
    narginchk(6, 6);
    
    %% Functionize
    if ~isa(F, 'function_handle')
        F = @(t) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(t) Q;
    end
    if ~isa(G, 'function_handle')
        G = @(t) G;
    end
    if ~isa(R, 'function_handle')
        R = @(t) R;
    end
    
    %% Create Model
    % Initial state distribution
    Nx = size(m0, 1);
    C0 = chol(P0).';
    px0 = struct('rand', @(M) m0*ones(1, M)+C0*randn(Nx, M));
    
    % State transition densiy
    px = struct( ...
        'fast', 1, ...
        'rand', @(x, t) F(t)*x + chol(Q(t)).'*randn(Nx, size(x, 2)), ...
        'logpdf', @(xp, x, t) logmvnpdf(xp.', (F(t)*x).', Q(t).').', ...
        'pdf', @(xp, x, t) mvnpdf(xp.', (F(t)*x).', Q(t).').', ...
        'rho', mvnpdf(zeros(Nx, 1), zeros(Nx, 1), Q(0)) ...
    );

    % Likelihood
    py = struct( ...
        'fast', 1, ...
        'rand', @(x, t) G(t)*x + chol(R(t)).'*randn(Nx, size(x, 2)), ...
        'logpdf', @(y, x, t) logmvnpdf(y.', (G(t)*x).', R(t).').', ...
        'pdf', @(y, x, t) mvnpdf(y.', (G(t)*x).', R(t).').' ...
    );
    
    % Complete model
    model = struct( ...
        'F', F, 'Q', Q, 'G', G, 'R', R, 'm0', m0, 'P0', P0, ...
        'px', px, 'py', py, 'px0', px0 ...
    );
end
