function model = model_wiener_ssm(F, Q, g, R, m0, P0)
% Wiener state-space model
%
% USAGE
%   model = MODEL_WIENER_SSM(F, Q, g, R, m0, P0)
%
% DESCRIPTION
%   Defines the model structure for Wiener state-space models of the form
%
%       x[0] ~ N(m0, P0)
%       x[n] = F(t[n]) x[n-1] + q[n],
%       y[n] = g(x, t[n]) + r[n],
%
%   where q[n] ~ N(0, Q[n]) and r[n] ~ N(0, R[n]).
%
% PARAMETERS
%   F, Q    Matrices of the dynamic model.
%   g, R    Observation model.
%   m0, P0  Mean and covariance of the initial state.
%
% RETURNS
%   model   The model structure that contains the usual fields (the
%           probabilistic representation of the state-space model, i.e.
%           px0, px, py) as well as the following model-specific fields:
%
%               F, Q    Matrices of the dynamic model,
%               g, R    Observation model,
%               m0, P0  Mean and covariance of the initial state.
%
% AUTHORS
%   2018-05-18 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(6, 6);
    if ~isa(F, 'function_handle')
        F = @(t) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(t) Q;
    end
    if ~isa(R, 'function_handle')
        R = @(t) R;
    end
    
    %% Model structure
    % Initial state
    Nx = size(m0, 1);
    L0 = chol(P0).';
    px0 = struct();
    px0.rand = @(M) m0*ones(1, M)+L0*randn(Nx, M);
    
    % State transition densiy
    px = struct();
    px.fast = true;
    px.rand = @(x, t) F(t)*x + chol(Q(t)).'*randn(Nx, size(x, 2));
    px.logpdf = @(xp, x, t) logmvnpdf(xp.', (F(t)*x).', Q(t).').';
    
    % Likelihood
    py = struct();
    py.fast = true;
    py.logpdf = @(y, x, t) logmvnpdf(y.', g(x, t).', R(t).').';
    
    % Complete model
    model = struct( ...
        'F', F, 'Q', Q, 'g', g, 'R', R, 'm0', m0, 'P0', P0, ...
        'px', px, 'py', py, 'px0', px0 ...
    );
end
