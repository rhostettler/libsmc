function model = model_nonlinear_gaussian(f, Q, g, R, m0, P0)
% Nonlinear Gaussian State-Space Model
%
% USAGE
%   model = MODEL_NONLINEAR_GAUSSIAN(f, Q, g, R, m0, P0)
%
% DESCRIPTION
%   Defines the model structure for nonlinear state-space models with
%   Gaussian process- and measurement noise of the form
%
%       x[0] ~ N(m0, P0)
%       x[n] = f(x[n-1], n) + q[n],
%       y[n] = g(x[n], n) + r[n],
%
%   where q[n] ~ N(0, Q[n]) and r[n] ~ N(0, R[n]).
%
% PARAMETERS
%   f, Q    Dynamic model.
%   g, R    Observation model.
%   m0, P0  Mean and covariance of the initial state.
%
% RETURNS
%   model   The model structure that contains the usual fields (the
%           probabilistic representation of the state-space model, i.e.
%           px0, px, py).
%
% AUTHORS
%   2018-12-17 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Check how we can set the "fast" flag
%   * Make "functionization" of all the variables; also if they should
%     include parameters (rather than 't')

    %% Defaults
    narginchk(6, 6);
if 0
    if ~isa(F, 'function_handle')
        F = @(t) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(t) Q;
    end
    if ~isa(R, 'function_handle')
        R = @(t) R;
    end
end
    
    %% Initialize model struct
    % Initial state
    Nx = size(m0, 1);
    L0 = chol(P0).';
    px0 = struct();
    px0.rand = @(M) m0*ones(1, M)+L0*randn(Nx, M);
    
    % State transition densiy
    LQ = chol(Q).';
    px = struct();
    px.fast = true;
    px.rand = @(x, t) f(x, t) + LQ*randn(Nx, size(x, 2));
    px.logpdf = @(xp, x, t) logmvnpdf(xp.', f(x, t).', Q.').';
    
    % Likelihood
    py = struct();
    py.fast = true;
    py.logpdf = @(y, x, t) logmvnpdf(y.', g(x, t).', R.').';
    
    % Complete model
    model = struct('px0', px0, 'px', px, 'py', py);
end
