function model = model_clgss_mixed()
% Mixed linear/non-linear Gaussian state-space model
% 
% SYNOPSIS
%   model = MODEL_CLGSS_MIXED()
%
% DESCRIPTION
%   A mixed linear/non-linear Gaussian state-space model of the form
%
%       xn[n] = fn(xn[n-1], u[n], t[n])
%                   + An(xn[n-1], u[n], t[n]) xl[n-1] + qn[n]
%       xl[n] = fl(xn[n-1], u[n], t[n])
%                   + Al(xn[n-1], u[n], t[n]) xl[n-1] + ql[n]
%       y[n] = h(xn[n], u[n], t[n]) + C(xn[n], u[n], t[n]) xl[n-1] 
%                   + r[n]
%   where
%
%       * q[n] = [qn[n], ql[n]]^T ~ N(0, Q(xn[n-1], u[n], t[n]),
%       * r[n] ~ N(0, R(xn[n], u[n], t[n])),
%       * x[0] = [xn[0], xl[0]]^T ~ N(m0, P0).
%
%   This class can be used in two ways:
%
%       1. As a convenience class for simple models where the different
%          functions and/or matrices are easily defined as function 
%          handles in a script.
%
%       2. As the parent class for more complex models that directly
%          implement the corresponding functions. In this form, all the
%          functions must be implemented by the subclass.
%
% PROPERTIES
%   This class inhertis the properties defined by the AWGNModel and
%   defines the following additional properties.
%
%   in (r/w, default: [])
%       Vector of indices to the non-linear states in the state vector.
%
%   il (r/w, default: [])
%       Vector of the indices to the linear states in the state vector.
%
% METHODS
%   MixedCLGSSModel(fn, An, fl, Al, Q, h, C, R, m0, P0, in, il)
%       Constructor that initializes the model when the simple form is 
%       used. In this case, all arguments must be provided.
%
%       fn, An, fl, Al, Q, h, C, R 
%           Function handles of the form @(xn, t, u) corresponding to
%           the respective function/matrix as defined in the model
%           above.
%
%       m0, P0
%           Initial state distribution mean and covariance.
%
%       in, il
%           Indices of the non-linear and linear states in the state
%           vector such that
%
%               xn = x(in)
%               xl = x(il)
%
%           where x is the overall state vector.
%
% SEE ALSO
%   AWGNModel, RBGF
%
% AUTHORS
%   2017-11-15 -- Roland Hostettler <roland.hostettler@aalto.fi>
    


    narginchk();
    
    %% 
    
    
    
    model = struct();
    model.in = in;
    model.il = il;
    
    %% Initial state
    model.m0 = m0;
    model.P0 = P0;
    model.px0 = struct();
    model.px0.fast = true;
    model.px0.rand = @(M) m0*ones(1, M) + chol(P0).'*randn(Nx, M);
    model.px0.logpdf = @(x) logmvnpdf(x.', m0.', P0.').';
    
    %% Dynamics
    model.fn = fn;
    model.Fn = Fn;
    model.Qn = Q(in
    model.fl = fl;
    model.Fl = Fl;
    model.Q = Q;
    model.Ql = Q(il, il);
    
    f = @(x, t) [
        fn(x(:, in), t) + Fn(x(:, in), t)*x(:, il);
        fl(x(:, in), t) + Fl(x(:, in), t)*x(:, il);
    ];
    Q = @(s, t) [
        Qn(
    
    model.px = struct();
    model.px.fast = true;
    model.px.rand = @(x, t) f(x, t) + chol(Q(x(:, in), t)).'*randn(Nx, size(x, 2));
    model.px.logpdf = @(xp, x, t) logmvnpdf(xp.', f(x, t).', Q(x(:, in), t)).';
    
    %% Likelihood
    model.h = h;
    model.C = C;
    model.R = R;
    
    model.py = struct();
    model.py.fast = true;
    g = @(x, t) h(x(:, in), t) + C(x(:, in), t)*x(:, il);
    model.py.logpdf = @(y, x, t) logmvnpdf(y.', g(x, t).', R(x(:, in), t)).';
end
