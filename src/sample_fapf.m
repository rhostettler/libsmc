function [xp, alpha, lq, qstate] = sample_fapf(model, y, x, lw, theta)
% # Sample from the fully adapted importance density (LGSSM only)
% ## Usage
% * `[xp, alpha, lq, qstate] = sample_fapf(model, y, x, lw, theta)`
%
% ## Description
% Samples a set of new samples x[n] from the fully adapted importance 
% density for linear, Gaussian state-space models: For models of the form
%
%   x[n]|x[n-1] ~ N(F*x[n-1], Q),
%   y[n]|x[n] ~ N(G*x[n], R),
%
% the fully adapted auxiliary particle filter samples from
%
%   alpha[n] ~ Cat{w[n-1] N(y; yp, S)},
%   x[n] ~ N(m, P),
%
% with
%
%   mp = F*x,
%   yp = G*mp,
%   S = G*Q*G' + R,
%   K = (G*Q)'/S,
%   m = mp + K*(y-yp),
%   P = Q - K*S*K'.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: Measurement vector y[n].
% * `x`: Samples at x[n-1].
% * `lw`: Log-weights of x[n-1].
% * `theta`: Model parameters.
%
% ## Output
% * `xp`: The new samples x[n].
% * `alpha`: The ancestor indices of x[n].
% * `lq`: 1-times-J vector of the importance density of the jth sample.
% * `qstate`: Sampling algorithm state information, see `resample_ess`.
%
% ## Author
% 2021-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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

    %% Defaults
    narginchk(5, 5);
    qstate = [];

    %% Sampling
    [dx, J] = size(x);
    Idx = eye(dx);
    F = model.px.mean(Idx, theta);
    Q = model.px.cov([], theta);
    G = model.py.mean(Idx, theta);
    R = model.py.cov([], theta);
    
    % Calculate proposal
    mp = F*x;
    S = G*Q*G' + R;
    K = (G*Q)'/S;
    m = mp + K*(y - G*mp);
    P = Q - K*S*K';
    
    % Sample ancestor indices (resampling)
    lqalpha = lw + logmvnpdf((y*ones(1, J)).', (G*mp).', S).';
    qalpha = exp(lqalpha-max(lqalpha));
    qalpha = qalpha/sum(qalpha);
    alpha = resample_stratified(qalpha);
    lqalpha = log(qalpha(alpha));
            
    % Sample state
    xp = m(:, alpha) + chol(P).'*randn(dx, J);
    lqx = logmvnpdf(xp.', m(:, alpha).', P).';
    
    % Importance density evaluated at {xp(j), alpha(j)}.
    lq = lqx + lqalpha;
end
