function model = model_lgssm(F, Q, G, R, m0, P0)
% # Linear Gaussian state-space model
% ## Usage
% * `model = MODEL_LGSSM(F, Q, G, R, m0, P0)`
%
% ## Description
% Creates a model structure with the appropriate probabilistic description
% for linear, Gaussian state-space models. Given the model of the form
%
%     x[n] = F*x[n-1] + q
%     y[n] = G*x[n] + r
%
% where q[n] ~ N(0, Q), r[n] ~ N(0, R), and p(x[0]) = N(m0, P0) (F, G, Q, 
% and R may all depend on t[n] or any other parameter(s)), the function 
% initializes the corresponding transition density and likelihood given by
%
%     p(x[n] | x[n-1]) = N(x[n]; F*x[n-1], Q), and
%     p(y[n] | x[n]) = N(y[n]; G*x[n], R),
%
% respectively.
%
% ## Input
% * `F`, `Q`, `G`, `R`, `m0`, `P0`: Model parameters as described above. If
%   any of the parameters is time-varying or depends on any other
%   parameters, it must be a function handle of the form @(~, theta).
%
% ## Output
% * `model`: Model struct containing px0, px, and py as described above.
%
% ## Authors
% 2017-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * Functionalization is not fully solved yet; should be passed to mvn_pdf
%   directly instead.

    %% Defaults
    narginchk(6, 6);
    
    %% Functionize
    if ~isa(F, 'function_handle')
        F = @(theta) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(theta) Q;
    end
    if ~isa(G, 'function_handle')
        G = @(theta) G;
    end
    if ~isa(R, 'function_handle')
        R = @(theta) R;
    end
    
    %% Create Model
    % Initial state distribution
    dx = size(m0, 1);
    L0 = chol(P0).';
    px0 = struct('rand', @(M) m0*ones(1, M)+L0*randn(dx, M));     % TODO: Should make use of mvn_pdf() as well, but that is not suitable
                                                                  % for initial pdfs yet (actually, it's rather the different filters
                                                                  % that are not ready yet...).
    
    % State transition densiy
if 0
    px = struct( ...
        'fast', 1, ...
        'rand', @(x, t) F(t)*x + chol(Q(t)).'*randn(dx, size(x, 2)), ...
        'logpdf', @(xp, x, t) logmvnpdf(xp.', (F(t)*x).', Q(t).').', ...
        'pdf', @(xp, x, t) mvnpdf(xp.', (F(t)*x).', Q(t).').', ...
        'rho', mvnpdf(zeros(dx, 1), zeros(dx, 1), Q(0)) ...
    );
end
    px = pdf_mvn(@(x, theta) F(theta)*x, @(~, theta) Q(theta), true);

    % Likelihood
if 0
    py = struct( ...
        'fast', 1, ...
        'rand', @(x, t) G(t)*x + chol(R(t)).'*randn(dx, size(x, 2)), ...
        'logpdf', @(y, x, t) logmvnpdf(y.', (G(t)*x).', R(t).').', ...
        'pdf', @(y, x, t) mvnpdf(y.', (G(t)*x).', R(t).').' ...
    );
end
    py = pdf_mvn(@(x, theta) G(theta)*x, @(~, theta) R(theta), true);
    
    % Complete model
if 0
    model = struct( ...
        'F', F, 'Q', Q, 'G', G, 'R', R, 'm0', m0, 'P0', P0, ...
        'px', px, 'py', py, 'px0', px0 ...
    );
end
    model = struct( 'px0', px0, 'px', px, 'py', py);
end
