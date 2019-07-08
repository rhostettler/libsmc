function py = pdf_mvn(m, P, fast)
% # Multivariate normal pdf structure
% ## Usage
% * `py = pdf_mvn(m)`
% * `py = pdf_mvn(m, P, fast)`
%
% ## Description
% Initializes the pdf struct for a multivariate normal distribution with
% mean m (mean function m(x, theta)) and covariance P (covariance function
% P(x, theta).
%
% ## Input
% * `m`: Mean vector (dx-times-1). May be static or a function handle of
%   the form @(x, theta).
% * `P`: Covariance matrix (dx-times-dx). May be static or a function
%   handle of the form @(x, theta). Default: `eye(dx)`.
% * `fast`: `true` if `m(x, theta)` and `P(x, theta)` can evaluated for a
%   complete dx-times-J particle matrix at once. Default: `false`.
%
% ## Output
% * `py`: pdf struct with fields:
%   - `rand(x, theta)`: Random sample generator.
%   - `logpdf(y, x, theta)`: Log-pdf.
%   - `fast`: Flag for particle matrix evaluation.
%   - `kappa(x, theta)`: Bounding constant of the pdf such that `p(y) <= 
%      kappa` for all `y`.
%
% ## Authors
%   2019-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
    narginchk(1, 3);
    dx = size(m, 1);
    if nargin < 2 || isempty(P)
        P = eye(dx);
    end
    if nargin < 3 || isempty(fast)
        fast = false;
    end
        
    %% Generate the struct
    if ~isa(m, 'function_handle') && ~isa(P, 'function_handle')
        LP = chol(P).';
        py = struct( ...
            'rand', @(x, theta) m + LP*randn(dx, size(x, 2)), ...
            'logpdf', @(y, x, theta) logmvnpdf(y.', m.', P).', ...
            'fast', fast, ...
            'kappa', @(x, theta) (2*pi)^(-dx/2)*sqrt(det(P)) ...
        );
    elseif ~isa(m, 'function_handle') && isa(P, 'function_handle')
        py = struct( ...
            'rand', @(x, theta) m + chol(P(x, tehta)).'*randn(dx, size(x, 2)), ...
            'logpdf', @(y, x, theta) logmvnpdf(y.', m.', P(x, theta)).', ...
            'fast', fast, ...
            'kappa', @(x, theta) (2*pi)^(-dx/2)*sqrt(det(P(x, theta))) ...
        );
    elseif isa(m, 'function_handle') && ~isa(P, 'function_handle')
        LP = chol(P).';
        py = struct( ...
            'rand', @(x, theta) m(x, theta) + LP*randn(dx, size(x, 2)), ...
            'logpdf', @(y, x, theta) logmvnpdf(y.', m(x, theta).', P).', ...
            'fast', fast, ...
            'kappa', @(x, theta) (2*pi)^(-dx/2)*sqrt(det(P)) ...
        );        
    elseif isa(m, 'function_handle') && isa(P, 'function_handle')
        py = struct( ...
            'rand', @(x, theta) m(x, theta) + chol(P(x, theta)).'*randn(dx, size(x, 2)), ...
            'logpdf', @(y, x, theta) logmvnpdf(y.', m(x, theta).', P(x, theta)).', ...
            'fast', fast, ...
            'kappa', @(x, theta) (2*pi)^(-dx/2)*sqrt(det(P(x, theta))) ...
        );
    else
        error('Something went wrong while initializing pdf struct, check your mean and covariance functions.');
    end
end
