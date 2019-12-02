function n = wnorm(x, W)
% # Calculates the weighted norm x'*W*x
% ## Usage
% * `n = wnorm(x, P)`
%
% ## Description
% Efficiently calculates the weighted norm `x^T W x` for N-times-M matrix 
% `x` and N-times-N matrix `W` using Cholesky decomposition.
%
% ## Input
% * `x`: N-times-M matrix where each column corresponds a vector.
% * `W`: Positive definite N-times-N weight matrix.
%
% ## Output
% * `n`: 1-times-M vector of weighted norms.
%
% ## Authors
% * 2017-present -- Roland Hostettler

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

    %% Sanity Checks
    narginchk(1, 2);
    if nargin < 2 || isempty(W)
        W = eye(size(x, 1));
    end

    %% Calculations
    L = chol(W);  % W = L'*L
    f = L*x;
    n = sum(f.^2, 1);   % (L*x)'*(L*x) = x'*L'*L*x = x*W*x
end
