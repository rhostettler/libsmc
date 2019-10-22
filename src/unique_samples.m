function Ju = unique_samples(x)
% # Calculate the number of unique samples
% ## Usage
% * `Ju = unique_samples(sys)`
%
% ## Description
% Calculates the number of unique samples in a sequential Monte Carlo
% posterior approximation to assess trajectory degeneracy.
%
% ## Inputs
% * `x`: dx-times-J-times-N matrix of state trajectories.
%
% ## Outputs
% * `Ju`: A 1-times-N vector with the number of unique samples for each 
%   time step `n = 1, ..., N`.
%
% ## Authors
% 2018-present -- Roland Hostettler

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

    narginchk(1, 1)
    N = size(x, 3);
    Ju = zeros(1, N);
    for n = 1:N
        tmp = unique(x(:, :, n).', 'rows');
        Ju(:, n) = size(tmp, 1);
    end
end
