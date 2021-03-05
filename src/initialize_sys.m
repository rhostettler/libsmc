function sys = initialize_sys(N, dx, J)
% # Initializes an empty particle system structure with common fields
% ## Usage
% * `sys = initialize_sys(N, dx, J)`
%
% ## Description
% Initializes an empty particle system with the most commonly used fields
% (see the Output section for a list of the fields).
%
% ## Input
% * `N`: Number of datapoints (including `x[0]`).
% * `dx`: State dimension.
% * `J`: Number of particles.
%
% ## Output
% * `sys`: Array of structs with the following fields:
%     - `x`: Marginal filtering density particles.
%     - `w`: Marginal filtering density particle weights.
%     - `alpha`: Ancestor indices.
%     - `r`: Resampling indicator.
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

    narginchk(3, 3);
    sys = repmat( ...
        struct( ...
            'x', zeros(dx, J), ...    % Marginal filtering density particles
            'w', zeros(1, J), ...     % Marginal filtering density weights
            'alpha', zeros(1, J), ... % Ancestor indices
            'qstate', struct() ...    % Sampling algorithm state
        ), ...
        [N, 1] ...
    );
end
