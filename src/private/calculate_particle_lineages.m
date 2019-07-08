function sys = calculate_particle_lineages(sys, alpha)
% # Calculates full particle trajectories by traversing the ancestral tree
% ## Usage
% * `sys = calculate_particle_lineages(sys)`
% * `sys = calculate_particle_lineages(sys, alpha)`
%
% ## Description
% Given a filtering particle system consisting of {x[n], w[n], alpha[n]} 
% for n = 1, ..., N, this function calculates the (degenerate) state 
% trajectories from 1 to N by walking down the ancestral tree backwards in 
% time.
%
% Note that this function only considers the filtered particles 
% (`sys(n).x`) and their weights (`sys(n).w`) but not any other fields that
% may be present in sys.
%
% ## Input
% * `sys`: The array of structs containing the particle system.
% * `alpha`: Vector of indices to build the ancestral tree for (e.g., if 
%   only one trajectory is required; default: 1:J).
%
% ## Output
% * `sys`: Updated particle system with full state trajectories. The
%   following fields are added:
%     - `xf`: Full trajectory samples.
%     - `wf`: Weight of the corresponding trajectory (only applicable to
%       `sys(N).wf`; empty for all other time instants).
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

    % Create full trajectories by walking down the ancestral tree backwards
    % in time to get the correct particle lineage.
    narginchk(1, 2);
    N = length(sys);
    J = size(sys(N).x, 2);
    
    % By default, calculate the lineages for all particles
    if nargin < 2 || isempty(alpha)
        alpha = 1:J;
    end
    
    % Traverse particles backwards in time
    sys(N).xf = sys(N).x(:, alpha);
    sys(N).wf = sys(N).w(:, alpha);
    for n = N-1:-1:1
        % Get ancestor indices
        alpha = sys(n+1).alpha(:, alpha);
        
        % Get ancestor particles & weights
        sys(n).xf = sys(n).x(:, alpha);
        %sys(n).wf = sys(n).w(:, alpha);
    end
end
