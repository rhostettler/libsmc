function sys = calculate_particle_lineages(sys, alpha)
% Calculates full particle trajectories by traversing the ancestral tree
%
% SYNOPSIS
%   sys = CALCULATE_PARTICLE_LINEAGES(sys)
%   sys = CALCULATE_PARTICLE_LINEAGES(sys, alpha)
%
% DESCRIPTION
%   Given a filtering particle system consisting of {xf[n], wf[n]} for n =
%   1, ..., N, this function calculates the (degenerate) state trajectories
%   from 1 to N by walking down the ancestral tree backwards in time.
%
%   Note that this function only considers the filtered particles xf and
%   their weights wf, not any other fields that may be present in sys.
%
% PARAMETERS
%   sys     The array of structs containing the particle system.
%
%   alpha   Vector of indices to build the ancestral tree for (e.g. if only
%           one trajectory is required; optional, default: 1:M).
%
% RETURNS
%   sys     Updated particle system.
%
% AUTHOR
%   2017-11-01 -- Roland Hostettler <roland.hostettler@aalto.fi>

    % Create full trajectories by walking down the ancestral tree backwards
    % in time to get the correct particle lineage.
    narginchk(1, 2);
    N = length(sys);
    M = size(sys(N).xf, 2);
    
    % By default, calculate the lineages for all particles
    if nargin < 2 || isempty(alpha)
        alpha = 1:M;
    end
    
    % Traverse particles backwards in time
    for n = N-1:-1:1
        % Get ancestor indices
        alpha = sys(n+1).af(:, alpha);
        
        % Get ancestor particles & weights
        sys(n).xf = sys(n).xf(:, alpha);
        sys(n).wf = sys(n).wf(:, alpha);
    end
end
