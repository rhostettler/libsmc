function sys = initialize_sys(N, Nx, M)
% Initializes an empty particle system structure with common fields
%
% SYNOPSIS
%   sys = INITIALIZE_SYS(N, Nx, M)
%
% DESCRIPTION
%   Initializes an empty particle system with the most commonly used fields
%   (marginal particles, their weights, ancestor indices, and resampling
%   indicator).
%
% PARAMETERS
%   N   Number of datapoints (including x[0]).
%   Nx  State dimension.
%   M   Number of particles.
%
% RETURNS
%   sys Array of structures with the following fields:
%
%           x       Marginal filtering density particles.
%           w       Marginal filtering density particle weights.
%           alpha   Ancestor indices.
%           r       Resampling indicator.
%
% AUTHORS
%   2017-11-02 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(3, 3);
    sys = repmat( ...
        struct( ...
            'x', zeros(Nx, M), ...    % Marginal filtering density particles
            'w', zeros(1, M), ...     % Marginal filtering density weights
            'alpha', zeros(1, M), ... % Ancestor indices
            'r', false ...                % Resampling indicator
        ), ...
        [N, 1] ...
    );
end
