function Mu = unique_samples(x)
% Calculate the number of unique particles
%
% USAGE
%   Mu = UNIQUE_PARTICLES(x)
%
% DESCRIPTION
%   Calculates the number of unique particles in a sequential Monte Carlo
%   approximation.
%
% PARAMETERS
%   x   3D matrix of state trajectories. x must be of dimension Nx*N*M
%       where Nx is the state dimension, N is the number of time samples,
%       and M is the number of particles.
%
% RETURNS
%   Mu  A 1xN vector with the number of unique samples for each time step
%       n = 1, ..., N.
%
% AUTHORS
%   2018-05 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Make useable with 'sys' structures as well

    narginchk(1, 1)
    N = size(x, 2);
    Mu = zeros(1, N);
    for n = 1:N
        tmp = unique(squeeze(x(:, n, :)).', 'rows').';
        Mu(:, n) = size(tmp, 2);
    end
end
