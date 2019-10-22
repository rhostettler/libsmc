function [xhat, sys] = ps(model, y, theta, Jf, Js, par, sys)
% # Generic forward-backward particle smoother
% ## Usage
% * `xhat, = ps(model, y)`
% * `[xhat, sys] = ps(model, y, theta, Jf, Js, par, sys)`
%
% ## Description
% Particle smoothing based on forward-backward smoothing where a
% traditional filter is run in the forward direction and a refining
% smoothing pass is run in the backward direction.

% Filtering can either be done outside of the `ps` function itself, for 
% example, using a custom filtering algorithm. In this case, the `sys`
% struct returned by the filter has to be passed to `ps`. If no valid
% `sys` struct is passed, a bootstrap particle filter is used for the
% forward pass.
%
% For smoothing, the forward-filtering, backward-simulation (FFBSi) 
% smoother is used by default. However, the actual smoothing (backward) run
% can be customized through the `par.smooth` parameter and arbitrary 
% smoothing algorithms can be implemented.
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: Additional parameters.
% * `Jf`: Number of particles to be used in the forward filter (if no `sys`
%   is provided, see below; default: 250).
% * `Js`: Number of particles for the smoother (default: 100).
% * `par`: Struct of additional parameters. The following parameters are
%   supported:
%     - `[xhat, sys] = par.smooth(model, y, theta, Js, sys)`: The actual
%       smoothing function used for the backward recursion (default:
%       `@smooth_ffbsi`).
% * `sys`: Particle system as obtained from a forward filter. If no system
%   is provided, a bootstrap particle filter is run to generate it. `sys`
%   must contain the following fields:
%     - `x`: Matrix of particles for the marginal filtering density.
%     - `w`: Vector of particle weights for the marginal filtering density.
%
% ## Output
% * `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
% * `sys`: Particle system array of structs for the smoothed particle
%   system. At least the following fields are added (additional fields may
%   be added by the specific backward recursions):
%     - `xs`: Smoothed particles.
%     - `ws`: Smoothed particle weights.
%
% ## References
% 1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
%    smoothing with application to audio signal enhancement," IEEE 
%    Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
%    2002.
%
% ## Authors
% 2017-present -- Roland Hostettler

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
    narginchk(2, 7);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(Jf)
        Jf = 250;
    end
    if nargin < 5 || isempty(Js)
        Js = 100;
    end
    if nargin < 6
        par = struct();
    end
    def = struct(...
    	'smooth', @smooth_ffbsi ...
    );
    par = parchk(par, def);
    
    % Expand 'theta'
    if size(theta, 2) == 1
        N = size(y, 2);
        theta = theta*ones(1, N);
    end

    %% Filter
    % If no filtered system is provided, run a bootstrap PF
    if nargin < 7 || isempty(sys)
        [~, sys] = pf(model, y, theta, Jf);
    end
    
    %% Smooth
    return_sys = (nargout >= 2);
    if return_sys
        [xhat, sys] = par.smooth(model, y, theta, Js, sys);
    else
        xhat = par.smooth(model, y, theta, Js, sys);
    end
end
