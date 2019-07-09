function [xhat, sys] = ps(model, y, theta, Jf, Js, par, sys)
% # Particle smoother
% ## Usage
% * `xhat, = ps(model, y)`
% * `[xhat, sys] = ps(model, y, theta, Jf, Js, par, sys)`
%
% ## Description
% Forward filtering backward simulation particle smoother as described in
% [1].
%
% ## Input
%
%
%
%   y       Ny times N matrix of measurements.
%   t       1 times N vector of timestamps.
%   model   State space model structure.
%   Mf      Number of particles for the filter (if no sys is provided, see
%           below; optional, default: 250).
%   Ms      Number of particles for the smoother (optional, default: 100).
%   par     Structure of additional (optional) parameters. May contain any
%           parameter accepted by bootstrap_pf (if no sys is provided, see
%           below) plus the following FFBSi-specific parameters:
%
%               TODO: Write these out once finalized (in particular wrt
%               rejection-sampling based version)
%
%   sys     Particle system as obtained from a forward filter. If no system
%           is provided, a bootstrap particle filter is run to generate it.
%           sys must contain the following fields:
%
%               xf  Matrix of particles for the marginal filtering density.
%               wf  Vector of particle weights for the marginal filtering
%                   density.
%
% ## Output
%
%
%   xhat    Minimum mean squared error state estimate (calculated using the
%           smoothing density).
%   sys     Particle system (array of structs) with all the fields returned
%           by the bootstrap particle filter (or the ones in the particle
%           system provided as an input) plus the following fields:
%           
%               xs  Nx times M matrix of particles for the joint smoothing
%                   density.
%               ws  1 times M matrix of particle weights for the joint
%                   smoothing density.
%               rs  Indicator whether the corresponding particle was
%                   sampled using rejection sampling or not.
%
%
% ## References
% 1. W. Fong, S. J. Godsill, A. Doucet, and M. West, "Monte Carlo 
%    smoothing with application to audio signal enhancement," IEEE 
%    Transactions on Signal Processing, vol. 50, pp. 438? 449, February 
%    2002.
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

% TODO
%   * Implement rejection sampling w/ adaptive stopping => should go into
%     an outside function
%   * Check how I can merge that with other backward simulation smoothers,
%     e.g. ksd_ps (they use exactly the same logic in the beginning, only
%     the smooth()-function is different
%   * Clean up code for the individual backward functions.

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
