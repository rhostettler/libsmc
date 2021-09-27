function [xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys, par)
% # Forward-filtering backward-simulation particle smoothing
% ## Usage
% * `xhat = smooth_ffbsi(model, y, theta, Js, sys)`
% * `[xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys, par)`
%
% ## Description
% Forward-filtering backward-simulation (FFBSi) particle smoother as
% described in [1].
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: Additional parameters.
% * `Js`: No. of smoothing particles.
% * `sys`: Particle system array of structs.
% * `par`: Algorithm parameter struct, may contain the following fields:
%     - `[beta, state] = sample_backward_simulation(model, xs, x, lw, theta)`:
%       Function to sample from the backward smoothing kernel (default:
%       `@sample_backward_simulation`).
%
% ## Output
% * `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
% * `sys`: Particle system array of structs for the smoothed particle
%   system. The following fields are added by `smooth_ffbsi`:
%     - `xs`: Smoothed particles.
%     - `ws`: Smoothed particle weights (`1/Js` for FFBSi).
%     - `state`: State of the backward simulation sampler.
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

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct( ...
        'sample_backward_simulation', @sample_backward_simulation ...
    );
    par = parchk(par, def);
    
    %% Preallocate
    N = length(sys);
    [dx, Jf] = size(sys(N).x);
    xhat = zeros(dx, N);
    theta = [NaN*size(theta, 1), theta];
    
    %% Initialize
    ir = resample_stratified(log(sys(N).w));
    beta = ir(randperm(Jf, Js));
    xs = sys(N).x(:, beta);
    xhat(:, N) = mean(xs, 2);
    
    return_sys = (nargout >= 2);
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = 1/Js*ones(1, Js);
        sys(N).state = [];
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        %% Sample backward trajectory
        [beta, state] = par.sample_backward_simulation(model, xs, sys(n).x, log(sys(n).w), theta(n+1));
        xs = sys(n).x(:, beta);
        xhat(:, n) = mean(xs, 2);
        
        %% Store
        if return_sys
            sys(n).xs = xs;
            sys(n).ws = 1/Js*ones(1, Js);
            sys(n).state = state;
        end
    end
    
    %% Post-processing
    % Strip x[0] as we don't want it in the MMSE estiamte; if needed, it
    % can be obtained from sys.
    xhat = xhat(:, 2:N);
end
