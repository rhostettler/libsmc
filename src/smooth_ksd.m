function [xhat, sys] = smooth_ksd(model, y, theta, Js, sys)
% # Kronander-Schon-Dahlin marginal particle smoother
% ## Usage
% * `xhat = smooth_ksd(model, y, theta, Js, sys)`
% * `[xhat, sys] = smooth_ksd(model, y, theta, Js, sys)`
% 
% ## Description
% Backward sampling particle smoother targeting the marginal smoothing
% density according to [1].
% 
% Note that it is well known that this smoother is biased, see [1].
%
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-N matrix of measurements.
% * `theta`: Additional parameters.
% * `Js`: No. of smoothing particles.
% * `sys`: Particle system array of structs.
%
% ## Output
% * `xhat`: dx-times-N matrix of smoothed state estimates (MMSE).
% * `sys`: Particle system array of structs for the smoothed particle
%   system. The following fields are added by `smooth_ffbsi`:
%     - `xs`: Smoothed particles.
%     - `ws`: Smoothed particle weights.
%
% ## References
% 1. J. Kronander, T. B. Schon, and J. Dahlin, "Backward sequential Monte 
%    Carlo for marginal smoothing," in IEEE Workshop on Statistical Signal 
%    Processing (SSP), June 2014, pp. 368-371.
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
    narginchk(5, 5);
    return_sys = (nargout >= 2);
    
    %% Preallocate
    N = length(sys);
    [dx, Jf] = size(sys(N).x);
    xhat = zeros(dx, N);
    lv = zeros(1, Js);
    y = [NaN*size(y, 1), y];
    theta = [NaN*size(theta, 1), theta];
    
    px = model.px;
    py = model.py;

    %% Initialize backward recursion
    ir = resample_stratified(log(sys(N).w));
    alpha = ir(randperm(Jf, Js));
    xs = sys(N).x(:, alpha);
    lw = log(sys(N).w(alpha));
    xhat(:, N) = mean(xs, 2);
    ws = 1/Js*ones(1, Js);
    lws = log(ws);
    
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = ws;
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        % Sample ancestor particles
        if py.fast
            lv = lws + py.logpdf(y(:, n+1)*ones(1, Js), xs, theta(n+1)) - lw;
        else
            for j = 1:Js
                lv(j) = lws(j) + py.logpdf(y(:, n+1), xs(:, j), theta(n+1)) - lw(j);
            end
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        beta = resample_stratified(log(v));
        xp = xs(:, beta);
        
        % Sample smoothed particles
        ir = resample_stratified(log(sys(n).w));
        alpha = ir(randperm(Jf, Js));
        xs = sys(n).x(:, alpha);
        lw = log(sys(n).w(alpha));
                
        % Calculate smoothed particle weights
        if px.fast
            lws = px.logpdf(xp, xs, theta(n+1));
        else
            for j = 1:Js
                lws(:, j) = px.logpdf(xp(:, j), xs(:, j), theta(n+1));
            end
        end
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
        
        % Estimate
        xhat(:, n) = xs*ws';
        
        %% Store
        if return_sys
            sys(n).xs = xs;
            sys(n).ws = ws;
        end
    end
    
    %% Post-processing
    % Strip x[0] as we don't want it in the MMSE estiamte; if needed, it
    % can be obtained from sys.
    xhat = xhat(:, 2:N);
end
