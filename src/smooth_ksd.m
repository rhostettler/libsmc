function [xhat, sys] = smooth_ksd(model, y, theta, Js, sys)
% # Kronander-Schon-Dahlin marginal particle smoother
% ## Usage
% 
% ## Description
%
% ## Input
%
% ## Output
%
% ## References
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

% TODO:
% * This file is WIP, **do not add to version control**
% * Documentation
% * Split into functions where appropriate

    %% Defaults
    narginchk(5, 5);
    return_sys = (nargout >= 2);
    
    %% Preallocate
    N = length(sys);
    [dx, Jf] = size(sys(N).x);
    xhat = zeros(dx, N);
    y = [NaN*size(y, 1), y];
    theta = [NaN*size(theta, 1), theta];
    
    % TODO: Maybe we don't need these here?
    px = model.px;
    py = model.py;

    %% Initialize backward recursion
    ir = resample_stratified(sys(N).w);
    beta = ir(randperm(Jf, Js));
    xs = sys(N).x(:, beta);
    xhat(:, N) = mean(xs, 2);
    ws = 1/Js*ones(1, Js);
    lws = log(ws);
    
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = ws;
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        % Sample ancestors
        lw = -Inf*ones(1, Js);
        [indf, locf] = ismember(xs.', sys(n+1).x.', 'rows'); %%%% this just finds 'beta again?'
        lw(indf) = log(sys(n+1).w(:, locf));
        if py.fast
            lv = lws + py.logpdf(y(:, n+1)*ones(1, Js), xs, theta(n+1)) - lw;
        else
            lv = zeros(1, Js);
            for j = 1:Js
                lv(j) = lws(j) + py.logpdf(y(:, n+1), xs(:, j), theta(n+1)) - lw(j);
            end
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        beta = resample_stratified(v);
        xp = xs(:, beta);
        
        % Sample new particles
        ir = resample_stratified(sys(n).w);
        alpha = ir(randperm(Jf, Js));
        xs = sys(n).x(:, alpha);
                
        % Calculate weights
        lws = calculate_smoothed_weights(xp, xs, theta(n+1), px);
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
        
        %% Estimate
        xhat(:, n) = xs*ws';
        
        %% Store
        if return_sys
            sys.xs(:, :, n) = xs;
            sys.ws(:, :, n) = ws;
        end
    end
    
    %% Post-processing
    % Strip x[0] as we don't want it in the MMSE estiamte; if needed, it
    % can be obtained from sys.
    xhat = xhat(:, 2:N);
end

%% Smoothed Weights
function lws = calculate_smoothed_weights(xp, x, t, px)
    %% Calculate Weights
    if px.fast
        lws = px.logpdf(xp, x, t);
    else
        Js = size(xp, 2);
        lws = zeros(1, Js);
        for j = 1:Js
            lws(:, j) = px.logpdf(xp(:, j), x(:, j), t);
        end
    end
end
