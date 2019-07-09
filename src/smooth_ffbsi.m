function [xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys, par)
% # Forward-filtering backward-simulation particle smoothing
% ## Usage
% * `[xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys)`
% * `[xhat, sys] = smooth_ffbsi(model, y, theta, Js, sys, par)`
%
% ## Description
% Forward filtering backward simulation particle smoother as described in
% [1].
%
% ## Input
%
% ## Output
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

    %% Defaults
    narginchk(5, 6);
    if nargin < 6 || isempty(par)
        par = struct();
    end
    def = struct();
    par = parchk(par, def);
    
    %% Backward recursion
    px = model.px;
    N = length(sys);
    [dx, Jf] = size(sys(N).x);
    xhat = zeros(dx, N);
    theta = [NaN*size(theta, 1), theta];
    
    %% Initialize
    ir = resample_stratified(sys(N).w);
    b = ir(randperm(Jf, Js));
    xs = sys(N).x(:, b);
    xhat(:, N) = mean(xs, 2);
    
    return_sys = (nargout >= 2);
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = 1/Js*ones(1, Js);
        sys(N).rs = zeros(1, Js);
    end
    
    %% Backward recursion
    for n = N-1:-1:1
        %% Sample trajectory backwards
        % TODO: We should be able to set this also through a parameter. I
        %       believe that is implemented in cpfas or cpfas_ps. Check
        %       there (i.e. rather than par.rs, have a
        %       par.sample_backward_particles()).
        if 1%~par.rs
            xs = sample_backward_particle(xs, sys(n).x, theta(n+1), log(sys(n).w), px);
            rs = zeros(1, Js);
        else
            [xs, rs] = sample_backward_particle_rs(xs, sys(n).x, theta(n+1), log(sys(n).w), px);
        end

        %% Estimate & Store
        xhat(:, n) = mean(xs, 2);
        if return_sys
            sys(n).xs = xs;
            sys(n).ws = 1/Js*ones(1, Js);
            sys(n).rs = rs;
        end
    end
    
    % Strip x[0] as we don't want it in the MMSE estiamte; if needed, it
    % can be obtained from sys.
    xhat = xhat(:, 2:N);
end

%% 
function xs = sample_backward_particle(xs, x, t, lw, px)
% Compute the backward smoothing weights
% j -> trajectory to expand
% i -> candidate particles
        
    Ms = size(xs, 2);
    Mf = size(x, 2);
    for j = 1:Ms
        lv = calculate_transition_weights(xs(:, j), x, t, px);
        lwb = lw + lv;
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);

        % Draw a new particle from the categorical distribution and
        % extend the trajectory
        ir = resample_stratified(wb);
        alpha = ir(randi(Mf, 1));
        xs(:, j) = x(:, alpha);
    end
end

%%
function [xs, rs] = sample_backward_particle_rs(xs, x, t, lw, px)
% TODO: It appears like there's a bug somewhere here.

    Mf = size(x, 2);
    Ms = size(xs, 2);
    L = 10;
    rs = zeros(1, Ms);
    
    % TODO: Scales O(Ms*log(Ms)) because we don't sample for all j at once,
    % but that will do for now
    ir = sysresample(exp(lw));
    for j = 1:Ms
        l = 0;
        done = 0;
        lv = zeros(1, Mf);
        iv = zeros(1, Mf);
        while ~done
            % Sample from prior
            alpha = ir(randi(Mf, 1));
            
            % Calculate non-normalized weight
            lv(alpha) = calculate_transition_weights(xs(:, j), x(:, alpha), t, px);
            iv(alpha) = 1;

            % Calculate upper bound on normalizing constant
            u = rand(1);
            paccept = (exp(lv(alpha))/px.rho);
            if paccept > 1
                warning('Acceptance probability larger than one, check your bounding constant.');
            end
            accepted = (u < paccept);

            l = l+1;
            done = accepted || (l >= L);
        end
        if ~accepted
            % Exhaustive search for the non-calculated ones
            %lv(~iv) = calculate_transition_weights(xs(:, j), x(:, ~iv), t, px);
            lv = calculate_transition_weights(xs(:, j), x, t, px);
            lv = lw + lv;
            v = exp(lv-max(lv));
            v = v/sum(v);
            tmp = sysresample(v);
            alpha = tmp(randi(Mf, 1));
        end
        xs(:, j) = x(:, alpha);
        rs(:, j) = accepted;
    end
end

%%
function lv = calculate_transition_weights(xp, x, t, px)
    M = size(x, 2);
    if px.fast
        lv = px.logpdf(xp*ones(1, M), x, t);
    else
        lv = zeros(1, M);
        for i = 1:M
           lv(i) = px.logpdf(xp, x(:, i), t);
        end
    end
end
