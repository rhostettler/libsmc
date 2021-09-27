function [alpha, state] = sample_ancestor_index_rs(model, y, xt, x, lw, theta, L)
% # Rejection-sampling-based ancestor index sampling for CPF-AS
% ## Usage
% * `[alpha, state] = SAMPLE_ANCESTOR_INDEX_RS(model, y, xt, x, lw, theta, L)`
% 
% ## Description
% Samples the ancestor index for the seed trajectory in the conditional
% particle filter with ancestor sampling (CPF-AS) using rejection sampling
% (for Markovian state-space models).
% 
% ## Input
% * `model`: State-space model struct.
% * `y`: dy-times-1 measurement vector.
% * `xtilde`: dx-times-1 sample of the seed trajectory, xtilde[n].
% * `x`: dx-times-J matrix of particles x[n-1]^j.
% * `lw`: 1-times-J row vector of particle log-weights log(w[n-1]^j).
% * `theta`: Additional parameters.
% * `L`: Maximum number of rejection sampling trials before falling back
%   to sampling from the categorical distribution (default: `10`).
%
% ## Output
% * `alpha`: Sampled ancestor index.
% * `state`: Internal state of the sampler. Struct that contains the
%   following fields:
%     - `l`: Number of rejection sampling trials performed.
%     - `accepted`: `true` if the ancestor index was sampled using
%       rejection sampling, `false` otherwise.
%     - `dgamma`: Difference in true acceptance probability and the lower
%       bound used in rejection sampling.
%
% ## Authors
% 2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

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
% * We should separate between debugging and state output.

    %% Defaults
    narginchk(6, 7);
    if nargin < 7 || isempty(L)
        L = 10;
    end
    
    %% Initialize
    % Non-normalized ancestor weights and indicator whether it has been
    % calculated or not
    J = size(x, 2);
    lvtilde = zeros(1, J);
    ivtilde = zeros(1, J);
    
    % Bounding constant on the acceptance probability with normalization
    % constant removed, for uniform sampling alpha ~ U{1, J}.
    % TODO: Empty [] x parameter may cause problems.
    llambda = max(lw) + log(model.px.kappa([], theta));

    %% Rejection sampling
    l = 0;
    done = 0;
    while ~done
        % Propose sample
        alpha = randi(J, 1);
        
        % Calculate non-normalized weight, but only if we haven't done so
        % before
        if ivtilde(alpha) == 0
            lvtilde(alpha) = calculate_ancestor_weights(model, y, xt, x(:, alpha), lw(alpha), theta);
            ivtilde(alpha) = 1;
        end
        
        % Accept/reject
        u = rand(1);
        gamma = exp(lvtilde(alpha) - llambda);
        if gamma > 1
            warning('Acceptance probability larger than one, check the bounding constant.');
        end
        accepted = (u <= gamma);
        l = l+1;
        done = accepted || (l >= L);
    end
    
    %% Fallback: Categorical sampling
    if ~accepted
        % Calculate non-normalized ancestor weights for the trajectories
        % not proposed by rejection sampling, then sample an ancestor index
        lvtilde(~ivtilde) = calculate_ancestor_weights(model, y, xt, x(:, ~ivtilde), lw(~ivtilde), theta);
        ivtilde(~ivtilde) = 1;
        wtilde = exp(lvtilde-max(lvtilde));
        wtilde = wtilde/sum(wtilde);
        tmp = resample_stratified(log(wtilde));
        alpha = tmp(randi(J, 1));
        l = NaN;
    end
    
    %% Debugging
    if nargout >= 2
        % Calculate true acceptance probabilities for debugging
        % TODO: One should be able to return the stats without calculating
        % the true acceptance probabilities (which is only used for
        % debugging).
        lvtilde(~ivtilde) = calculate_ancestor_weights(model, y, xt, x(:, ~ivtilde), lw(~ivtilde), theta);
        vtilde = exp(lvtilde);
        dgamma = vtilde/max(vtilde) - vtilde/exp(llambda);

        state = struct('l', l, 'accepted', accepted, 'dgamma', dgamma);
    end
end
