function [alpha, state] = sample_ancestor_index_rs(model, y, xt, x, lw, theta, L)
% Rejection-sampling-based ancestor index sampling for CPF-AS
%
% USAGE
%   [alpha, state] = SAMPLE_ANCESTOR_INDEX_RS(model, y, xt, x, lw, theta, L)
% 
% DESCRIPTION
%   Samples the ancestor index for the seed trajectory in the conditional
%   particle filter with ancestor sampling (CPF-AS) using rejection
%   sampling (for Markovian state-space models).
% 
% PARAMETERS
%   model   Model structure
%   y       Measurement vector
%   xtilde  Sample of the seed trajectory, xtilde[n]
%   x       Matrix of particles x[n-1]^j
%   lw      Row vector of particle log-weights log(w[n-1]^j)
%   theta   Additional parameters
%   L       Maximum number of rejection sampling trials before falling back
%           to sampling from the categorical distribution (optional,
%           default: 10)
%
% RETURNS
%   alpha   Sampled ancestor index
%   state   Sampler state (empty)
%
% AUTHOR(S)
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% TODO:
%   * Describe 'state'

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
    llambda = max(lw) + log(model.px.kappa);

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
        tmp = sysresample(wtilde);
        alpha = tmp(randi(J, 1));
        l = NaN;
    end
    
    %% Debugging
    if nargout >= 2
        % Calculate true acceptance probabilities for debugging
        lvtilde(~ivtilde) = calculate_ancestor_weights(model, y, xt, x(:, ~ivtilde), lw(~ivtilde), theta);
        vtilde = exp(lvtilde);
        dgamma = vtilde/max(vtilde) - vtilde/exp(llambda);

        state = struct('l', l, 'accepted', accepted, 'dgamma', dgamma);
    end
end
