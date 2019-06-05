function [alpha, accepted] = sample_ancestor_index_rs(xt, x, t, lw, model)
% Rejection-sampling-based ancesotr sampling for particle Gibbs
%
% USAGE
%
% 
% DESCRIPTION
%
% 
% PARAMETERS
%
% 
% RETURNS
%
% AUTHOR(S)
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% TODO:
%   * Accepted should maybe include more status info or something; need to
%     check how I am going to return this to the user.
% 	* We should also implement an adaptive version which uses the
%     newly drawn weight in the proposal
%     => How is this supposed to work?
%   * Document
%   * Clean up code

    %% Defaults
    
    %% 

    J = size(x, 2);
    L = 10;
    l = 0;
    done = 0;
    lv = lw + log(model.px.kappa);
    llambda = max(lv);  % Bounding constant with normalization constant removed. Assumes uniform q(alpha)
    
    iv = zeros(1, J);
    
    while ~done
        % Propose sample
        alpha = randi(J, 1);
        
        % Calculate non-normalized weight, but only if we haven't done so
        % before
        if iv(alpha) == 0
            lv(alpha) = calculate_ancestor_weights(xt, x(:, alpha), t, lw(alpha), model.px);
            iv(alpha) = 1;
        end
        
        % Calculate upper bound on normalizing constant
        u = rand(1);
        gamma = exp(lv(alpha) - llambda);
        if gamma > 1
            warning('Acceptance probability larger than one, check your bounding constant.');
        end
        accepted = (u <= gamma);
        l = l+1;
        done = accepted || (l >= L);
    end
    if ~accepted
        % Exhaustive search for the non-calculated ones
        lv(~iv) = calculate_ancestor_weights(xt, x(:, ~iv), t, lw(~iv), model.px);
        v = exp(lv-max(lv));
        v = v/sum(v);
        tmp = sysresample(v);
        alpha = tmp(randi(J, 1));
    end
end
