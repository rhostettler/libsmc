%% 
% TODO:
%   * This is very ad-hoc right now
% 	* We should also implement an adaptive version which uses the
%     newly drawn weight in the proposal
%   * We might as well sample from the prior; in fact, we alread resampled
%     (outside of the function), thus we could sample from the prior by
%     sampling random integers, which would be as efficient.
function [alpha, accepted] = sample_ancestor_index_rs(xt, x, t, lw, model)
    M = size(x, 2);
    J = 10;
    j = 0;
    done = 0;
    lv = lw + log(model.px.rho);
    %lv = calculate_ancestor_weights(xt, x, t, lw, model.px);
    iv = zeros(1, M);
    while ~done
        % Propose sample
        alpha = randi(M, 1);
        
        % Calculate non-normalized weight, but only if we haven't done so
        % before
        if iv(alpha) == 0
            lv(alpha) = calculate_ancestor_weights(xt, x(:, alpha), t, lw(alpha), model.px);
            iv(alpha) = 1;
        end
        
        % Calculate upper bound on normalizing constant
        %rho = sum(exp(lv));
        rho = exp(max(lv));
        kappa = 1; % M / M
        
        u = rand(1);
        paccept = (exp(lv(alpha))/(kappa*rho));
        if paccept > 1
            warning('Acceptance probability larger than one, check your bounding constant.');
        end
        accepted = (u < paccept);
        
        j = j+1;
        done = accepted || (j >= J);
    end
    if ~accepted
        % Exhaustive search for the non-calculated ones
        lv(~iv) = calculate_ancestor_weights(xt, x(:, ~iv), t, lw(~iv), model.px);
        v = exp(lv-max(lv));
        v = v/sum(v);
        tmp = sysresample(v);
        alpha = tmp(randi(M, 1));
    end
end
