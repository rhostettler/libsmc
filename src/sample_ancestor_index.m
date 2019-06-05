function [alpha, state] = sample_ancestor_index(model, y, xt, x, lw, theta)
% %% Draw ancestor index for seed trajectory
%
%
% TODO:
%   * Interface
%   * Document
%   * Comments

    state = NaN;
    M = size(x, 2);
    lv = calculate_ancestor_weights(model, y, xt, x, lw, theta);
    v = exp(lv-max(lv));
    v = v/sum(v);
    tmp = sysresample(v);
    alpha = tmp(randi(M, 1));
end

