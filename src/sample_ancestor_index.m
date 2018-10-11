
%% Draw ancestor index for seed trajectory
function [alpha, state] = sample_ancestor_index(xt, x, t, lw, model)
    state = NaN;
    M = size(x, 2);
    lv = calculate_ancestor_weights(xt, x, t, lw, model.px);
    v = exp(lv-max(lv));
    v = v/sum(v);
    tmp = sysresample(v);
    alpha = tmp(randi(M, 1));
end
