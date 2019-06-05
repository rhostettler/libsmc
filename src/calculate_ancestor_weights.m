function lv = calculate_ancestor_weights(model, y, xt, x, lw, theta)
% Calculate ancestor weigths for markovian state-space models
%
% TODO:
%   * interface
%   * document
%   * comments
%   * etc.

    J = size(x, 2);
    px = model.px;
    if px.fast
        lv = lw + px.logpdf(xt*ones(1, J), x, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = lw(j) + px.logpdf(xt, x(:, j), theta);
        end
    end
end
