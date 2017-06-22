function [v, lv] = sis_incremental_weights(y, xp, x, t, model, q)
% General incremental weights for sequential importance sampling
%

    M = size(xp, 2);
    px = model.px;
    py = model.py;
    
    if px.fast && py.fast && q.fast
        lv = ( ...
            py.logpdf(y*ones(1, M), xp, t) ...
            + px.logpdf(xp, x, t) ...
            - q.logpdf(xp, y*ones(1, M), x, t) ...
        );
    else
        M = size(xp, 2);
        lv = zeros(1, M);
        for m = 1:M
            lv(m) = ( ...
                py.logpdf(y, xp(:, m), t) ...
                + px.logpdf(xp(:, m), x(:, m), t) ...
                - q.logpdf(xp(:, m), y, x(:, m), t) ...
            );
        end
    end
    v = exp(lv-max(lv));
end
