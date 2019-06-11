function lv = calculate_incremental_weights_bootstrap2(model, y, x, theta)
% Calculate incremental weights for the bootstrap PF and non-Markovian SSMs
%
% USAGE
%   FIXME
%
% DESCRIPTION
%   FIXME
%
% PARAMETERS
%   FIXME
%
% RETURNS
%   FIXME
%
% AUTHOR
%   2017-2019 -- Roland Hostettler <roland.hostettler@angstrom.uu.se>

% TODO:
%   * Revise properly

    % Strip initial values (x[0] and t[0])
    [~, J, n] = size(x);
    py = model.py;
    if py.fast
        lv = py.logpdf(y, x, theta);
    else
        lv = zeros(1, J);
        for j = 1:J
            lv(j) = py.logpdf(y, x(:, j, :), theta);
        end
    end
end
